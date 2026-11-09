//! Time-blocked stencils: many steps of `u = step(u)` per pass over memory.
//!
//! [`Array::iterate`](crate::Array::iterate) calls the step once on a placeholder array and
//! inspects the expression it returns. If that is one fused elementwise expression of the
//! placeholder, its shifts (small offsets) and arrays that do not change, every launch loads a
//! tile of `u` with a halo into workgroup memory, runs several steps there (each step
//! shrinks the valid part of the tile by the stencil radius) and writes the tile back once.
//! Shifts are circular, so halo loads wrap around the array like the shifts do.

use std::fmt::Write;
use std::sync::Arc;

use crate::array::{Array, Failure, Storage};
use crate::context::RawArg;
use crate::error::LumaError;
use crate::ops::fold_workgroups;

use super::codegen::expression;
use super::program::{flatten, Instr, Ref, Tail};
use super::{ElemTy, OpKind, ShiftKind};

/// Invocations per workgroup.
const WORKGROUP: u32 = 256;
/// Workgroup memory for the two tile buffers; more costs occupancy on Apple GPUs.
const TILE_BYTES: u32 = 8 * 1024;

/// Largest stencil radius (per axis) handled; wider stencils iterate one kernel per step.
const MAX_RADIUS: u32 = 4;

/// A step expression that can run time-blocked.
pub(crate) struct Stencil {
    instrs: Vec<Instr>,
    /// Inputs of the expression; `u` is the iterated one, the others stay fixed.
    inputs: Vec<(Arc<Storage>, ElemTy)>,
    u: Option<u32>,
    scalar_tys: Vec<ElemTy>,
    scalars: Vec<u32>,
    /// Read offset (x, y) of every shift instruction, by instruction index.
    offsets: Vec<Option<(i32, i32)>>,
    radius: (u32, u32),
    dims: (u32, u32),
}

/// `s` in `0..dim` as the signed shift it stands for.
fn signed(s: u32, dim: u32) -> i32 {
    if s > dim / 2 {
        s as i32 - dim as i32
    } else {
        s as i32
    }
}

/// The step `next = step(probe)` as a stencil, if it is one.
pub(crate) fn plan(probe: &Array<f32>, next: &Array<f32>) -> Option<Stencil> {
    let dims = probe.shape().dims();
    if dims[2] != 1 || dims[3] != 1 || next.shape() != probe.shape() {
        return None;
    }
    let (w, h) = (u32::try_from(dims[0]).ok()?, u32::try_from(dims[1]).ok()?);
    let probe = Arc::clone(probe.storage().ok()?);
    let node = next.pending()?;
    let args = node.operands()?;
    let fused = flatten(&[(Arc::clone(node), args)], Tail::Store).ok()?;
    let out = *fused.key.outputs.first()?;
    if fused.key.instrs.get(out as usize)?.op.out_ty() != ElemTy::F32 {
        return None;
    }
    let u = fused
        .inputs
        .iter()
        .position(|input| Arc::ptr_eq(input, &probe))
        .map(|at| at as u32);
    let mut offsets = Vec::with_capacity(fused.key.instrs.len());
    let mut radius = (0u32, 0u32);
    for instr in &fused.key.instrs {
        let OpKind::Shift(kind) = instr.op.kind else {
            offsets.push(None);
            continue;
        };
        // Only shifts of `u` itself: a shifted fixed array would need halos of its own.
        if instr.args[0] != Ref::Input(u?) {
            return None;
        }
        let (dx, dy) = match kind {
            ShiftKind::Dims { shifts, .. } => (signed(shifts[0], w), signed(shifts[1], h)),
            ShiftKind::Linear => {
                let Ref::Scalar(k) = instr.args[1] else {
                    return None;
                };
                let amount = *fused.scalars.get(k as usize)?;
                if h == 1 {
                    (signed(amount, w), 0)
                } else if amount % w == 0 {
                    (0, signed(amount / w, h))
                } else {
                    return None;
                }
            }
        };
        // Element `c` of a shift is `u[c - shift]`.
        let (ox, oy) = (-dx, -dy);
        radius = (
            radius.0.max(ox.unsigned_abs()),
            radius.1.max(oy.unsigned_abs()),
        );
        offsets.push(Some((ox, oy)));
    }
    if radius.0 > MAX_RADIUS || radius.1 > MAX_RADIUS || (h == 1 && radius.1 > 0) {
        return None;
    }
    let inputs = fused
        .inputs
        .iter()
        .cloned()
        .zip(fused.key.input_tys.iter().copied())
        .collect();
    Some(Stencil {
        instrs: fused.key.instrs,
        inputs,
        u,
        scalar_tys: fused.key.scalar_tys,
        scalars: fused.scalars,
        offsets,
        radius,
        dims: (w, h),
    })
}

/// Output tile of a launch and the steps it runs, for `radius`: the most steps whose two tile
/// buffers (outputs plus halo) fit in [`TILE_BYTES`].
fn layout(radius: (u32, u32), one_d: bool, steps: usize) -> ((u32, u32), u32) {
    let (tx, ty) = if one_d { (1024, 1) } else { (32, 16) };
    let fits = |k: u32| (tx + 2 * k * radius.0) * (ty + 2 * k * radius.1) * 8 <= TILE_BYTES;
    // A pointwise step needs no halo; cap it anyway, the program is unrolled per step.
    let mut k = 1;
    while k < 8 && fits(k + 1) {
        k += 1;
    }
    ((tx, ty), k.min(steps.max(1) as u32))
}

impl Stencil {
    /// WGSL running `k` steps on tiles of `tile` outputs.
    fn source(&self, tile: (u32, u32), k: u32) -> String {
        let (tx, ty) = tile;
        let (hx, hy) = (k * self.radius.0, k * self.radius.1);
        let (lw, lh) = (tx + 2 * hx, ty + 2 * hy);
        let fixed: Vec<usize> = (0..self.inputs.len())
            .filter(|&i| Some(i as u32) != self.u)
            .collect();
        let mut src = String::from("struct Params { w: u32, h: u32, tiles_x: u32, ");
        for (k, ty) in self.scalar_tys.iter().enumerate() {
            let _ = write!(src, "s{k}: {}, ", ty.wgsl());
        }
        src.push_str("}\n");
        src.push_str("@group(1) @binding(0) var<storage, read_write> out: array<f32>;\n");
        src.push_str("@group(1) @binding(1) var<storage, read> u: array<f32>;\n");
        for (b, &i) in fixed.iter().enumerate() {
            let _ = writeln!(
                src,
                "@group(1) @binding({}) var<storage, read> in{i}: array<{}>;",
                b + 2,
                self.inputs[i].1.wgsl()
            );
        }
        let _ = writeln!(
            src,
            "@group(1) @binding({}) var<uniform> p: Params;",
            fixed.len() + 2
        );
        let _ = write!(
            src,
            "var<workgroup> b0: array<f32, {cells}>;
var<workgroup> b1: array<f32, {cells}>;

fn wrap(v: i32, n: u32) -> u32 {{ let m = i32(n); return u32(((v % m) + m) % m); }}

@compute @workgroup_size({WORKGROUP})
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let g = wid.x + wid.y * nwg.x;
    let tile_x = (g % p.tiles_x) * {tx}u;
    let tile_y = (g / p.tiles_x) * {ty}u;
    let ox = i32(tile_x) - {hx};
    let oy = i32(tile_y) - {hy};
    for (var e = lid; e < {cells}u; e = e + {WORKGROUP}u) {{
        b0[e] = u[wrap(ox + i32(e % {lw}u), p.w) + wrap(oy + i32(e / {lw}u), p.h) * p.w];
    }}
    workgroupBarrier();
",
            cells = lw * lh,
        );
        let out = self.instrs.len() - 1;
        for step in 1..=k {
            let (from, to) = if step % 2 == 1 {
                ("b0", "b1")
            } else {
                ("b1", "b0")
            };
            let (x0, y0) = (step * self.radius.0, step * self.radius.1);
            let (rw, rh) = (lw - 2 * x0, lh - 2 * y0);
            let _ = write!(
                src,
                "    for (var e = lid; e < {cells}u; e = e + {WORKGROUP}u) {{
        let lx = {x0}u + e % {rw}u;
        let ly = {y0}u + e / {rw}u;
        let c = lx + ly * {lw}u;
",
                cells = rw * rh
            );
            if !fixed.is_empty() {
                src.push_str(
                    "        let gi = wrap(ox + i32(lx), p.w) + wrap(oy + i32(ly), p.h) * p.w;\n",
                );
            }
            for (i, (_, ty)) in self.inputs.iter().enumerate() {
                let load = if Some(i as u32) == self.u {
                    format!("{from}[c]")
                } else {
                    format!("in{i}[gi]")
                };
                let _ = writeln!(src, "        let a{i}: {} = {load};", ty.wgsl());
            }
            for (t, instr) in self.instrs.iter().enumerate() {
                let value = match self.offsets[t] {
                    Some((dx, dy)) => format!("{from}[u32(i32(c) + {})]", dx + dy * lw as i32),
                    None => expression(instr),
                };
                let _ = writeln!(
                    src,
                    "        let t{t}: {} = {value};",
                    instr.op.out_ty().wgsl()
                );
            }
            let _ = write!(
                src,
                "        {to}[c] = t{out};\n    }}\n    workgroupBarrier();\n"
            );
        }
        let last = if k % 2 == 1 { "b1" } else { "b0" };
        let _ = write!(
            src,
            "    for (var e = lid; e < {cells}u; e = e + {WORKGROUP}u) {{
        let x = tile_x + e % {tx}u;
        let y = tile_y + e / {tx}u;
        if (x < p.w && y < p.h) {{ out[x + y * p.w] = {last}[({hx}u + e % {tx}u) + ({hy}u + e / {tx}u) * {lw}u]; }}
    }}
}}
",
            cells = tx * ty,
        );
        src
    }

    /// Runs `steps` steps starting from `start`, as many per launch as the tile allows.
    pub(crate) fn run(&self, start: &Array<f32>, steps: usize) -> Result<Array<f32>, Failure> {
        let ctx = start.context();
        let jit = ctx.jit();
        let (w, h) = self.dims;
        let (tile, per_launch) = layout(self.radius, h == 1, steps);
        let tiles = [w.div_ceil(tile.0), h.div_ceil(tile.1)];
        let groups = fold_workgroups(ctx, tiles[0] * tiles[1])?;
        let mut params = vec![w, h, tiles[0]];
        params.extend_from_slice(&self.scalars);
        let fixed: Vec<&Arc<Storage>> = self
            .inputs
            .iter()
            .enumerate()
            .filter(|(i, _)| Some(*i as u32) != self.u)
            .map(|(_, (s, _))| s)
            .collect();

        let mut current: Option<Array<f32>> = None;
        let mut done = 0;
        while done < steps {
            let k = per_launch.min((steps - done) as u32);
            let core = jit.stencil_kernel(
                ctx,
                self.source(tile, k),
                &format!("Stencil({k} steps) f32"),
            )?;
            let next = Array::<f32>::alloc(ctx, start.shape())?;
            {
                let from = current.as_ref().unwrap_or(start);
                let (src, bytes) = from.ready()?;
                let (dst, _) = next.ready()?;
                let mut raw = Vec::with_capacity(fixed.len() + 3);
                raw.push(RawArg::write(dst, bytes));
                raw.push(RawArg::read(src, bytes));
                for input in &fixed {
                    raw.push(RawArg::read(
                        input.buffer(),
                        u64::from(w) * u64::from(h) * 4,
                    ));
                }
                raw.push(RawArg::Uniform(bytemuck::cast_slice(&params)));
                ctx.launch(&core, groups, [w, h, 1], &raw)?;
            }
            current = Some(next);
            done += k as usize;
        }
        current.ok_or_else(|| Failure::New(LumaError::Gpu("stencil ran no steps".into())))
    }
}
