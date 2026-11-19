//! Time-blocked stencils: many steps of `u = step(u)` per pass over memory.
//!
//! [`Array::iterate`](crate::Array::iterate) calls the step once on a placeholder array and
//! inspects the expression it returns. If that is one fused elementwise expression of the
//! placeholder, its shifts (small offsets) and arrays that do not change, every launch loads a
//! tile of `u` with a halo into workgroup memory, runs several steps there (each step
//! shrinks the valid part of the tile by the stencil radius) and writes the tile back once.
//! Shifts are circular, so halo loads wrap around the array like the shifts do.

use std::sync::Arc;

use crate::array::{Array, Failure, Storage};
use crate::context::{Context, RawArg};
use crate::error::LumaError;
use crate::ops::fold_workgroups;
use crate::shape::Shape;

use luma_core::kir::{BarrierScope, BuiltinVar, KernelModule, KirType, ModuleBuilder};

use crate::gpu::lower::finish;

use super::kirgen::{Frame, Lower, Uni, helper_library, li, lu, reg};
use super::program::{flatten_stencil, Fused, Instr, Ref};
use super::{ElemTy, OpKind, ShiftKind};
use crate::tuning::{stencil_bytes, Tuning, STENCIL_RADIUS as MAX_RADIUS};

/// Invocations per workgroup.
pub(super) const WORKGROUP: u32 = 256;

/// A step expression that can run time-blocked.
pub(crate) struct Stencil {
    instrs: Vec<Instr>,
    /// Inputs of the expression; the state inputs are iterated, the others stay fixed.
    inputs: Vec<(Arc<Storage>, ElemTy)>,
    /// The state (of `state_count`) each input reads, `None` for a fixed input.
    state_of: Vec<Option<usize>>,
    state_count: usize,
    scalar_tys: Vec<ElemTy>,
    scalars: Vec<u32>,
    /// State, read offset (x, y) of every shift instruction, by instruction index.
    offsets: Vec<Option<(usize, i32, i32)>>,
    radius: (u32, u32),
    /// The instruction computing the next value of each state.
    outs: Vec<u32>,
    dims: (u32, u32),
}

/// `s` in `0..dim` as the signed shift it stands for.
pub(super) fn signed(s: u32, dim: u32) -> i32 {
    if s > dim / 2 {
        s as i32 - dim as i32
    } else {
        s as i32
    }
}

/// The step `next = step(probe)` as a stencil, if it is one.
pub(crate) fn plan(probe: &Array<f32>, next: &Array<f32>) -> Option<Stencil> {
    let shape = probe.shape();
    let dims = shape.dims();
    if dims[2] != 1 || dims[3] != 1 || next.shape() != shape {
        return None;
    }
    let probe = Arc::clone(probe.storage().ok()?);
    // Tiles load circular halos from one buffer: arrays spanning several buffers take the
    // step-by-step path.
    if probe.segments() > 1 {
        return None;
    }
    // If `step(&probe)` bound `probe`'s buffer (eager ops or forced sub-expressions), it is not
    // a single fused stencil expression.
    if probe.was_materialized() {
        return None;
    }
    let node = next.pending()?;
    let args = node.operands()?;
    let fused = flatten_stencil(node, args, probe.id()).ok()?;
    if probe.was_materialized() {
        return None;
    }
    let mut state_of = vec![None; fused.inputs.len()];
    if let Some(at) = fused.inputs.iter().position(|input| Arc::ptr_eq(input, &probe)) {
        state_of[at] = Some(0);
    }
    let outs = vec![*fused.key.outputs.first()?];
    plan_inner(fused, state_of, 1, outs, shape)
}

/// The store program `fused` (one `f32` output per state) as a stencil on arrays of `shape`, if
/// it is one. Inputs `0..state_count` are the states; the others stay fixed.
pub(super) fn plan_multi(fused: Fused, state_count: usize, shape: Shape) -> Option<Stencil> {
    if !(1..=4).contains(&state_count)
        || fused.key.outputs.len() != state_count
        || fused.key.input_tys.get(..state_count).is_none_or(|tys| tys.iter().any(|&ty| ty != ElemTy::F32))
    {
        return None;
    }
    let state_of = (0..fused.inputs.len()).map(|k| (k < state_count).then_some(k)).collect();
    let outs = fused.key.outputs.to_vec();
    plan_inner(fused, state_of, state_count, outs, shape)
}

fn plan_inner(fused: Fused, state_of: Vec<Option<usize>>, state_count: usize, outs: Vec<u32>, shape: Shape) -> Option<Stencil> {
    let dims = shape.dims();
    if dims[2] != 1 || dims[3] != 1 {
        return None;
    }
    let (w, h) = (u32::try_from(dims[0]).ok()?, u32::try_from(dims[1]).ok()?);
    if !fused.key.subs.is_empty() || !fused.key.narrow() || super::codegen::generates(&fused.key) {
        return None;
    }
    if outs.iter().any(|&out| fused.key.instrs.get(out as usize).is_none_or(|instr| instr.op.out_ty() != ElemTy::F32)) {
        return None;
    }
    let mut offsets = Vec::with_capacity(fused.key.instrs.len());
    let mut radius = (0u32, 0u32);
    for instr in fused.key.instrs.iter() {
        let OpKind::Shift(kind) = instr.op.kind else {
            offsets.push(None);
            continue;
        };
        // Only shifts of a state itself: a shifted fixed array would need halos of its own.
        let Ref::Input(k) = instr.args[0] else {
            return None;
        };
        let s = (*state_of.get(k as usize)?)?;
        let (dx, dy) = match kind {
            ShiftKind::Dims { dims: s_dims, shifts } if s_dims == [w, h, 1, 1] => {
                (signed(shifts[0], w), signed(shifts[1], h))
            }
            ShiftKind::Dims { .. } | ShiftKind::View { .. } | ShiftKind::Take { .. } | ShiftKind::ParamView { .. } | ShiftKind::ParamTake { .. } | ShiftKind::ParamDims { .. } => return None,
            ShiftKind::Linear => {
                let Ref::Scalar(k) = instr.args[1] else {
                    return None;
                };
                let amount = *fused.scalars.get(k as usize)?;
                if h == 1 {
                    (signed(amount, w), 0)
                } else if amount.is_multiple_of(w) {
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
        offsets.push(Some((s, ox, oy)));
    }
    if radius.0 > MAX_RADIUS || radius.1 > MAX_RADIUS || (h == 1 && radius.1 > 0) {
        return None;
    }
    fused.mark_fused();
    let inputs = fused
        .inputs
        .iter()
        .cloned()
        .zip(fused.key.input_tys.iter().copied())
        .collect();
    Some(Stencil {
        instrs: fused.key.instrs.to_vec(),
        inputs,
        state_of,
        state_count,
        scalar_tys: fused.key.scalar_tys,
        scalars: fused.scalars,
        offsets,
        radius,
        outs,
        dims: (w, h),
    })
}

/// Output tile of a launch and the steps it runs, for `radius` and `state_count` states: the
/// tile shrinks until one step fits the tuning's workgroup memory budget (the two buffers of
/// every state, outputs plus halo), then as many steps as fit (at most 8) are blocked. `None`
/// if not even the smallest tile fits.
fn layout(tuning: &Tuning, radius: (u32, u32), one_d: bool, steps: usize, state_count: usize) -> Option<((u32, u32), u32)> {
    let (mut tx, mut ty) = if one_d { (tuning.stencil_tile_1d, 1) } else { tuning.stencil_tile_2d };
    let fits = |(tx, ty): (u32, u32), k: u32| state_count as u32 * stencil_bytes((tx, ty), (k * radius.0, k * radius.1)) <= tuning.stencil_tile_bytes;
    while !fits((tx, ty), 1) {
        if one_d || tx >= ty {
            tx /= 2;
        } else {
            ty /= 2;
        }
        if tx < 4 || (!one_d && ty < 4) {
            return None;
        }
    }
    // A pointwise step needs no halo; cap it anyway, the program is unrolled per step.
    let mut k = 1;
    while k < 8 && fits((tx, ty), k + 1) {
        k += 1;
    }
    Some(((tx, ty), k.min(steps.max(1) as u32)))
}

impl Stencil {
    /// Whether this stencil runs on `ctx`: its buffers fit the binding limit and a tile of one
    /// step fits workgroup memory.
    pub(super) fn supported(&self, ctx: &Context) -> bool {
        let fixed = self.state_of.iter().filter(|state| state.is_none()).count();
        2 * self.state_count + fixed <= ctx.jit().max_inputs() + 1
            && layout(ctx.tuning(), self.radius, self.dims.1 == 1, 1, self.state_count).is_some()
    }

    /// What the module of `k` steps on tiles of `tile` outputs depends on, to cache its kernel by.
    fn cache_key(&self, tile: (u32, u32), k: u32) -> String {
        let (lw, lh) = (tile.0 + 2 * k * self.radius.0, tile.1 + 2 * k * self.radius.1);
        let tys: Vec<ElemTy> = self.inputs.iter().map(|(_, ty)| *ty).collect();
        format!(
            "stencil {:?} {tys:?} {:?} {} {:?} {:?} {:?} {} {} {tile:?} {k} {}",
            self.instrs,
            self.state_of,
            self.state_count,
            self.scalar_tys,
            self.offsets,
            self.outs,
            self.radius.0,
            self.radius.1,
            self.dims.0 >= lw && (self.dims.1 == 1 || self.dims.1 >= lh),
        )
    }

    /// Kernel IR running `k` steps on tiles of `tile` outputs.
    fn module(&self, tile: (u32, u32), k: u32, tuning: &Tuning) -> Result<KernelModule, LumaError> {
        let (tx, ty) = tile;
        let m = self.state_count;
        let (hx, hy) = (k * self.radius.0, k * self.radius.1);
        let (lw, lh) = (tx + 2 * hx, ty + 2 * hy);
        let cells = lw * lh;
        // A tile and its halo fit in the dimension, so that indices stay within one period.
        let small = self.dims.0 >= lw && (self.dims.1 == 1 || self.dims.1 >= lh);
        let fixed: Vec<usize> = (0..self.inputs.len()).filter(|&i| self.state_of[i].is_none()).collect();
        let lib = helper_library(&[self.instrs.as_slice()], tuning, false, false)?;
        let mut mb = ModuleBuilder::new(format!("Stencil({k} steps x {m} states) f32"), "main", [WORKGROUP, 1, 1]);
        let mut env = Lower::new(lib.clone(), &[]);
        let name = |base: &str, s: usize| if m == 1 { base.to_owned() } else { format!("{base}{s}") };
        let outs: Vec<_> = (0..m).map(|s| mb.add_storage_read_write(&name("out", s), KirType::F32)).collect();
        let us: Vec<_> = (0..m).map(|s| mb.add_storage_read(&name("u", s), KirType::F32)).collect();
        env.ins = vec![None; self.inputs.len()];
        for &i in &fixed {
            env.ins[i] = Some(mb.add_storage_read(&format!("in{i}"), reg(self.inputs[i].1)));
        }
        let mut fields: Vec<(String, KirType)> = vec![
            ("w".to_owned(), KirType::U32),
            ("h".to_owned(), KirType::U32),
            ("tiles_x".to_owned(), KirType::U32),
        ];
        for (s, &ty) in self.scalar_tys.iter().enumerate() {
            fields.push((format!("s{s}"), reg(ty)));
        }
        let uni = Uni::declare(&mut mb, fields);
        env.set_uniform(uni);
        let b0: Vec<_> = (0..m).map(|s| mb.add_shared(&format!("b0_{s}"), KirType::F32, cells)).collect();
        let b1: Vec<_> = (0..m).map(|s| mb.add_shared(&format!("b1_{s}"), KirType::F32, cells)).collect();
        env.declare_wrap(&mut mb, small);
        {
            let mut fb = mb.entry_builder();
            let lid = fb.builtin(BuiltinVar::LocalInvocationIndex);
            let wid = fb.builtin(BuiltinVar::WorkgroupId);
            let nwg = fb.builtin(BuiltinVar::NumWorkgroups);
            let (wx, wy, nx) = (fb.field(wid, 0, KirType::U32), fb.field(wid, 1, KirType::U32), fb.field(nwg, 0, KirType::U32));
            let (pw, ph, ptiles) = (env.named(&fb, "w"), env.named(&fb, "h"), env.named(&fb, "tiles_x"));
            let g = fb.let_val("g", wx + wy * nx);
            let tile_x = fb.let_val("tile_x", (g % ptiles) * lu(&fb, tx));
            let tile_y = fb.let_val("tile_y", (g / ptiles) * lu(&fb, ty));
            let ox = fb.let_val("ox", fb.convert(tile_x, KirType::I32) - li(&fb, hx as i32));
            let oy = fb.let_val("oy", fb.convert(tile_y, KirType::I32) - li(&fb, hy as i32));
            let env = &env;
            fb.for_range("e", lid, lu(&fb, cells), lu(&fb, WORKGROUP), |b, e| {
                let xs = ox + b.convert(e % lu(b, lw), KirType::I32);
                let ys = oy + b.convert(e / lu(b, lw), KirType::I32);
                let at = env.wrap(b, xs, pw) + env.wrap(b, ys, ph) * pw;
                for s in 0..m {
                    let value = b.load(us[s], at);
                    b.shared_store(b0[s], e, value);
                }
            });
            fb.barrier(BarrierScope::Workgroup);
            for step in 1..=k {
                let (from, to) = if step % 2 == 1 { (&b0, &b1) } else { (&b1, &b0) };
                let (x0, y0) = (step * self.radius.0, step * self.radius.1);
                let (rw, rh) = (lw - 2 * x0, lh - 2 * y0);
                let fixed = &fixed;
                fb.for_range("e", lid, lu(&fb, rw * rh), lu(&fb, WORKGROUP), |b, e| {
                    let lx = b.let_val("lx", lu(b, x0) + e % lu(b, rw));
                    let ly = b.let_val("ly", lu(b, y0) + e / lu(b, rw));
                    let c = b.let_val("c", lx + ly * lu(b, lw));
                    let gi = if fixed.is_empty() {
                        lu(b, 0)
                    } else {
                        let xs = ox + b.convert(lx, KirType::I32);
                        let ys = oy + b.convert(ly, KirType::I32);
                        let at = env.wrap(b, xs, pw) + env.wrap(b, ys, ph) * pw;
                        b.let_val("gi", at)
                    };
                    let mut frame = Frame::with_len(self.instrs.len());
                    for (i, (_, ty)) in self.inputs.iter().enumerate() {
                        let value = match self.state_of[i] {
                            Some(s) => b.shared_load(from[s], c),
                            None => env.load_in(b, *ty, env.ins[i].unwrap_or(0), gi),
                        };
                        let value = b.let_val(&format!("a{i}"), value);
                        frame.a.insert((i as u32, 0), value);
                    }
                    for (t, instr) in self.instrs.iter().enumerate() {
                        let value = match self.offsets[t] {
                            Some((s, dx, dy)) => {
                                let centre = b.bitcast(c, KirType::I32);
                                let at = b.bitcast(centre + li(b, dx + dy * lw as i32), KirType::U32);
                                b.shared_load(from[s], at)
                            }
                            None => {
                                let index = lu(b, 0);
                                env.expression(b, &frame, instr, index, false, false)
                            }
                        };
                        let value = b.let_val(&format!("t{t}"), value);
                        frame.t[t] = Some(value);
                    }
                    for s in 0..m {
                        match frame.t(self.outs[s] as usize) {
                            Some(value) => b.shared_store(to[s], c, value),
                            None => {
                                env.fail(b, "the stencil has no output instruction");
                            }
                        }
                    }
                });
                fb.barrier(BarrierScope::Workgroup);
            }
            let last = if k % 2 == 1 { &b1 } else { &b0 };
            fb.for_range("e", lid, lu(&fb, tx * ty), lu(&fb, WORKGROUP), |b, e| {
                let lx = e % lu(b, tx);
                let ly = e / lu(b, tx);
                let x = b.let_val("x", tile_x + lx);
                let y = b.let_val("y", tile_y + ly);
                let inside = b.and(b.lt(x, pw), b.lt(y, ph));
                b.if_then(inside, |b| {
                    let cell = (lu(b, hx) + lx) + (lu(b, hy) + ly) * lu(b, lw);
                    for s in 0..m {
                        let value = b.shared_load(last[s], cell);
                        b.store(outs[s], x + y * pw, value);
                    }
                });
            });
            fb.finish();
        }
        if let Some(message) = env.error() {
            return Err(LumaError::Gpu(format!("stencil kernel: {message}")));
        }
        Ok(finish(mb, lib.as_deref()))
    }

    /// Runs `steps` steps starting from `start`, as many per launch as the tile allows.
    pub(crate) fn run(&self, start: &Array<f32>, steps: usize) -> Result<Array<f32>, Failure> {
        let init = Arc::clone(start.storage()?);
        let mut states = self.run_multi(start.context(), &[init], steps)?;
        let state = states.pop().ok_or_else(|| Failure::New(LumaError::Gpu("stencil ran no steps".into())))?;
        Ok(Array::from_storage(start.context(), start.shape(), state))
    }

    /// Runs `steps` steps of all states from `inits`, as many per launch as the tiles allow.
    pub(super) fn run_multi(&self, ctx: &Context, inits: &[Arc<Storage>], steps: usize) -> Result<Vec<Arc<Storage>>, Failure> {
        let jit = ctx.jit();
        let (w, h) = self.dims;
        if inits.len() != self.state_count {
            return Err(Failure::New(LumaError::Gpu("stencil states do not match its program".into())));
        }
        let (tile, per_launch) = layout(ctx.tuning(), self.radius, h == 1, steps, self.state_count)
            .ok_or_else(|| Failure::New(LumaError::Gpu("stencil tile does not fit workgroup memory".into())))?;
        let tiles = [w.div_ceil(tile.0), h.div_ceil(tile.1)];
        let groups = fold_workgroups(ctx, tiles[0] * tiles[1])?;
        let mut params = vec![w, h, tiles[0]];
        params.extend_from_slice(&self.scalars);
        let fixed: Vec<&Arc<Storage>> = self
            .inputs
            .iter()
            .enumerate()
            .filter(|(i, _)| self.state_of[*i].is_none())
            .map(|(_, (s, _))| s)
            .collect();

        let mut current: Vec<Arc<Storage>> = inits.to_vec();
        let mut done = 0;
        while done < steps {
            let k = per_launch.min((steps - done) as u32);
            let core = jit.module_kernel(ctx, self.cache_key(tile, k), &format!("Stencil({k} steps x {} states) f32", self.state_count), || {
                self.module(tile, k, ctx.tuning())
            })?;
            let next: Vec<Arc<Storage>> = current
                .iter()
                .map(|state| Arc::new(Storage::new(ctx, state.len(), ElemTy::F32.layout())))
                .collect();
            {
                let dst_pins = next.iter().map(|state| state.pin_single()).collect::<Result<Vec<_>, _>>()?;
                let src_pins = current.iter().map(|state| state.pin_single()).collect::<Result<Vec<_>, _>>()?;
                let fixed_pins = fixed.iter().map(|input| input.pin_single()).collect::<Result<Vec<_>, _>>()?;
                let mut raw = Vec::with_capacity(2 * self.state_count + fixed.len() + 1);
                for (pin, state) in dst_pins.iter().zip(&next) {
                    raw.push(RawArg::write(pin.buffer(), state.byte_len()));
                }
                for (pin, state) in src_pins.iter().zip(&current) {
                    raw.push(RawArg::read(pin.buffer(), state.byte_len()));
                }
                for (pin, input) in fixed_pins.iter().zip(&fixed) {
                    raw.push(RawArg::read(pin.buffer(), input.byte_len()));
                }
                raw.push(RawArg::Uniform(bytemuck::cast_slice(&params)));
                ctx.launch(&core, groups, [w, h, 1], &raw)?;
            }
            current = next;
            done += k as usize;
        }
        if done == 0 {
            return Err(Failure::New(LumaError::Gpu("stencil ran no steps".into())));
        }
        Ok(current)
    }
}
