//! The tiled transpose: a store program that only moves the elements of one materialized array
//! through an axis permutation runs as [`Builtin::Transpose`](crate::builtin::Builtin::Transpose)
//! instead of the generic kernel, which reads or writes with a stride whenever the permutation
//! moves the fastest axis.
//!
//! Workgroups of 32 x 8 invocations move tiles of 32 x 32 elements (4 each): the tile is loaded
//! into workgroup memory with consecutive invocations reading consecutive elements of the
//! source's axis 0, and written with consecutive invocations writing consecutive elements of the
//! output's axis 0. The other two axes are batch axes, folded with the tile coordinates into the
//! workgroup index.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::array::Storage;
use crate::builtin::{Builtin, UnOp};
use crate::context::{Context, RawArg};
use crate::error::LumaError;
use crate::ops::{fold_workgroups, len_u32};

use super::program::{FusedKey, Ref, Tail};
use super::{Edge, ElemTy, OpKind, ShiftKind};

/// Edge of a tile, in elements.
const TILE: u32 = 32;

/// A store program that [`plan`] sends to the transpose kernel.
pub(super) struct Plan {
    ty: ElemTy,
    /// The words of the kernel's uniform (see [`source`]).
    params: [u32; 12],
    /// Workgroups of the launch, before folding into a grid.
    tiles: u32,
}

/// The WGSL type the kernel moves: any 4-byte element is a `u32` (a copy of the bits), a `Df32`
/// its two words.
fn word_ty(ty: ElemTy) -> Option<&'static str> {
    match ty {
        ElemTy::F32 | ElemTy::U32 | ElemTy::I32 => Some("u32"),
        ElemTy::Df32 => Some("vec2<f32>"),
        ElemTy::Bool | ElemTy::F64 => None,
    }
}

/// Whether `key` over `inputs`, writing `len` elements, is exactly a materialized permutation of
/// the axes of one input whose output axis 0 is not the input's axis 0:
///
/// - the tail is a plain store of one output and the program one instruction, optionally followed
///   by the copy of its value that is the output (no sub-programs, no in-place input), reading input 0 through a [`ShiftKind::View`] with [`Edge::None`] whose
///   output type is the input's, of a type of 4 bytes (`f32`, `u32`, `i32`) or a `Df32`;
/// - every source axis starts at 0 with step 1 (no offset, no stride other than the
///   permutation), `axes[a].from` is a permutation of the 4 axes with `out[axes[a].from] ==
///   src[a]`, and the shapes multiply to `len`;
/// - `axes[0].from != 0` (the fastest output axis reads another source axis than axis 0) and the
///   tile is not degenerate: both exchanged axes have more than one element;
/// - the input is one buffer of its own (not a window of a pack) and so is the output.
pub(super) fn plan(ctx: &Context, key: &FusedKey, inputs: &[Arc<Storage>], len: usize) -> Option<Plan> {
    // The view, then at most a copy of it (what `x.transpose() * 1.0` simplifies to).
    let (instr, last) = match key.instrs.as_slice() {
        [view] => (view, 0),
        [view, copy] if matches!(copy.op.kind, OpKind::Unary(UnOp::Id)) && copy.op.ty == view.op.ty && copy.args[0] == Ref::Tmp(0) => (view, 1),
        _ => return None,
    };
    let [input] = inputs else { return None };
    let OpKind::Shift(ShiftKind::View { map, edge: Edge::None }) = instr.op.kind else { return None };
    let ty = instr.op.out_ty();
    word_ty(ty)?;
    if key.tail != Tail::Store
        || key.outputs.as_slice() != [last]
        || !key.subs.is_empty()
        || !key.in_place.is_empty()
        || instr.args[0] != Ref::Input(0)
        || key.input_tys.as_slice() != [ty]
        || input.layout() != ty.layout()
        || input.segments() != 1
        || input.is_window()
        || len as u64 > ctx.segment_bytes() / ty.stride()
        || map.out_len() != len
        || input.len() != map.src.iter().map(|&d| d as usize).product::<usize>()
    {
        return None;
    }
    let mut seen = [false; 4];
    for (a, axis) in map.axes.iter().enumerate() {
        let from = axis.from as usize;
        if axis.start != 0 || axis.step != 1 || from >= 4 || std::mem::replace(&mut seen[from], true) || map.out[from] != map.src[a] {
            return None;
        }
    }
    let q = map.axes[0].from as usize;
    if q == 0 {
        return None;
    }
    // The source axis the output's axis 0 reads, and the two batch axes.
    let p = (1..4).find(|&a| map.axes[a].from == 0)?;
    let [r1, r2] = {
        let mut rest = (1..4).filter(|&a| a != p);
        [rest.next()?, rest.next()?]
    };
    let (a, b) = (map.src[0], map.src[p]);
    if a < 2 || b < 2 {
        return None;
    }
    let stride = |dims: &[u32; 4], axis: usize| -> u32 { dims[..axis].iter().product() };
    let (tx, ty_tiles) = (a.div_ceil(TILE), b.div_ceil(TILE));
    let tiles = u32::try_from(u64::from(tx) * u64::from(ty_tiles) * u64::from(map.src[r1]) * u64::from(map.src[r2])).ok()?;
    let params = [
        a,
        b,
        tx,
        ty_tiles,
        map.src[r1],
        stride(&map.src, p),
        stride(&map.src, r1),
        stride(&map.src, r2),
        stride(&map.out, q),
        stride(&map.out, map.axes[r1].from as usize),
        stride(&map.out, map.axes[r2].from as usize),
        tiles,
    ];
    // The grid must fit the device; otherwise the generic kernel runs.
    fold_workgroups(ctx, tiles).ok()?;
    Some(Plan { ty, params, tiles })
}

/// Runs the transpose of `input` into a new array of `len` elements.
pub(super) fn run(ctx: &Context, plan: &Plan, input: &Storage, len: usize) -> Result<Storage, LumaError> {
    let wgsl = word_ty(plan.ty).ok_or_else(|| LumaError::Gpu("transpose of an element type that cannot be tiled".into()))?;
    let core = ctx.builtin(Builtin::Transpose, wgsl)?;
    let out = Storage::new(ctx, len, plan.ty.layout());
    {
        let out_pin = out.pin(0)?;
        let in_pin = input.pin(0)?;
        let raw = [
            RawArg::write(out_pin.buffer(), out.byte_len()),
            RawArg::read(in_pin.buffer(), input.byte_len()),
            RawArg::Uniform(bytemuck::cast_slice(&plan.params)),
        ];
        ctx.launch(&core, fold_workgroups(ctx, plan.tiles)?, [len_u32(len)?, 1, 1], &raw)?;
    }
    let stats = &ctx.jit().stats;
    stats.elementwise_launches.fetch_add(1, Ordering::Relaxed);
    stats.transposes.fetch_add(1, Ordering::Relaxed);
    Ok(out)
}

/// The WGSL of [`Builtin::Transpose`] for elements of the WGSL type `ty`. Arguments: out, src,
/// uniform `Params`: `a` the extent of the source's axis 0, `b` that of the source axis the
/// output's axis 0 reads, `tx`/`ty` the tiles along them, `n1` the extent of the first batch
/// axis, the source strides of the `b` axis and of both batch axes (`ss_*`), the output strides
/// of the axis that holds source axis 0 and of both batch axes (`os_*`), and the tile count.
/// The tile is stored row by row of `b` with a row pitch of 33 words, so that neither the loads
/// (along `a`) nor the stores (down a row of `b`) hit one bank.
pub(crate) fn source(ty: &str) -> String {
    format!(
        "struct Params {{ a: u32, b: u32, tx: u32, ty: u32, n1: u32, ss_p: u32, ss_1: u32, ss_2: u32, os_q: u32, os_1: u32, os_2: u32, tiles: u32 }}
@group(1) @binding(0) var<storage, read_write> out: array<{ty}>;
@group(1) @binding(1) var<storage, read> src: array<{ty}>;
@group(1) @binding(2) var<uniform> p: Params;

var<workgroup> tile: array<{ty}, 1056>;

@compute @workgroup_size(32, 8)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let t = wid.x + wid.y * nwg.x;
    if (t >= p.tiles) {{ return; }}
    let ta = t % p.tx;
    let rest = t / p.tx;
    let tb = rest % p.ty;
    let batch = rest / p.ty;
    let c1 = batch % p.n1;
    let c2 = batch / p.n1;
    let a0 = ta * 32u;
    let b0 = tb * 32u;
    let sbase = c1 * p.ss_1 + c2 * p.ss_2;
    let obase = c1 * p.os_1 + c2 * p.os_2;
    let la = a0 + lid.x;
    for (var k = 0u; k < 4u; k = k + 1u) {{
        let row = lid.y + k * 8u;
        let b = b0 + row;
        if (la < p.a && b < p.b) {{
            tile[row * 33u + lid.x] = src[sbase + b * p.ss_p + la];
        }}
    }}
    workgroupBarrier();
    let sb = b0 + lid.x;
    for (var k = 0u; k < 4u; k = k + 1u) {{
        let col = lid.y + k * 8u;
        let a = a0 + col;
        if (a < p.a && sb < p.b) {{
            out[obase + a * p.os_q + sb] = tile[lid.x * 33u + col];
        }}
    }}
}}
"
    )
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use crate::array::Array;
    use crate::context::{Context, ContextOptions};
    use crate::gpu_type::{Df32, ScalarType};
    use crate::jit::cost::CostModel;

    fn ctx() -> Context {
        let ctx = Context::with_options(ContextOptions::from_env().calibrate(false)).unwrap();
        ctx.set_cost_model(CostModel::TEST);
        ctx
    }

    /// The array `data` of `dims` permuted by `perm`, computed on the host.
    fn permuted<T: Copy>(data: &[T], dims: [usize; 4], perm: [usize; 4]) -> Vec<T> {
        let out: [usize; 4] = std::array::from_fn(|k| dims[perm[k]]);
        let src_stride: [usize; 4] = std::array::from_fn(|a| dims[..a].iter().product());
        (0..data.len())
            .map(|mut o| {
                let mut at = 0;
                for k in 0..4 {
                    at += (o % out[k]) * src_stride[perm[k]];
                    o /= out[k];
                }
                data[at]
            })
            .collect()
    }

    /// Permutes `data` on the context and compares it with the host; `fast` says whether the
    /// tiled kernel must have run.
    fn check<T: ScalarType + Copy + Debug + PartialEq>(ctx: &Context, make: impl Fn(usize) -> T, dims: [usize; 4], perm: [usize; 4], fast: bool) {
        let len: usize = dims.iter().product();
        let data: Vec<T> = (0..len).map(make).collect();
        let x = Array::from_slice_on(ctx, &data, dims).unwrap();
        let before = ctx.jit().stats();
        let y = x.permute_axes(perm);
        let got = y.to_vec().unwrap();
        let used = ctx.jit().stats().transposes - before.transposes;
        assert_eq!(got, permuted(&data, dims, perm), "{dims:?} permuted by {perm:?}");
        assert_eq!(used, u64::from(fast), "{dims:?} permuted by {perm:?}: tiled kernel launches");
    }

    fn check_all<T: ScalarType + Copy + Debug + PartialEq>(make: impl Fn(usize) -> T + Copy) {
        let ctx = ctx();
        // A transpose of a matrix: the tiled kernel, at ragged and thin sizes.
        for dims in [[33, 17, 1, 1], [8192, 3, 1, 1], [3, 8192, 1, 1], [64, 64, 1, 1]] {
            check(&ctx, make, dims, [1, 0, 2, 3], true);
        }
        let dims = [40, 31, 3, 2];
        check(&ctx, make, dims, [1, 0, 2, 3], true);
        check(&ctx, make, dims, [2, 0, 1, 3], true);
        check(&ctx, make, dims, [3, 2, 1, 0], true);
        check(&ctx, make, dims, [3, 0, 2, 1], true);
        // Axis 0 stays in place: the generic kernel.
        check(&ctx, make, dims, [0, 2, 1, 3], false);
    }

    #[test]
    fn tiled_transpose_f32_matches_the_host() {
        check_all(|i| i as f32 * 0.25 - 100.0);
    }

    #[test]
    fn tiled_transpose_u32_matches_the_host() {
        check_all(|i| (i as u32).wrapping_mul(2654435761));
    }

    #[test]
    fn tiled_transpose_i32_matches_the_host() {
        check_all(|i| (i as i32).wrapping_mul(-1640531527));
    }

    #[test]
    fn tiled_transpose_df32_matches_the_host() {
        let ctx = ctx();
        let make = |i: usize| Df32 { hi: i as f32 + 0.5, lo: (i as f32).recip() * 1e-8 };
        let same = |ctx: &Context, dims: [usize; 4], perm: [usize; 4], fast: bool| {
            let len: usize = dims.iter().product();
            let data: Vec<Df32> = (0..len).map(make).collect();
            let x = Array::from_slice_on(ctx, &data, dims).unwrap();
            let before = ctx.jit().stats();
            let got = x.permute_axes(perm).to_vec().unwrap();
            let used = ctx.jit().stats().transposes - before.transposes;
            let bits = |v: &[Df32]| v.iter().map(|d| (d.hi.to_bits(), d.lo.to_bits())).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&permuted(&data, dims, perm)), "{dims:?} permuted by {perm:?}");
            assert_eq!(used, u64::from(fast), "{dims:?} permuted by {perm:?}: tiled kernel launches");
        };
        same(&ctx, [33, 17, 1, 1], [1, 0, 2, 3], true);
        same(&ctx, [8192, 3, 1, 1], [1, 0, 2, 3], true);
        same(&ctx, [3, 8192, 1, 1], [1, 0, 2, 3], true);
        same(&ctx, [40, 31, 3, 2], [2, 0, 1, 3], true);
        same(&ctx, [40, 31, 3, 2], [3, 2, 1, 0], true);
        same(&ctx, [40, 31, 3, 2], [0, 2, 1, 3], false);
    }

    /// The public paths a materialized transpose takes: evaluated as it is, copied by a
    /// multiplication by one (which simplifies to a copy), cached, and through `permute_axes`.
    #[test]
    fn transposes_through_the_public_api_take_the_tiled_kernel() {
        let ctx = ctx();
        let (w, h) = (96usize, 70usize);
        let data: Vec<f32> = (0..w * h).map(|i| i as f32 * 0.5 - 7.0).collect();
        let want = permuted(&data, [w, h, 1, 1], [1, 0, 2, 3]);
        let a = Array::from_slice_on(&ctx, &data, [w, h]).unwrap();
        type Build<'a> = Box<dyn Fn() -> Array<f32> + 'a>;
        let cases: [(&str, Build<'_>); 4] = [
            ("transpose", Box::new(|| a.transpose())),
            ("transpose * 1.0", Box::new(|| a.transpose() * 1.0)),
            ("transpose cached", Box::new(|| a.transpose().cache())),
            ("permute_axes", Box::new(|| a.permute_axes([1, 0, 2, 3]))),
        ];
        for (name, build) in cases {
            let before = ctx.jit().stats();
            let got = build().to_vec().unwrap();
            assert_eq!(got, want, "{name}");
            assert_eq!(ctx.jit().stats().transposes - before.transposes, 1, "{name}: tiled kernel launches");
        }
    }

    #[test]
    fn the_transpose_kernel_is_valid_wgsl() {
        for ty in ["u32", "vec2<f32>"] {
            let wgsl = super::source(ty);
            luma_core::kir::parse_wgsl(&wgsl, "main", None, luma_core::Features::empty(), [256; 3], 256, 64, true)
                .unwrap_or_else(|error| panic!("{ty}: {error}\n{wgsl}"));
        }
    }
}
