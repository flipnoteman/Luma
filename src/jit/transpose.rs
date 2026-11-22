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
//!
//! A store program that reads transposed operands elementwise (`a.t() + b`, `a.t() * s + b`, the
//! transpose of a sum) runs the same way ([`plan_fused`]): a workgroup stages its tile of every
//! transposed operand in workgroup memory, then computes the elements of its tile of the output
//! with the whole program, reading the staged operands from workgroup memory and every other
//! operand at the output's element. Loads of the transposed operands and the writes of the result
//! are both coalesced; the generic kernel reads the transposed operand with a stride.

use std::fmt::Write;
use std::sync::{Arc, Mutex};
use std::sync::atomic::Ordering;

use crate::array::Storage;
use crate::builtin::{Builtin, UnOp};
use crate::context::{Context, RawArg};
use crate::error::LumaError;
use crate::ops::{fold_workgroups, len_u32};

use luma_core::KernelTrust;

use crate::gpu::hash::{FxHasher, FxMap};
use crate::gpu::lock;
use crate::kernel::{self, KernelCore, KernelKey};

use super::codegen;
use super::program::{FusedKey, Instr, Ref, Tail};
use super::{Edge, ElemTy, OpKind, ShiftKind, ViewMap};

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

/// The type the kernel moves: any 4-byte element is a `u32` (a copy of the bits), a `Df32` its
/// two words, a small integer the bits of a field of a word (`"u8"` or `"u16"`, which
/// [`source`] reads as the packed kernel).
fn word_ty(ty: ElemTy) -> Option<&'static str> {
    match ty {
        ElemTy::F32 | ElemTy::U32 | ElemTy::I32 => Some("u32"),
        ElemTy::Df32 => Some("vec2<f32>"),
        ElemTy::U8 | ElemTy::I8 => Some("u8"),
        ElemTy::U16 | ElemTy::I16 => Some("u16"),
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
    let Geometry { params, tiles } = geometry(ctx, &map)?;
    // Small integers are written as whole words: every row of the output the tile writes, and every
    // batch, must start on a word and be made of whole words (see `source`).
    if ty.is_small() {
        let per = 32 / ty.small_bits();
        if [params[1], params[8], params[9], params[10]].iter().any(|&v| v % per != 0) {
            return None;
        }
    }
    Some(Plan { ty, params, tiles })
}

/// How the tiled kernels walk a transposing [`ViewMap`]: the words of their uniform's first twelve
/// fields (see [`source`]) and the workgroups they launch before folding.
struct Geometry {
    params: [u32; 12],
    tiles: u32,
}

/// The geometry of `map` if it is a permutation of the four axes that takes every source axis whole
/// (start 0, step 1), moves the fastest axis, and exchanges two axes of more than one element,
/// and the grid of its tiles fits the device.
fn geometry(ctx: &Context, map: &ViewMap) -> Option<Geometry> {
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
    Some(Geometry { params, tiles })
}

/// Most transposed operands one fused kernel stages in workgroup memory (4.1 KB each: three fit
/// the 16 KB every device offers).
const MAX_TILED: usize = 3;

/// Fewest elements a fused kernel with a transposed operand tiles: below it the generic kernel's
/// strided reads stay in cache, and building the source per launch would cost more than it saves.
const MIN_FUSED_LEN: usize = 1 << 16;

/// Whether registers and buffers of `ty` are one word that is the element itself.
fn is_word(ty: ElemTy) -> bool {
    matches!(ty, ElemTy::F32 | ElemTy::U32 | ElemTy::I32)
}

/// What a launch of a store program that [`plan_fused`] sends to the fused transpose kernel runs.
pub(super) struct FusedPlan {
    program: Arc<Program>,
    /// The words of the uniform: the geometry (see [`source`]), then the scalars of the program.
    params: Vec<u32>,
    tiles: u32,
}

/// The kernel of a [`FusedKey`] and what it needs of the inputs it is launched on; it depends on
/// the key only, so [`FusedCache`] keeps it and a repeated launch builds no source.
pub(super) struct Program {
    core: Arc<KernelCore>,
    /// The transposing view every staged operand is read through.
    map: ViewMap,
    out_ty: ElemTy,
    /// The inputs of the kernel in binding order: (input of the key, its place among the inputs
    /// the launch is given, which lack the in-place ones).
    bound: Vec<(usize, usize)>,
    /// Per input of the key, the elements it must have, one entry per way the program reads it.
    sized: Vec<(u32, usize)>,
}

/// The [`Program`]s of a context by the key they were made for (`None`: the key is not covered).
#[derive(Default)]
pub(super) struct FusedCache(Mutex<FxMap<u64, Vec<CacheEntry>>>);

impl std::fmt::Debug for FusedCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusedCache").finish_non_exhaustive()
    }
}

struct CacheEntry {
    instrs: Arc<Vec<Instr>>,
    subs: Arc<Vec<Vec<Instr>>>,
    input_tys: Vec<ElemTy>,
    scalar_tys: Vec<ElemTy>,
    outputs: Vec<u32>,
    in_place: Vec<(u32, u32)>,
    program: Option<Arc<Program>>,
}

impl CacheEntry {
    fn matches(&self, key: &FusedKey) -> bool {
        (Arc::ptr_eq(&self.instrs, &key.instrs) || self.instrs == key.instrs)
            && (Arc::ptr_eq(&self.subs, &key.subs) || self.subs == key.subs)
            && self.in_place == key.in_place
            && self.input_tys == key.input_tys
            && self.scalar_tys == key.scalar_tys
            && self.outputs == key.outputs
    }
}

/// The hash a program is cached by: what [`FusedKey::shape`] leaves out.
fn cache_hash(key: &FusedKey) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = FxHasher::default();
    key.shape.hash(&mut hasher);
    key.in_place.hash(&mut hasher);
    hasher.finish()
}

/// Whether the store program `key` over `inputs` (without those `key.in_place` names), computing
/// `len` elements with the scalar words `scalars`, is an elementwise expression of operands that
/// are read at the element computed, through views (broadcasts, slices) and through transposes
/// that [`geometry`] tiles, with at least one transpose; the kernel then stages every transposed
/// operand through workgroup memory (see [`fused_source`]), so that its reads and the writes of the
/// result are coalesced. Everything else (shifts other than views, sub-programs, generators,
/// elements other than `f32`, `u32` and `i32`, several outputs, a transposed operand of another
/// geometry than the others) is left to the generic kernel, which is also what the CPU backend runs.
pub(super) fn plan_fused(ctx: &Context, key: &FusedKey, inputs: &[Arc<Storage>], scalars: &[u32], len: usize) -> Result<Option<FusedPlan>, LumaError> {
    // The cheap tests first: most store programs have no transposed operand.
    if len < MIN_FUSED_LEN
        // The CPU backend runs the generic kernel faster (a tile and a barrier cost more than the strided reads do).
        || ctx.is_cpu()
        || key.tail != Tail::Store
        || !key.instrs.iter().any(|instr| matches!(instr.op.kind, OpKind::Shift(ShiftKind::View { map, edge: Edge::None }) if map.axes[0].from != 0 && map.out_len() == len))
    {
        return Ok(None);
    }
    let jit = ctx.jit();
    let hash = cache_hash(key);
    let known = lock(&jit.fused.0).get(&hash).and_then(|list| list.iter().find(|entry| entry.matches(key)).map(|entry| entry.program.clone()));
    let program = match known {
        Some(program) => program,
        None => {
            let program = analyze(ctx, key, len).map(|analysis| build(ctx, analysis)).transpose()?.map(Arc::new);
            let mut cache = lock(&jit.fused.0);
            if cache.len() >= 4096 {
                cache.clear();
            }
            cache.entry(hash).or_default().push(CacheEntry {
                instrs: Arc::clone(&key.instrs),
                subs: Arc::clone(&key.subs),
                input_tys: key.input_tys.clone(),
                scalar_tys: key.scalar_tys.clone(),
                outputs: key.outputs.clone(),
                in_place: key.in_place.clone(),
                program: program.clone(),
            });
            program
        }
    };
    let Some(program) = program else { return Ok(None) };
    if program.map.out_len() != len || scalars.len() != key.scalar_tys.len() || len as u64 > ctx.segment_bytes() / program.out_ty.stride() {
        return Ok(None);
    }
    for &(k, at) in &program.bound {
        let input = &inputs[at];
        if input.layout() != key.input_tys[k].layout()
            || input.segments() != 1
            || input.is_window()
            || program.sized.iter().any(|&(of, n)| of as usize == k && input.len() != n)
        {
            return Ok(None);
        }
    }
    let Some(Geometry { params, tiles }) = geometry(ctx, &program.map) else { return Ok(None) };
    let mut params = params.to_vec();
    params.extend_from_slice(scalars);
    Ok(Some(FusedPlan { program, params, tiles }))
}

/// What [`plan_fused`] makes of a key before the kernel is built: [`Program`] without it, and the
/// kernel's source and label. `None` if the key is not covered; `len` is the elements it computes.
fn analyze(ctx: &Context, key: &FusedKey, len: usize) -> Option<Analysis> {
    if key.outputs.len() != 1
        || key.guard.is_some()
        || key.staging.is_some()
        || key.tile.is_some()
        || key.runs
        || !key.spans.is_empty()
        || key.params.is_some()
        || !key.scalar_tys.iter().all(|&ty| is_word(ty))
        || !key.input_tys.iter().all(|&ty| is_word(ty))
    {
        return None;
    }
    let out_at = key.outputs[0] as usize;
    let out_ty = key.instrs.get(out_at)?.op.out_ty();
    if !is_word(out_ty) {
        return None;
    }
    let keep = codegen::needed(&key.instrs, &[out_at]);
    let mut geo: Option<ViewMap> = None;
    // What the workgroup stages: an input, or a sub-program computed at the elements of the source.
    let mut tiles_of: Vec<Staged> = Vec::new();
    // The elements the program reads of an input, at least: its length must be that.
    let mut sized: Vec<(u32, usize)> = Vec::new();
    let mut direct = vec![false; key.input_tys.len()];
    let mut used = vec![false; key.input_tys.len()];
    let mut sub_used = vec![false; key.subs.len()];
    let mut tiled_at: Vec<Option<usize>> = vec![None; key.instrs.len()];
    for (at, instr) in key.instrs.iter().enumerate().filter(|&(at, _)| keep[at]) {
        if !word_or_bool(instr.op.ty) || !word_or_bool(instr.op.out_ty()) {
            return None;
        }
        let skip = match instr.op.kind {
            OpKind::Shift(ShiftKind::View { map, edge }) => {
                if map.out_len() != len {
                    return None;
                }
                let source = match instr.args[0] {
                    Ref::Input(k) => Staged::Input(k),
                    Ref::Sub(s) => Staged::Sub(s),
                    _ => return None,
                };
                let elements = map.src.iter().map(|&d| d as usize).product();
                if edge == Edge::None && geometry(ctx, &map).is_some() {
                    if geo.is_some_and(|known| known != map) {
                        return None;
                    }
                    geo = Some(map);
                    tiled_at[at] = Some(match tiles_of.iter().position(|&known| known == source) {
                        Some(n) => n,
                        None => {
                            tiles_of.push(source);
                            tiles_of.len() - 1
                        }
                    });
                    if let Staged::Sub(s) = source {
                        // The sub-program runs at the elements of the source: its inputs have them all.
                        let sub = key.subs.get(s as usize)?;
                        if !sub_is_plain(sub) || !is_word(sub.last()?.op.out_ty()) {
                            return None;
                        }
                        sub_used[s as usize] = true;
                        for instr in sub {
                            for arg in instr.args {
                                if let Ref::Input(k) = arg {
                                    used[k as usize] = true;
                                    sized.push((k, elements));
                                }
                            }
                        }
                    }
                }
                match (instr.args[0], tiled_at[at]) {
                    (Ref::Input(k), _) => sized.push((k, elements)),
                    // A view that is not staged reads an input only.
                    (_, None) => return None,
                    _ => {}
                }
                1
            }
            OpKind::Shift(_) | OpKind::Generate(_) => return None,
            _ => 0,
        };
        for (n, arg) in instr.args.iter().enumerate() {
            match *arg {
                Ref::Input(k) => {
                    used[k as usize] = true;
                    if n >= skip {
                        direct[k as usize] = true;
                        sized.push((k, len));
                    }
                }
                Ref::Sub(_) if n == 0 && skip == 1 => {}
                Ref::Sub(_) => return None,
                _ => {}
            }
        }
    }
    let map = geo?;
    if tiles_of.len() > MAX_TILED {
        return None;
    }
    geometry(ctx, &map)?;
    // The generic views: whichever the transposes are not.
    let views: Vec<(ViewMap, Edge)> = codegen::views_of(key).into_iter().filter(|&(view, edge)| !(edge == Edge::None && view == map)).collect();
    // Which of the inputs the launch is given each kernel input is.
    let mut bound = Vec::new();
    for (k, &is_used) in used.iter().enumerate() {
        if !is_used {
            continue;
        }
        if key.in_place.iter().any(|&(paired, _)| paired as usize == k) {
            // Read from the output buffer at the element, and nowhere else.
            if !direct[k] || sized.iter().any(|&(of, n)| of as usize == k && n != len) {
                return None;
            }
            continue;
        }
        bound.push((k, k - key.in_place.iter().filter(|&&(paired, _)| (paired as usize) < k).count()));
    }
    let bound_inputs: Vec<usize> = bound.iter().map(|&(k, _)| k).collect();
    let source = fused_source(key, out_at, &keep, &tiled_at, &tiles_of, &sub_used, &bound_inputs, &views);
    let label = format!("Transposed {}", key.label());
    Some(Analysis { map, out_ty, bound, sized, source, label })
}

struct Analysis {
    map: ViewMap,
    out_ty: ElemTy,
    bound: Vec<(usize, usize)>,
    sized: Vec<(u32, usize)>,
    source: String,
    label: String,
}

/// The [`Program`] of `analysis`: its kernel compiled, or fetched from the context's cache of
/// kernels by source.
fn build(ctx: &Context, analysis: Analysis) -> Result<Program, LumaError> {
    let Analysis { map, out_ty, bound, sized, source, label } = analysis;
    let key = KernelKey::trusted(source, &label);
    let (core, _) = ctx.kernel_cache().get_or_build(&key, || kernel::build_core(ctx, &key, Some(&label), KernelTrust::Trusted).map(Arc::new))?;
    Ok(Program { core, map, out_ty, bound, sized })
}

/// What a workgroup stages in workgroup memory (see [`fused_source`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Staged {
    /// The elements of an input.
    Input(u32),
    /// The values of a sub-program of the key.
    Sub(u32),
}

/// Whether registers of `ty` are a word that is the element or a boolean.
fn word_or_bool(ty: ElemTy) -> bool {
    is_word(ty) || ty == ElemTy::Bool
}

/// Whether `sub` is elementwise arithmetic on its inputs and scalars only: no shifts, generators or
/// other sub-programs, elements of word types.
fn sub_is_plain(sub: &[Instr]) -> bool {
    sub.iter().all(|instr| {
        !matches!(instr.op.kind, OpKind::Shift(_) | OpKind::Generate(_))
            && word_or_bool(instr.op.ty)
            && word_or_bool(instr.op.out_ty())
            && instr.args.iter().all(|arg| !matches!(arg, Ref::Sub(_)))
    })
}

/// The WGSL of the fused transpose kernel of `key`. Arguments: the output `out0`, the `bound`
/// inputs (`in{k}`, one binding each), the uniform `p` of the geometry (the fields of
/// [`source`]) and the scalars `s{k}`. Every workgroup computes a tile of 32 x 32 elements of the
/// output: it stages the same tile of every transposed operand (`tile{n}`, as in [`source`]: an
/// input, or a sub-program computed at the source's elements, which is how the transpose of a
/// sum is staged), then computes each element of its tile with the program, reading the staged
/// operands from workgroup memory and every other operand at the element of the output, and
/// writes it; the invocations of a row of 32 are consecutive elements of the output.
#[allow(clippy::too_many_arguments)]
fn fused_source(key: &FusedKey, out_at: usize, keep: &[bool], tiled_at: &[Option<usize>], staged: &[Staged], sub_used: &[bool], bound: &[usize], views: &[(ViewMap, Edge)]) -> String {
    let elem = |k: usize| key.input_tys[k].wgsl();
    let out = key.instrs[out_at].op.out_ty().wgsl();
    let mut src = String::from("struct Params { a: u32, b: u32, tx: u32, ty: u32, n1: u32, ss_p: u32, ss_1: u32, ss_2: u32, os_q: u32, os_1: u32, os_2: u32, tiles: u32, ");
    for (k, ty) in key.scalar_tys.iter().enumerate() {
        let _ = write!(src, "s{k}: {}, ", ty.wgsl());
    }
    src.push_str("}\n");
    let _ = writeln!(src, "@group(1) @binding(0) var<storage, read_write> out0: array<{out}>;");
    for (n, &k) in bound.iter().enumerate() {
        let _ = writeln!(src, "@group(1) @binding({}) var<storage, read> in{k}: array<{}>;", n + 1, elem(k));
    }
    let _ = writeln!(src, "@group(1) @binding({}) var<uniform> p: Params;", bound.len() + 1);
    let tile_ty = |from: &Staged| match *from {
        Staged::Input(k) => elem(k as usize),
        Staged::Sub(s) => key.subs[s as usize].last().map_or("f32", |instr| instr.op.out_ty().wgsl()),
    };
    for (n, from) in staged.iter().enumerate() {
        let _ = writeln!(src, "var<workgroup> tile{n}: array<{}, 1056>;", tile_ty(from));
    }
    src.push_str(&codegen::view_helpers(views));
    let names = codegen::Names { views, in_place: &key.in_place };
    for (s, instrs) in key.subs.iter().enumerate().filter(|&(s, _)| sub_used[s]) {
        let ty = instrs.last().map_or(ElemTy::F32, |instr| instr.op.out_ty()).wgsl();
        let body = codegen::body_with(instrs, &key.input_tys, &names, "i", false, None, &|_, _| None);
        let _ = write!(src, "\nfn sub{s}(i: u32) -> {ty} {{\n{body}    return t{};\n}}\n", instrs.len() - 1);
    }
    let body = codegen::body_with(&key.instrs, &key.input_tys, &names, "i", false, Some(keep), &|at, _| tiled_at[at].map(|n| format!("tile{n}[lid.x * 33u + col]")));
    src.push_str(
        "
@compute @workgroup_size(32, 8)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let t = wid.x + wid.y * nwg.x;
    if (t >= p.tiles) { return; }
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
    for (var k = 0u; k < 4u; k = k + 1u) {
        let row = lid.y + k * 8u;
        let b = b0 + row;
        if (la < p.a && b < p.b) {
            let o = sbase + b * p.ss_p + la;
",
    );
    for (n, from) in staged.iter().enumerate() {
        let load = match *from {
            Staged::Input(k) => format!("in{k}[o]"),
            Staged::Sub(s) => format!("sub{s}(o)"),
        };
        let _ = writeln!(src, "            tile{n}[row * 33u + lid.x] = {load};");
    }
    src.push_str(
        "        }
    }
    workgroupBarrier();
    let sb = b0 + lid.x;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let col = lid.y + k * 8u;
        let a = a0 + col;
        if (a < p.a && sb < p.b) {
            let i = obase + a * p.os_q + sb;
",
    );
    src.push_str(&body.lines().map(|line| format!("    {line}\n")).collect::<String>());
    let _ = writeln!(src, "            out0[i] = t{out_at};\n        }}\n    }}\n}}");
    if src.contains("luma_udiv") || src.contains("luma_urem") {
        src.push_str(codegen::DIV_HELPERS);
    }
    src
}

/// Runs the fused transpose kernel of `plan` over `inputs` (the launch's, without the in-place
/// ones) into `reused` (the buffer of the in-place input) or a new array of `len` elements.
pub(super) fn run_fused(ctx: &Context, plan: &FusedPlan, inputs: &[Arc<Storage>], len: usize, reused: Option<Storage>) -> Result<Storage, LumaError> {
    let program = &plan.program;
    let out = match reused {
        Some(storage) => storage,
        None => Storage::new(ctx, len, program.out_ty.layout()),
    };
    {
        let out_pin = out.pin(0)?;
        let pins = program.bound.iter().map(|&(_, at)| inputs[at].pin(0).map(|pin| (pin, inputs[at].byte_len()))).collect::<Result<Vec<_>, _>>()?;
        let mut raw = Vec::with_capacity(pins.len() + 2);
        raw.push(RawArg::write(out_pin.buffer(), out.byte_len()));
        raw.extend(pins.iter().map(|(pin, bytes)| RawArg::read(pin.buffer(), *bytes)));
        raw.push(RawArg::Uniform(bytemuck::cast_slice(&plan.params)));
        ctx.launch(&program.core, fold_workgroups(ctx, plan.tiles)?, [len_u32(len)?, 1, 1], &raw)?;
    }
    let stats = &ctx.jit().stats;
    stats.elementwise_launches.fetch_add(1, Ordering::Relaxed);
    stats.transposes.fetch_add(1, Ordering::Relaxed);
    Ok(out)
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
    if let Some(bits) = match ty {
        "u8" => Some(8u32),
        "u16" => Some(16),
        _ => None,
    } {
        return packed_source(bits);
    }
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

/// [`source`] for elements of `bits` (8 or 16) bits sharing the words of both buffers. Loads read
/// the field of an element out of its word, as the tile's slot; stores write whole words: a word is
/// `32 / bits` neighbouring elements of a row of the output, which [`plan`] has made start on a
/// word and end on one (the extent of the source axis the output's axis 0 reads, and the output
/// strides, are multiples of the elements to a word), so no two invocations write one word. The
/// invocations of a workgroup number the words of the tile (`32` columns of `32 / per` words) and
/// each takes every 256th.
fn packed_source(bits: u32) -> String {
    let per = 32 / bits;
    let shift = per.trailing_zeros();
    let mask = (1u64 << bits) - 1;
    format!(
        "struct Params {{ a: u32, b: u32, tx: u32, ty: u32, n1: u32, ss_p: u32, ss_1: u32, ss_2: u32, os_q: u32, os_1: u32, os_2: u32, tiles: u32 }}
@group(1) @binding(0) var<storage, read_write> out: array<u32>;
@group(1) @binding(1) var<storage, read> src: array<u32>;
@group(1) @binding(2) var<uniform> p: Params;

var<workgroup> tile: array<u32, 1056>;

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
            let e = sbase + b * p.ss_p + la;
            tile[row * 33u + lid.x] = (src[e >> {shift}u] >> ((e & {low}u) * {bits}u)) & {mask}u;
        }}
    }}
    workgroupBarrier();
    let id = lid.y * 32u + lid.x;
    for (var m = id; m < {words}u; m = m + 256u) {{
        let col = m / {wpc}u;
        let wj = m % {wpc}u;
        let a = a0 + col;
        let sb = b0 + wj * {per}u;
        if (a < p.a && sb < p.b) {{
            var word = 0u;
            for (var j = 0u; j < {per}u; j = j + 1u) {{
                word = word | (tile[(wj * {per}u + j) * 33u + col] << (j * {bits}u));
            }}
            out[(obase + a * p.os_q + sb) >> {shift}u] = word;
        }}
    }}
}}
",
        low = per - 1,
        words = 32 * (32 / per),
        wpc = 32 / per,
    )
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use crate::array::Array;
    use crate::ops::select;
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

    /// The tiled kernel on small integers: `fast` where the output rows are whole words, the generic
    /// kernel (which gives the same elements) where they are not.
    fn check_small<T: ScalarType + Copy + Debug + PartialEq>(make: impl Fn(usize) -> T + Copy, per: usize) {
        let ctx = ctx();
        for dims in [[64, 64, 1, 1], [33, 16, 1, 1], [8192, 8, 1, 1], [3, 4096, 1, 1], [100, 36, 1, 1]] {
            check(&ctx, make, dims, [1, 0, 2, 3], dims[1] % per == 0);
        }
        // Rows of 17 or 6 elements end inside a word (of 2 or 4): not tiled.
        check(&ctx, make, [33, 17, 1, 1], [1, 0, 2, 3], false);
        check(&ctx, make, [40, 6, 1, 1], [1, 0, 2, 3], per == 2);
        // Batches: a batch of the output starts on a word only if the transposed plane is a multiple of it.
        let dims = [40, 32, 3, 2];
        check(&ctx, make, dims, [1, 0, 2, 3], true);
        check(&ctx, make, [40, 32, 4, 2], [2, 0, 1, 3], true);
        check(&ctx, make, [40, 32, 4, 4], [3, 2, 1, 0], true);
        check(&ctx, make, [40, 31, 3, 2], [3, 2, 1, 0], per == 2);
        check(&ctx, make, dims, [0, 2, 1, 3], false);
        // Whether the rows of these end on words depends on the type; the elements are the same either way.
        for (dims, perm) in [([40, 32, 3, 2], [2, 0, 1, 3]), ([40, 33, 2, 6], [3, 0, 1, 2]), ([7, 5, 3, 2], [1, 2, 3, 0])] {
            let len: usize = dims.iter().product();
            let data: Vec<T> = (0..len).map(make).collect();
            let got = Array::from_slice_on(&ctx, &data, dims).unwrap().permute_axes(perm).to_vec().unwrap();
            assert_eq!(got, permuted(&data, dims, perm), "{dims:?} permuted by {perm:?}");
        }
    }

    #[test]
    fn tiled_transpose_of_bytes_matches_the_host() {
        check_small(|i| (i as u32).wrapping_mul(2654435761).to_be_bytes()[1], 4);
        check_small(|i| (i as i8).wrapping_mul(-37), 4);
    }

    #[test]
    fn tiled_transpose_of_halves_matches_the_host() {
        check_small(|i| (i as u32).wrapping_mul(2654435761).to_be_bytes()[0] as u16 * 257 + i as u16, 2);
        check_small(|i| (i as i16).wrapping_mul(-1357), 2);
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

    /// The launches of the tiled kernel a fused expression takes on the context: `n` on a GPU, none
    /// on the CPU backend, which runs the generic kernel.
    fn fused(ctx: &Context, n: u64) -> u64 {
        if ctx.is_cpu() { 0 } else { n }
    }

    /// The fused expression `build` over matrices of `dims` (the operand to transpose first),
    /// on the context and on the host: the results are equal (`f32`, the host computing the same
    /// operations), and the tiled kernel ran `fast` times.
    fn fused_check(ctx: &Context, name: &str, dims: [usize; 4], fast: u64, build: impl Fn(&Array<f32>, &Array<f32>) -> Array<f32>, want: impl Fn(&[f32], &[f32]) -> Vec<f32>) {
        let len: usize = dims.iter().product();
        let xs: Vec<f32> = (0..len).map(|i| ((i * 7919) % 1021) as f32 * 0.25 - 100.0).collect();
        let ys: Vec<f32> = (0..len).map(|i| ((i * 104729) % 997) as f32 * 0.125 + 1.0).collect();
        let x = Array::from_slice_on(ctx, &xs, dims).unwrap();
        let y = Array::from_slice_on(ctx, &ys, [dims[1], dims[0], dims[2], dims[3]]).unwrap();
        let before = ctx.jit().stats();
        let got = build(&x, &y).to_vec().unwrap();
        let used = ctx.jit().stats().transposes - before.transposes;
        assert_eq!(got, want(&xs, &ys), "{name} {dims:?}");
        assert_eq!(used, fused(ctx, fast), "{name} {dims:?}: tiled kernel launches");
    }

    /// `x` of `dims` transposed on the host (axes 0 and 1 exchanged).
    fn swapped(data: &[f32], dims: [usize; 4]) -> Vec<f32> {
        permuted(data, dims, [1, 0, 2, 3])
    }

    /// Expressions that read a transposed operand elementwise take the fused tiled kernel: at
    /// sizes that are not multiples of the tile, with a batch axis, with scalars, with the
    /// transpose of a sum, with a broadcast operand, and with an operand that is also read as it is.
    #[test]
    fn fused_transposed_expressions_match_the_host() {
        let ctx = ctx();
        for dims in [[301, 257, 1, 1], [257, 301, 1, 1], [512, 160, 1, 1], [71, 67, 15, 1], [60, 41, 7, 5]] {
            let t = |data: &[f32]| swapped(data, dims);
            fused_check(&ctx, "a^T + b", dims, 1, |a, b| &a.permute_axes([1, 0, 2, 3]) + b, |a, b| t(a).iter().zip(b).map(|(x, y)| x + y).collect());
            fused_check(&ctx, "a^T * 0.5 - b", dims, 1, |a, b| a.permute_axes([1, 0, 2, 3]) * 0.5 - b, |a, b| t(a).iter().zip(b).map(|(x, y)| x * 0.5 - y).collect());
            fused_check(&ctx, "(a + 2)^T * b", dims, 1, |a, b| (a + 2.0).permute_axes([1, 0, 2, 3]) * b, |a, b| t(a).iter().zip(b).map(|(x, y)| (x + 2.0) * y).collect());
            fused_check(&ctx, "max(a^T, b) + a^T", dims, 1, |a, b| a.permute_axes([1, 0, 2, 3]).maximum(b) + a.permute_axes([1, 0, 2, 3]), |a, b| t(a).iter().zip(b).map(|(x, y)| x.max(*y) + x).collect());
        }
        // Two transposed operands (of one shape), and one that is transposed and also read as is.
        let dims = [301, 257, 1, 1];
        let xs: Vec<f32> = (0..301 * 257).map(|i| (i % 251) as f32 - 100.0).collect();
        let zs: Vec<f32> = (0..301 * 257).map(|i| (i % 13) as f32 * 0.5).collect();
        let x = Array::from_slice_on(&ctx, &xs, dims).unwrap();
        let z = Array::from_slice_on(&ctx, &zs, dims).unwrap();
        let before = ctx.jit().stats();
        let got = (x.transpose() * z.transpose() + x.transpose()).to_vec().unwrap();
        let (tx, tz) = (swapped(&xs, dims), swapped(&zs, dims));
        assert_eq!(got, tx.iter().zip(&tz).map(|(a, b)| a * b + a).collect::<Vec<_>>());
        assert_eq!(ctx.jit().stats().transposes - before.transposes, fused(&ctx, 1));
        // A mask from a transposed comparison selects between transposed and plain operands.
        let w = Array::from_slice_on(&ctx, &zs, [257, 301]).unwrap();
        let before = ctx.jit().stats();
        let got = select(&x.transpose().gt(0.0), x.transpose(), &w).to_vec().unwrap();
        assert_eq!(got, tx.iter().zip(&zs).map(|(&a, &b)| if a > 0.0 { a } else { b }).collect::<Vec<_>>());
        assert_eq!(ctx.jit().stats().transposes - before.transposes, fused(&ctx, 1));
    }

    /// A transposed operand next to a broadcast row, a broadcast column and a scalar, and with
    /// integers.
    #[test]
    fn fused_transposed_expressions_with_broadcasts_and_integers() {
        let ctx = ctx();
        let (m, n) = (301usize, 257usize);
        let xs: Vec<f32> = (0..m * n).map(|i| (i % 251) as f32 - 100.0).collect();
        let row: Vec<f32> = (0..m).map(|i| i as f32 * 0.5).collect();
        let col: Vec<f32> = (0..n).map(|i| 1.0 + i as f32).collect();
        let x = Array::from_slice_on(&ctx, &xs, [m, n]).unwrap();
        let tx = swapped(&xs, [m, n, 1, 1]);
        // The transpose is `n` by `m` (axis 0 the fastest): a `1 x m` row and an `n x 1` column broadcast over it.
        let row_a = Array::from_slice_on(&ctx, &row, [1, m]).unwrap();
        let col_a = Array::from_slice_on(&ctx, &col, [n, 1]).unwrap();
        let before = ctx.jit().stats();
        let got = (x.transpose() * &row_a + &col_a).to_vec().unwrap();
        let want: Vec<f32> = (0..m * n).map(|o| tx[o] * row[o / n] + col[o % n]).collect();
        assert_eq!(got, want);
        assert_eq!(ctx.jit().stats().transposes - before.transposes, fused(&ctx, 1));

        // The other operand dies with the expression, so the result takes over its buffer.
        let ys: Vec<f32> = (0..m * n).map(|i| (i % 17) as f32).collect();
        let doubled = (Array::from_slice_on(&ctx, &ys, [n, m]).unwrap() * 2.0).cache();
        let before = ctx.jit().stats();
        let got = (x.transpose() + doubled).to_vec().unwrap();
        assert_eq!(got, (0..m * n).map(|o| tx[o] + ys[o] * 2.0).collect::<Vec<_>>());
        assert_eq!(ctx.jit().stats().transposes - before.transposes, fused(&ctx, 1));

        let ints: Vec<i32> = (0..m * n).map(|i| (i as i32).wrapping_mul(-1640531527) >> 8).collect();
        let a = Array::from_slice_on(&ctx, &ints, [m, n]).unwrap();
        let b = Array::from_slice_on(&ctx, &ints.iter().rev().copied().collect::<Vec<_>>(), [n, m]).unwrap();
        let before = ctx.jit().stats();
        let got = (a.transpose() + &b - 3).to_vec().unwrap();
        let ti = permuted(&ints, [m, n, 1, 1], [1, 0, 2, 3]);
        let want: Vec<i32> = ti.iter().zip(ints.iter().rev()).map(|(&x, &y)| x.wrapping_add(y).wrapping_sub(3)).collect();
        assert_eq!(got, want);
        assert_eq!(ctx.jit().stats().transposes - before.transposes, fused(&ctx, 1));
    }

    /// Where the fused tiling does not apply the generic kernel runs, with the same results: small
    /// arrays, a transpose that leaves the fastest axis in place, thin matrices.
    #[test]
    fn fused_transposed_expressions_that_are_not_tiled_still_match() {
        let ctx = ctx();
        fused_check(&ctx, "small a^T + b", [40, 30, 1, 1], 0, |a, b| &a.transpose() + b, |a, b| swapped(a, [40, 30, 1, 1]).iter().zip(b).map(|(x, y)| x + y).collect());
        fused_check(&ctx, "thin a^T + b", [65536 + 3, 1, 1, 1], 0, |a, b| &a.transpose() + b, |a, b| a.iter().zip(b).map(|(x, y)| x + y).collect());
        // A permutation that leaves axis 0 in place is not tiled.
        let (dims, perm) = ([300usize, 260, 3, 1], [0, 2, 1, 3]);
        let xs: Vec<f32> = (0..300 * 260 * 3).map(|i| (i % 509) as f32 - 250.0).collect();
        let ys: Vec<f32> = (0..300 * 260 * 3).map(|i| (i % 7) as f32).collect();
        let x = Array::from_slice_on(&ctx, &xs, dims).unwrap();
        let y = Array::from_slice_on(&ctx, &ys, [300, 3, 260, 1]).unwrap();
        let before = ctx.jit().stats();
        let got = (x.permute_axes(perm) + &y).to_vec().unwrap();
        assert_eq!(got, permuted(&xs, dims, perm).iter().zip(&ys).map(|(a, b)| a + b).collect::<Vec<_>>());
        assert_eq!(ctx.jit().stats().transposes - before.transposes, 0);
        // Two permutations that move axis 0 differently: the generic kernel.
        let (dims, dz) = ([40usize, 33, 60, 1], [60usize, 33, 40, 1]);
        let xs: Vec<f32> = (0..40 * 33 * 60).map(|i| (i % 509) as f32 - 250.0).collect();
        let zs: Vec<f32> = (0..40 * 33 * 60).map(|i| (i % 11) as f32).collect();
        let x = Array::from_slice_on(&ctx, &xs, dims).unwrap();
        let z = Array::from_slice_on(&ctx, &zs, dz).unwrap();
        let before = ctx.jit().stats();
        let got = (x.permute_axes([1, 0, 2, 3]) + z.permute_axes([1, 2, 0, 3])).to_vec().unwrap();
        let want: Vec<f32> = permuted(&xs, dims, [1, 0, 2, 3]).iter().zip(permuted(&zs, dz, [1, 2, 0, 3])).map(|(a, b)| a + b).collect();
        assert_eq!(got, want);
        assert_eq!(ctx.jit().stats().transposes - before.transposes, 0);
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
