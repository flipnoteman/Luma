//! WGSL generation for fused programs.
//!
//! The source depends on the [`FusedKey`] only. Bindings, in order of the arguments: the outputs
//! (`out0`, `out1`, ...; a reduction has one, holding the partial results of every op), the
//! input buffers (`in0`, `in1`, ...; the one a key marks `in_place` is `out0` instead and gets no
//! binding), then, if the program has scalars (or is a reduction), one
//! uniform `p` holding them (`p.s0`, `p.s1`, ... after `p.n` for reductions).
//!
//! The segmented variant (`key.spans` not empty) runs once per segment of arrays spanning several
//! buffers and has shifts that leave their segment. `p` then also holds `base` (the flat index of
//! the segment's first element), `total` (the array length) and, per input read through shifts,
//! the ids of the segments bound as `in{k}_0 ..`; `ld{k}` loads a flat index through them. An
//! input is bound as `in{k}` there only if the program also loads it at `i`.
//!
//! The program loads the inputs at element `i` and runs the instructions as `let` bindings. A
//! store kernel writes its output instructions; a reduction wraps the program in `luma_eval(i)`
//! and feeds the last value into the tree reductions of the first pass.

use std::fmt::Write;

use crate::builtin::{class_operand, df_library, reduce_main, substitute, ReduceElem};
use crate::gpu::hash::FxMap;
use crate::tuning::{Tuning, STENCIL_RADIUS};

use super::inline::{Inlined, Lowered, Region, StagedPrograms};
use super::params;
use super::program::{calls_df_library, FusedKey, Instr, Ref, Tail};
use super::staging::Staging;
use super::stencil::{signed, wrap_source, WORKGROUP};
use super::{Edge, ElemTy, Gen, Op, OpKind, ShiftKind, ViewMap};

/// The WGSL name of an operand of type `ty`. The scalars of an `f64` or `Df32` operand are
/// several `f32` slots of the uniform (see `Scalar::words`).
fn name(reference: Ref, ty: ElemTy) -> String {
    match reference {
        Ref::Unused => "0u".to_owned(),
        Ref::Input(k) => format!("a{k}"),
        Ref::Scalar(k) => match ty {
            ElemTy::F64 => format!("(f64(p.s{k}) + f64(p.s{}) + f64(p.s{}))", k + 1, k + 2),
            ElemTy::Df32 => format!("vec2<f32>(p.s{k}, p.s{})", k + 1),
            ElemTy::F32 | ElemTy::U32 => format!("p.s{k}"),
        },
        Ref::Tmp(k) => format!("t{k}"),
        Ref::Sub(k) => format!("sub{k}"),
    }
}

/// What the WGSL of a program refers to beyond its own instructions.
pub(super) struct Names<'a> {
    /// The distinct views of the kernel (see [`views_of`]); view `n` is the functions `view{n}`.
    pub(super) views: &'a [(ViewMap, Edge)],
    /// Pairs (input `k`, output `j`): input `k` is read from the output buffer `out{j}`.
    pub(super) in_place: &'a [(u32, u32)],
}

/// [`Names`] of a program without views or in-place inputs.
pub(super) const NO_NAMES: Names<'static> = Names { views: &[], in_place: &[] };

/// The WGSL expression of the operation `op` (anything but a shift) on the operands named `a`,
/// `b` and `c`; `index` names the element it is computed at. `None` for a shift.
fn operation(op: Op, a: &str, b: &str, c: &str, index: &str) -> Option<String> {
    let ty = op.ty;
    Some(match op.kind {
        OpKind::Unary(op) => substitute(op.expr_for(ty).unwrap_or_else(|| op.expr()), a, ""),
        OpKind::Binary(op) => substitute(op.expr_for(ty).unwrap_or_else(|| op.expr()), a, b),
        OpKind::Compare(op) => substitute(op.expr_for(ty), a, b),
        OpKind::Classify(class) => substitute(class.expr(), &class_operand(ty, a), ""),
        OpKind::Powf => format!("pow({a}, {b})"),
        OpKind::Clamp if ty == ElemTy::Df32 => format!("df_min(df_max({a}, {b}), {c})"),
        OpKind::Clamp => format!("clamp({a}, {b}, {c})"),
        OpKind::Select => format!("select({c}, {b}, {a} != 0u)"),
        OpKind::Cast(to) => cast(ty, to, a),
        OpKind::Generate(kind) => generate(kind, op.out_ty(), index, a, b),
        OpKind::Shift(_) => return None,
    })
}

/// The WGSL expression of `instr`. `index` names the element a shift is relative to;
/// `segmented` makes shifts of inputs read through the accessors `ld{k}`
/// (which take a flat index over all segments) instead of indexing `in{k}` directly.
pub(super) fn expression(instr: &Instr, index: &str, segmented: bool, names: &Names) -> String {
    let [a, b, c] = [0, 1, 2].map(|i| name(instr.args[i], instr.op.operand_ty(i)));
    if let Some(value) = operation(instr.op, &a, &b, &c, index) {
        return value;
    }
    let OpKind::Shift(kind) = instr.op.kind else { return String::new() };
    let (at, view) = match kind {
        ShiftKind::View { map, edge } => {
            let n = names.views.iter().position(|&known| known == (map, edge));
            // A view missing from `names` is not valid shader code and surfaces as a
            // shader compile error.
            (
                format!("view{}({index})", n.map_or_else(|| "_missing".to_owned(), |n| n.to_string())),
                n.filter(|_| edge == Edge::Fill).map(|n| format!("view{n}_in({index})")),
            )
        }
        ShiftKind::ParamView { id, spec } => (format!("pview{id}({index})"), (spec.edge == Edge::Fill).then(|| format!("pview{id}_in({index})"))),
        _ => (shift_index(kind, &b, index, &c), None),
    };
    let read = match instr.args[0] {
        Ref::Input(k) if segmented => format!("ld{k}({at})"),
        Ref::Input(k) => match names.in_place.iter().find(|&&(input, _)| input == k) {
            Some(&(_, j)) => format!("out{j}[{at}]"),
            None => format!("in{k}[{at}]"),
        },
        Ref::Sub(k) => format!("sub{k}({at})"),
        // A shift only reads an input buffer or a sub-program; anything else is not
        // valid shader code and surfaces as a shader compile error.
        other => format!("{}[{at}]", name(other, ElemTy::U32)),
    };
    match view {
        Some(inside) => format!("select({b}, {read}, {inside})"),
        None => read,
    }
}

/// The distinct views of `key`, in order of first appearance: the index of a view in this list
/// names its functions.
pub(super) fn views_of(key: &FusedKey) -> Vec<(ViewMap, Edge)> {
    let mut views = Vec::new();
    for instr in key.instrs.iter().chain(key.subs.iter().flatten()) {
        if let OpKind::Shift(ShiftKind::View { map, edge }) = instr.op.kind
            && !views.contains(&(map, edge))
        {
            views.push((map, edge));
        }
    }
    if let Tail::StoreMapped(map) = key.tail
        && !views.contains(&(map, Edge::None))
    {
        views.push((map, Edge::None));
    }
    views
}

/// [`views_of`] for a lowered key: the distinct views of the contexts of `lowered` (the programs
/// of one kernel; they include the compositions no instruction of the key names), in order of
/// creation.
fn views_of_inlined(key: &FusedKey, lowered: &[&Inlined]) -> Vec<(ViewMap, Edge)> {
    let mut views = Vec::new();
    for ctx in lowered.iter().flat_map(|inlined| &inlined.ctxs) {
        if let ShiftKind::View { map, edge } = ctx.kind
            && !views.contains(&(map, edge))
        {
            views.push((map, edge));
        }
    }
    if let Tail::StoreMapped(map) = key.tail
        && !views.contains(&(map, Edge::None))
    {
        views.push((map, Edge::None));
    }
    views
}

/// The WGSL functions that fold a coordinate back into `0..n`.
pub(super) const EDGE_HELPERS: &str = "fn luma_clamp_i(c: i32, n: i32) -> i32 { return clamp(c, 0, n - 1); }
fn luma_mirror_i(c: i32, n: i32) -> i32 { let a = select(c, -c - 1, c < 0); let b = select(a, 2 * n - 1 - a, a >= n); return clamp(b, 0, n - 1); }
fn luma_wrap_i(c: i32, n: i32) -> i32 { return ((c % n) + n) % n; }
";

/// The WGSL functions of `views`: `view{n}(i)` is the flat source index view `n` reads for
/// output element `i` and, for [`Edge::Fill`], `view{n}_in(i)` whether it lies inside the source.
pub(super) fn view_helpers(views: &[(ViewMap, Edge)]) -> String {
    if views.is_empty() {
        return String::new();
    }
    let mut src = String::from(EDGE_HELPERS);
    for (n, &(map, edge)) in views.iter().enumerate() {
        let mut out_stride = [1u64; 4];
        let mut src_stride = [1u64; 4];
        for d in 1..4 {
            out_stride[d] = out_stride[d - 1] * u64::from(map.out[d - 1]);
            src_stride[d] = src_stride[d - 1] * u64::from(map.src[d - 1]);
        }
        // The unclipped source coordinate of axis `a`.
        let raw = |a: usize| {
            let axis = map.axes[a];
            let from = axis.from as usize;
            if map.out[from] == 1 {
                return format!("{}", axis.start);
            }
            let quotient = if out_stride[from] == 1 { "i".to_owned() } else { format!("(i / {}u)", out_stride[from]) };
            format!("(i32(({quotient}) % {}u) * ({}) + ({}))", map.out[from], axis.step, axis.start)
        };
        // Whether the unclipped coordinate of axis `a` is always inside the source.
        let inside = |a: usize| {
            let (first, last) = map.ends(a);
            let size = i64::from(map.src[a]);
            (0..size).contains(&first) && (0..size).contains(&last)
        };
        let mut terms = Vec::new();
        for a in (0..4).filter(|&a| map.src[a] > 1) {
            let size = map.src[a];
            let coord = match edge {
                _ if inside(a) => raw(a),
                Edge::None => raw(a),
                Edge::Clamp | Edge::Fill => format!("luma_clamp_i({}, {size})", raw(a)),
                Edge::Mirror => format!("luma_mirror_i({}, {size})", raw(a)),
                Edge::Wrap => format!("luma_wrap_i({}, {size})", raw(a)),
            };
            terms.push(if src_stride[a] == 1 { format!("u32({coord})") } else { format!("u32({coord}) * {}u", src_stride[a]) });
        }
        let sum = if terms.is_empty() { "0u".to_owned() } else { terms.join(" + ") };
        let _ = writeln!(src, "fn view{n}(i: u32) -> u32 {{ return {sum}; }}");
        if edge == Edge::Fill {
            let tests: Vec<String> = (0..4)
                .filter(|&a| !inside(a))
                .map(|a| format!("{0} >= 0 && {0} < {1}", raw(a), map.src[a]))
                .collect();
            let all = if tests.is_empty() { "true".to_owned() } else { tests.join(" && ") };
            let _ = writeln!(src, "fn view{n}_in(i: u32) -> bool {{ return {all}; }}");
        }
    }
    src
}

/// `value` of type `from` converted to `to`. Every conversion that has a `Df32` end passes the
/// value that a reassociating compiler could fold away through `luma_o` (see
/// `builtin::df_library`), so that `x - to_df32(to_f32(x))` keeps the low word of `x`.
fn cast(from: ElemTy, to: ElemTy, value: &str) -> String {
    match (from, to) {
        (ElemTy::F32, ElemTy::F64) => format!("f64({value})"),
        (ElemTy::F64, ElemTy::F32) => format!("f32({value})"),
        (ElemTy::F32, ElemTy::Df32) => format!("vec2<f32>(luma_o({value}), 0.0)"),
        (ElemTy::Df32, ElemTy::F32) => format!("luma_o({value}.x + {value}.y)"),
        (ElemTy::F64, ElemTy::Df32) => format!("vec2<f32>(luma_o(f32({value})), f32({value} - f64(luma_o(f32({value})))))"),
        (ElemTy::Df32, ElemTy::F64) => format!("(f64(luma_o({value}.x)) + f64({value}.y))"),
        _ => value.to_owned(),
    }
}

/// The double-float library a kernel of `key` defines (a reduction of `Df32` brings its own,
/// see `reduce_main`): behind the guard uniform when the program has one, else with a constant
/// guard, for the functions that never read it (ordering, minimum, maximum, absolute value).
pub(super) fn library_for(key: &FusedKey, tuning: &Tuning) -> String {
    let library = |guard: &str| df_library(guard, tuning.df_fma, tuning.df_sloppy_add);
    match key.guard {
        Some(guard) => library(&format!("p.s{guard}")),
        None if key.instrs.iter().chain(key.subs.iter().flatten()).any(calls_df_library) => library("0u"),
        None => String::new(),
    }
}

fn generate(kind: Gen, ty: ElemTy, index: &str, a: &str, b: &str) -> String {
    match kind {
        Gen::Bits => format!("luma_threefry({index}, {a}, {b}).x"),
        Gen::Uniform => format!("luma_uniform(luma_threefry({index}, {a}, {b}).x)"),
        Gen::Normal => format!("luma_normal(luma_threefry({index}, {a}, {b}))"),
        Gen::Index => match ty {
            ElemTy::U32 => index.to_owned(),
            ElemTy::F32 | ElemTy::F64 | ElemTy::Df32 => format!("f32({index})"),
        },
        Gen::RangePow2 => match ty {
            ElemTy::U32 => format!("(({index} >> {a}) & {b})"),
            ElemTy::F32 | ElemTy::F64 | ElemTy::Df32 => format!("f32(({index} >> {a}) & {b})"),
        },
        Gen::RangeConst { stride, size } => {
            let coord = if stride == 1 { format!("({index} % {size}u)") } else { format!("(({index} / {stride}u) % {size}u)") };
            match ty {
                ElemTy::U32 => coord,
                ElemTy::F32 | ElemTy::F64 | ElemTy::Df32 => format!("f32({coord})"),
            }
        }
        Gen::Linspace => format!("({a} + f32({index}) * {b})"),
    }
}

/// Rotation amounts of Threefry-2x32 (Random123).
const THREEFRY_ROTATIONS: [u32; 8] = [13, 15, 26, 6, 17, 29, 16, 24];

/// The WGSL helpers of the random generators, if any instruction of `programs` uses them.
pub(super) fn generator_helpers<'a>(programs: impl IntoIterator<Item = &'a [Instr]>) -> String {
    let random = programs.into_iter().flatten().any(|instr| {
        matches!(instr.op.kind, OpKind::Generate(Gen::Bits | Gen::Uniform | Gen::Normal))
    });
    if !random {
        return String::new();
    }
    let keys = ["k0", "k1", "k2"];
    let mut src =
        String::from("fn luma_threefry(c: u32, k0: u32, k1: u32) -> vec2<u32> {\n    let k2 = 0x1BD11BDAu ^ k0 ^ k1;\n    var x0 = c + k0;\n    var x1 = k1;\n");
    for round in 0..20 {
        let r = THREEFRY_ROTATIONS[round % 8];
        let _ = writeln!(src, "    x0 = x0 + x1; x1 = ((x1 << {r}u) | (x1 >> {}u)) ^ x0;", 32 - r);
        if round % 4 == 3 {
            let s = round / 4 + 1;
            let _ = writeln!(src, "    x0 = x0 + {}; x1 = x1 + {} + {s}u;", keys[s % 3], keys[(s + 1) % 3]);
        }
    }
    src.push_str(
        "    return vec2<u32>(x0, x1);
}
fn luma_uniform(w: u32) -> f32 { return f32(w >> 8u) * 5.9604645e-8; }
fn luma_normal(w: vec2<u32>) -> f32 {
    let u1 = (f32(w.x >> 8u) + 1.0) * 5.9604645e-8;
    return sqrt(-2.0 * log(u1)) * cos(6.2831855 * f32(w.y >> 8u) * 5.9604645e-8);
}
",
    );
    src
}

/// Whether a program uses the element's index in the whole array, so a kernel run per segment
/// needs the segment's first index.
pub(super) fn generates(key: &FusedKey) -> bool {
    key.instrs.iter().chain(key.subs.iter().flatten()).any(|instr| matches!(instr.op.kind, OpKind::Generate(_)))
}

/// The index element `i` of a shift reads; `s` names the scalar shift of [`ShiftKind::Linear`],
/// `i` the element the shift is relative to and `total` the array length (scalar operand 2).
fn shift_index(kind: ShiftKind, s: &str, i: &str, total: &str) -> String {
    match kind {
        ShiftKind::Linear => format!("select({i} - {s}, {i} + ({total} - {s}), {i} < {s})"),
        ShiftKind::View { .. } | ShiftKind::ParamView { .. } => unreachable!("views are indexed by their own functions"),
        ShiftKind::Take { len } => format!("min({s}, {}u)", len.saturating_sub(1)),
        ShiftKind::ParamTake { max_slot } => format!("min({s}, p.s{max_slot})"),
        ShiftKind::Dims { dims, shifts } => {
            let [d0, d1, d2, _] = dims;
            let strides = [1, d0, d0 * d1, d0 * d1 * d2];
            let mut terms = Vec::new();
            for axis in 0..4 {
                let (dim, shift, stride) = (dims[axis], shifts[axis], strides[axis]);
                if dim == 1 {
                    continue;
                }
                let coord = match (stride, axis) {
                    (1, _) => format!("({i} % {dim}u)"),
                    (_, 3) => format!("({i} / {stride}u)"),
                    _ => format!("(({i} / {stride}u) % {dim}u)"),
                };
                let wrapped = if shift == 0 {
                    coord
                } else {
                    format!("select({coord} - {shift}u, {coord} + {}u, {coord} < {shift}u)", dim - shift)
                };
                terms.push(if stride == 1 { wrapped } else { format!("{wrapped} * {stride}u") });
            }
            if terms.is_empty() {
                "0u".to_owned()
            } else {
                terms.join(" + ")
            }
        }
    }
}

/// Whether input `k` is loaded at the element being computed (as opposed to only through
/// shifts), so the kernel declares its buffer as `in{k}`. The launch code binds by the same rule.
pub(super) fn direct(instrs: &[Instr], k: u32) -> bool {
    instrs.iter().any(|instr| {
        let skip = usize::from(matches!(instr.op.kind, OpKind::Shift(_)));
        instr.args.iter().skip(skip).any(|arg| *arg == Ref::Input(k))
    })
}

/// Whether the kernel for `key` reads a scalar from the uniform `p`. A tiled kernel (see
/// [`Tile`]) bakes the amount of a shift of a sub-program into its source, so without another
/// scalar the uniform stays unused: the kernel has no binding for it (only the globals an entry
/// point uses count) and the launch must not bind it. The generated source and the launch code
/// ([`super::launch::params`]) both decide by this rule. The guard (read by the arithmetic
/// library, see [`library_for`]) is a scalar that no instruction names.
pub(super) fn reads_scalars(key: &FusedKey) -> bool {
    let programs = std::iter::once(key.instrs.as_slice()).chain(key.subs.iter().map(Vec::as_slice));
    key.guard.is_some()
        || key.params.as_ref().is_some_and(|params| params.slots)
        || programs.enumerate().any(|(p, instrs)| {
            instrs.iter().enumerate().any(|(at, instr)| {
                let baked = key.tile.as_ref().is_some_and(|tile| tile.offsets.get(p).and_then(|row| row.get(at)).is_some_and(Option::is_some));
                !baked && instr.args.iter().any(|arg| matches!(arg, Ref::Scalar(_)))
            })
        })
}

/// Workgroup memory a stack kernel may use to write its cells together.
const STACK_STAGE_BYTES: u64 = 16384;

/// Whether the kernel of a [`Tail::Stack`] key writes through workgroup memory: when it holds all
/// the parts of an array stacked along the fastest axis, an invocation's values are `total`
/// neighbours in the result and the workgroup's cells are one contiguous stretch of it; written
/// by the invocations of the workgroup in order it is a coalesced store instead of one stride
/// of `total` elements per invocation. Needs the stretch (`wg * total` elements of `ty`) to fit.
fn stack_stages(key: &FusedKey, wg: u32, ty: &str) -> bool {
    let Tail::Stack { total, first: 0, block: 1 } = key.tail else { return false };
    let bytes = if ty == "f32" || ty == "u32" || ty == "i32" { 4 } else { 8 };
    key.outputs.len() == total as usize && total > 1 && u64::from(wg) * u64::from(total) * bytes <= STACK_STAGE_BYTES
}

/// The WGSL source of the kernel for `key`, with the workgroup shapes `tuning` gives.
/// `segment_len` (a power of two) is the elements per segment of the arrays the segmented
/// variant (`key.spans` not empty) runs on. `lowered` is the lowering of `key` (see
/// `jit::inline`) if it [`inlines`](FusedKey::inlines): the body then computes every
/// sub-program at the indices it is read at, and no `sub{k}` functions exist; for a
/// [`Lowered::Staged`] key (`key.staging` set) the staged sub-programs are computed into
/// workgroup memory instead.
pub(crate) fn source(key: &FusedKey, tuning: &Tuning, segment_len: u32, lowered: Option<&Lowered>) -> String {
    let inlined = match lowered {
        Some(Lowered::Plain(inlined)) => Some(inlined),
        Some(Lowered::Staged(programs)) => Some(&programs.main),
        None => None,
    };
    let wg = tuning.elementwise_workgroup;
    let reduce = matches!(key.tail, Tail::Reduce(_));
    let segmented = !key.spans.is_empty();
    let slots = |k: usize| key.spans.get(k).copied().unwrap_or(0);
    let based = generates(key) && !segmented;
    let uniform = reduce || (!key.scalar_tys.is_empty() && reads_scalars(key)) || segmented || based;
    let mut src = String::new();

    if uniform {
        src.push_str("struct Params { ");
        if reduce {
            src.push_str("n: u32, ");
        }
        if segmented {
            src.push_str("base: u32, total: u32, ");
            for k in 0..key.input_tys.len() {
                for s in 0..slots(k) {
                    let _ = write!(src, "c{k}_{s}: u32, ");
                }
            }
        } else if based {
            src.push_str("base: u32, ");
        }
        for (k, ty) in key.scalar_tys.iter().enumerate() {
            let _ = write!(src, "s{k}: {}, ", ty.wgsl());
        }
        if let Tail::Reduce(ops) = key.tail {
            src.push_str(&ops.epilogue().fields());
        }
        src.push_str("}\n");
    }
    let out_ty = |at: usize| key.instrs.get(at).map_or(ElemTy::F32, |instr| instr.op.out_ty()).wgsl();
    let outputs: Vec<usize> = match key.tail {
        Tail::Store | Tail::StoreMapped(_) | Tail::ParamStoreMapped { .. } => key.outputs.iter().map(|&at| at as usize).collect(),
        Tail::Scatter { .. } | Tail::ParamScatter { .. } | Tail::Stack { .. } => key.outputs.iter().take(1).map(|&at| at as usize).collect(),
        Tail::Reduce(_) => vec![key.instrs.len().saturating_sub(1)],
    };
    for (j, &at) in outputs.iter().enumerate() {
        // A reduction's single output is `out`, as `reduce_main` expects.
        let name = if reduce { "out".to_owned() } else { format!("out{j}") };
        let _ = writeln!(src, "@group(1) @binding({j}) var<storage, read_write> {name}: array<{}>;", out_ty(at));
    }
    let mut binding = outputs.len();
    for (k, ty) in key.input_tys.iter().enumerate() {
        if key.in_place.iter().any(|&(paired, _)| paired == k as u32) {
            continue;
        }
        if !segmented || direct(&key.instrs, k as u32) {
            let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, read> in{k}: array<{}>;", ty.wgsl());
            binding += 1;
        }
        for s in 0..slots(k) {
            let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, read> in{k}_{s}: array<{}>;", ty.wgsl());
            binding += 1;
        }
    }
    if uniform {
        let _ = writeln!(src, "@group(1) @binding({binding}) var<uniform> p: Params;");
    }
    if segmented {
        for (k, ty) in key.input_tys.iter().enumerate() {
            let m = slots(k);
            if m == 0 {
                continue;
            }
            // `segment_len` counts 4-byte words: an 8-byte element halves it.
            let per = segment_len / (ty.bytes() as u32 / 4).max(1);
            let (log, mask) = (per.trailing_zeros(), per.wrapping_sub(1));
            let _ = writeln!(src, "fn ld{k}(g: u32) -> {} {{\n    let c = g >> {log}u; let o = g & {mask}u;", ty.wgsl());
            for s in 0..m - 1 {
                let _ = writeln!(src, "    if (c == p.c{k}_{s}) {{ return in{k}_{s}[o]; }}");
            }
            let _ = writeln!(src, "    return in{k}_{}[o];\n}}", m - 1);
        }
    }
    let views = match lowered {
        Some(Lowered::Plain(inlined)) => views_of_inlined(key, &[inlined]),
        Some(Lowered::Staged(programs)) => views_of_inlined(key, &programs.bodies().collect::<Vec<_>>()),
        None => views_of(key),
    };
    // The lowered instruction holding the value of main-program instruction `at`.
    let value = |at: usize| inlined.and_then(|inlined| inlined.main.get(at)).map_or(at, |&lowered| lowered as usize);
    // A reduction of `Df32` brings the library itself (see `reduce_main`).
    let reduces_df = reduce && out_ty(outputs[0]) == "vec2<f32>";
    if !reduces_df {
        src.push_str(&library_for(key, tuning));
    }
    src.push_str(&generator_helpers(std::iter::once(key.instrs.as_slice()).chain(key.subs.iter().map(Vec::as_slice))));
    src.push_str(&view_helpers(&views));
    if key.params.is_some() {
        src.push_str(&params::view_helpers(&params::views_used(key, inlined)));
    }
    let names = Names { views: &views, in_place: &key.in_place };
    if key.tile.is_none() && lowered.is_none() {
        src.push_str(&subs(key, &views, &key.in_place));
    }

    let index = match (segmented, based) {
        (true, _) | (false, true) => "gi",
        (false, false) => "i",
    };
    let global = if segmented || based { "    let gi = p.base + i;\n" } else { "" };
    match key.tail {
        Tail::Store | Tail::StoreMapped(_) | Tail::ParamStoreMapped { .. } | Tail::Scatter { .. } | Tail::ParamScatter { .. } | Tail::Stack { .. } => {
            let stores: String = match key.tail {
                Tail::StoreMapped(map) => {
                    let n = views.iter().position(|&known| known == (map, Edge::None)).unwrap_or(0);
                    outputs.iter().enumerate().map(|(j, &at)| format!("    out{j}[view{n}(i)] = t{};\n", value(at))).collect()
                }
                Tail::ParamStoreMapped { id, .. } => {
                    outputs.iter().enumerate().map(|(j, &at)| format!("    out{j}[pview{id}(i)] = t{};\n", value(at))).collect()
                }
                Tail::Scatter { .. } | Tail::ParamScatter { .. } => {
                    let len = match key.tail {
                        Tail::ParamScatter { len_slot } => format!("p.s{len_slot}"),
                        Tail::Scatter { len } => format!("{len}u"),
                        _ => unreachable!(),
                    };
                    let (stored, index) = (key.outputs.first().copied().unwrap_or(0), key.outputs.get(1).copied().unwrap_or(0));
                    format!("    let j = t{};\n    if (j < {len}) {{ out0[j] = t{}; }}\n", value(index as usize), value(stored as usize))
                }
                Tail::Stack { total, .. } if stack_stages(key, wg, out_ty(outputs[0])) => {
                    let mut stores = String::new();
                    for (j, &at) in key.outputs.iter().enumerate() {
                        let _ = writeln!(stores, "        luma_stage[lid.x * {total}u + {j}u] = t{};", value(at as usize));
                    }
                    stores
                }
                Tail::Stack { total, first, block } => {
                    let mut stores = if block == 1 {
                        format!("    let o = i * {total}u + {first}u;\n")
                    } else {
                        format!("    let q = i / {block}u;\n    let o = q * {}u + (i - q * {block}u) + {}u;\n", u64::from(total) * u64::from(block), u64::from(first) * u64::from(block))
                    };
                    for (j, &at) in key.outputs.iter().enumerate() {
                        let _ = writeln!(stores, "    out0[o + {}u] = t{};", j as u64 * u64::from(block), value(at as usize));
                    }
                    stores
                }
                _ => outputs.iter().enumerate().map(|(j, &at)| format!("    out{j}[i] = t{};\n", value(at))).collect(),
            };
            if let Some(tile) = &key.tile {
                src.push_str(&tiled_main(key, tile, based, &stores, &names));
                return src;
            }
            if let (Some(Lowered::Staged(programs)), Some(staging)) = (lowered, &key.staging) {
                src.push_str(&staged_main(key, staging, programs, based, &stores, &names));
                return src;
            }
            if let Tail::Stack { total, .. } = key.tail
                && stack_stages(key, wg, out_ty(outputs[0]))
            {
                // Each invocation computes its cell's `total` values into workgroup memory, then
                // the workgroup writes its contiguous stretch of the result, coalesced.
                let _ = write!(
                    src,
                    "
var<workgroup> luma_stage: array<{}, {}>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let base = gid.x - lid.x + gid.y * nwg.x * {wg}u;
    let i = base + lid.x;
    if (i < luma.size.x) {{
{global}{}{stores}    }}
    workgroupBarrier();
    let cells = min({wg}u, luma.size.x - base);
    for (var e = lid.x; e < cells * {total}u; e = e + {wg}u) {{
        out0[base * {total}u + e] = luma_stage[e];
    }}
}}
",
                    out_ty(outputs[0]),
                    wg * total,
                    match inlined {
                        Some(inlined) => lowered_body(key, inlined, &names, index, None),
                        None => body(&key.instrs, &key.input_tys, &names, index, segmented),
                    }
                );
                return src;
            }
            let _ = write!(
                src,
                "
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {wg}u;
    if (i >= luma.size.x) {{ return; }}
{global}{}{stores}}}
",
                match inlined {
                    Some(inlined) => lowered_body(key, inlined, &names, index, None),
                    None => body(&key.instrs, &key.input_tys, &names, index, segmented),
                }
            );
        }
        Tail::Reduce(ops) => {
            let (ty, last) = (out_ty(outputs[0]), outputs[0]);
            let guard = format!("p.s{}", key.guard.unwrap_or(0));
            let elem = if reduces_df { ReduceElem::Df32 { guard: &guard, fma: tuning.df_fma } } else { ReduceElem::Plain(ty) };
            let _ = write!(
                src,
                "
fn luma_eval(i: u32) -> {ty} {{
{global}{}    return t{};
}}
{}",
                match inlined {
                    Some(inlined) => lowered_body(key, inlined, &Names { views: &views, in_place: &[] }, index, None),
                    None => body(&key.instrs, &key.input_tys, &Names { views: &views, in_place: &[] }, index, segmented),
                },
                value(last),
                reduce_main(elem, ops, false, "luma_eval", tuning)
            );
        }
    }
    src
}

/// A kernel running `p.steps` iterations of the store program `key` per element, with the
/// state in registers. `states[k]` is the state that input `k` reads (`None`: a loop-invariant
/// input). Bindings: one output and one initial value per state, the invariant inputs, `p`.
pub(super) fn loop_source(key: &FusedKey, states: &[Option<usize>], tuning: &Tuning) -> String {
    let wg = tuning.elementwise_workgroup;
    let count = key.outputs.len();
    // The type of the states (`run_loop_fused` checked that they agree).
    let state_ty = key.outputs.first().and_then(|&at| key.instrs.get(at as usize)).map_or("f32", |instr| instr.op.out_ty().wgsl());
    let mut src = String::from("struct Params { steps: u32, ");
    for (k, ty) in key.scalar_tys.iter().enumerate() {
        let _ = write!(src, "s{k}: {}, ", ty.wgsl());
    }
    src.push_str("}\n");
    let mut binding = 0;
    for (role, access) in [("out", "read_write"), ("init", "read")] {
        for j in 0..count {
            let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, {access}> {role}{j}: array<{state_ty}>;");
            binding += 1;
        }
    }
    for (k, ty) in key.input_tys.iter().enumerate() {
        if states.get(k).copied().flatten().is_none() {
            let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, read> in{k}: array<{}>;", ty.wgsl());
            binding += 1;
        }
    }
    let _ = writeln!(src, "@group(1) @binding({binding}) var<uniform> p: Params;");
    // The tests of the class of a value (`luma_bits`) come with the library of the guard.
    src.push_str(&library_for(key, tuning));
    src.push_str(&generator_helpers([key.instrs.as_slice()]));
    let _ = write!(
        src,
        "
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {wg}u;
    if (i >= luma.size.x) {{ return; }}
"
    );
    for j in 0..count {
        let _ = writeln!(src, "    var s{j}: {state_ty} = init{j}[i];");
    }
    for (k, ty) in key.input_tys.iter().enumerate() {
        if states.get(k).copied().flatten().is_none() && direct(&key.instrs, k as u32) {
            let _ = writeln!(src, "    let a{k}: {} = in{k}[i];", ty.wgsl());
        }
    }
    src.push_str("    for (var step = 0u; step < p.steps; step = step + 1u) {\n");
    for (k, state) in states.iter().enumerate() {
        if let Some(j) = state {
            let _ = writeln!(src, "        let a{k}: {state_ty} = s{j};");
        }
    }
    for (k, instr) in key.instrs.iter().enumerate() {
        let _ = writeln!(src, "        let t{k}: {} = {};", instr.op.out_ty().wgsl(), expression(instr, "i", false, &NO_NAMES));
    }
    for (j, at) in key.outputs.iter().enumerate() {
        let _ = writeln!(src, "        s{j} = t{at};");
    }
    src.push_str("    }\n");
    for j in 0..count {
        let _ = writeln!(src, "    out{j}[i] = s{j};");
    }
    src.push_str("}\n");
    src
}

/// The loads of the inputs at element `i` and the instructions, as `let` bindings. The inputs
/// `names.in_place` lists are read from the output buffers, before they are overwritten.
/// `index` and `segmented` are those of [`expression`].
fn body(instrs: &[Instr], input_tys: &[ElemTy], names: &Names, index: &str, segmented: bool) -> String {
    body_with(instrs, input_tys, names, index, segmented, &|_, _| None)
}

/// [`body`], where `staged(k, instr)` may replace the expression of instruction `k`.
fn body_with(
    instrs: &[Instr],
    input_tys: &[ElemTy],
    names: &Names,
    index: &str,
    segmented: bool,
    staged: &dyn Fn(usize, &Instr) -> Option<String>,
) -> String {
    let mut src = String::new();
    // Inputs read only through shifts are not loaded at `i`.
    for (k, ty) in input_tys.iter().enumerate() {
        if direct(instrs, k as u32) {
            let buffer = match names.in_place.iter().find(|&&(input, _)| input == k as u32) {
                Some(&(_, j)) => format!("out{j}"),
                None => format!("in{k}"),
            };
            let _ = writeln!(src, "    let a{k}: {} = {buffer}[i];", ty.wgsl());
        }
    }
    for (k, instr) in instrs.iter().enumerate() {
        let value = staged(k, instr).unwrap_or_else(|| expression(instr, index, segmented, names));
        let _ = writeln!(src, "    let t{k}: {} = {value};", instr.op.out_ty().wgsl());
    }
    src
}

/// The name of an operand of a lowered instruction computed at context `at`: an input is the
/// value loaded at that context's index.
fn flat_name(reference: Ref, ty: ElemTy, at: u32) -> String {
    match reference {
        Ref::Input(k) if at != 0 => format!("a{k}_{at}"),
        other => name(other, ty),
    }
}

/// The body of a lowered key (see [`Inlined`]): the index of every context, the inputs loaded at
/// them, then the instructions, each defined (`t{n}`) after everything it reads. A `Select` the
/// lowering marks as a branch is an `if`/`else` that computes the instructions only its own
/// operand needs. `names` and `index` are those of [`expression`].
fn lowered_body(key: &FusedKey, inlined: &Inlined, names: &Names, index: &str, staging: Option<&Staging>) -> String {
    struct Emit<'a> {
        key: &'a FusedKey,
        inlined: &'a Inlined,
        names: &'a Names<'a>,
        index: &'a str,
        staging: Option<&'a Staging>,
        src: String,
        /// The contexts whose index is declared, and the inputs loaded at them, in the block
        /// being written (a block that ends forgets what it declared).
        defined: Vec<bool>,
        loaded: Vec<(u32, u32)>,
        /// The instructions of every region not written yet, in order.
        members: FxMap<Region, Vec<u32>>,
    }
    impl Emit<'_> {
        /// The index of context `c`.
        fn index_of(&self, c: u32) -> String {
            if c == 0 { self.index.to_owned() } else { format!("i{c}") }
        }

        /// The buffer input `k` is read from.
        fn buffer(&self, k: u32) -> String {
            match self.names.in_place.iter().find(|&&(input, _)| input == k) {
                Some(&(_, j)) => format!("out{j}"),
                None => format!("in{k}"),
            }
        }

        /// Declares the index of context `c`, after its parents'.
        fn define(&mut self, c: u32) {
            let Some(ctx) = self.inlined.ctxs.get(c as usize) else { return };
            if c == 0 || self.defined[c as usize] {
                return;
            }
            self.defined[c as usize] = true;
            self.define(ctx.parent);
            let from = self.index_of(ctx.parent);
            let expr = match ctx.kind {
                ShiftKind::View { map, edge } => {
                    let n = self.names.views.iter().position(|&known| known == (map, edge));
                    format!("view{}({from})", n.map_or_else(|| "_missing".to_owned(), |n| n.to_string()))
                }
                ShiftKind::ParamView { id, .. } => format!("pview{id}({from})"),
                kind => {
                    if let Ref::Input(k) = ctx.args[0] {
                        self.load(k, ctx.parent);
                    }
                    let [s, total] = ctx.args.map(|reference| flat_name(reference, ElemTy::U32, ctx.parent));
                    shift_index(kind, &s, &from, &total)
                }
            };
            let _ = writeln!(self.src, "    let i{c}: u32 = {expr};");
        }

        /// Declares input `k` read at the index of context `c`.
        fn load(&mut self, k: u32, c: u32) {
            if self.loaded.contains(&(k, c)) {
                return;
            }
            self.loaded.push((k, c));
            self.define(c);
            let ty = self.key.input_tys.get(k as usize).map_or("f32", |ty| ty.wgsl());
            let buffer = self.buffer(k);
            if c == 0 {
                let _ = writeln!(self.src, "    let a{k}: {ty} = {buffer}[i];");
            } else {
                let _ = writeln!(self.src, "    let a{k}_{c}: {ty} = {buffer}[i{c}];");
            }
        }

        /// Writes the instructions computed in `region`.
        fn region(&mut self, region: Region) {
            for n in self.members.remove(&region).unwrap_or_default() {
                self.instruction(n as usize);
            }
        }

        /// How far the cell of context `c` is from the cell of the program: the starts of the
        /// translation views on the way (what `staging` lets a staged read be through).
        fn translation(&self, mut c: u32) -> [i32; 2] {
            let mut offset = [0, 0];
            while c != 0 {
                let Some(ctx) = self.inlined.ctxs.get(c as usize) else { break };
                if let ShiftKind::View { map, .. } = ctx.kind {
                    offset[0] += map.axes[0].start;
                    offset[1] += map.axes[1].start;
                }
                c = ctx.parent;
            }
            offset
        }

        /// Writes instruction `n`.
        fn instruction(&mut self, n: usize) {
            let inlined = self.inlined;
            let Some(flat) = inlined.instrs.get(n) else { return };
            // A read from workgroup memory needs no index: the cell is `(lx, ly)` plus offsets.
            let staged_read = match (flat.op.kind, flat.args[0]) {
                (OpKind::Shift(_), Ref::Sub(s)) => self.staging.is_some_and(|layout| layout.find(s).is_some()),
                _ => false,
            };
            if !staged_read {
                self.define(flat.at);
            }
            if inlined.branch.get(n).copied().unwrap_or(false) {
                return self.branch(n);
            }
            let op = flat.op;
            let value = match op.kind {
                OpKind::Shift(kind) => {
                    self.define(flat.child);
                    let at = self.index_of(flat.child);
                    let read = match flat.args[0] {
                        Ref::Input(k) => format!("{}[{at}]", self.buffer(k)),
                        Ref::Tmp(v) => format!("t{v}"),
                        Ref::Sub(s) => match (self.staging.and_then(|layout| layout.find(s)), kind) {
                            // A read from workgroup memory: the region's cell `(lx, ly)` plus the
                            // offsets of the contexts the read is in and of the read itself.
                            (Some(entry), ShiftKind::View { map, .. }) => {
                                let base = self.translation(flat.at);
                                let (dx, dy) = (map.axes[0].start + base[0] - entry.lo[0], map.axes[1].start + base[1] - entry.lo[1]);
                                format!("w{s}[u32(lx + ({dx})) + u32(ly + ({dy})) * {}u]", entry.width())
                            }
                            _ => format!("{}[{at}]", name(Ref::Sub(s), ElemTy::U32)),
                        },
                        other => format!("{}[{at}]", name(other, ElemTy::U32)),
                    };
                    match kind {
                        ShiftKind::View { map, edge: Edge::Fill } => {
                            let fill = flat_name(flat.args[1], op.operand_ty(1), flat.at);
                            let view = self.names.views.iter().position(|&known| known == (map, Edge::Fill));
                            let inside = format!("view{}_in({})", view.map_or_else(|| "_missing".to_owned(), |n| n.to_string()), self.index_of(flat.at));
                            format!("select({fill}, {read}, {inside})")
                        }
                        ShiftKind::ParamView { id, spec } if spec.edge == Edge::Fill => {
                            let fill = flat_name(flat.args[1], op.operand_ty(1), flat.at);
                            format!("select({fill}, {read}, pview{id}_in({}))", self.index_of(flat.at))
                        }
                        _ => read,
                    }
                }
                _ => {
                    for reference in flat.args.iter().take(op.kind.arity()) {
                        if let Ref::Input(k) = *reference {
                            self.load(k, flat.at);
                        }
                    }
                    let [a, b, c] = [0, 1, 2].map(|i| flat_name(flat.args[i], op.operand_ty(i), flat.at));
                    operation(op, &a, &b, &c, &self.index_of(flat.at)).unwrap_or_default()
                }
            };
            let _ = writeln!(self.src, "    let t{n}: {} = {value};", op.out_ty().wgsl());
        }

        /// Writes the `Select` `n` as `if`/`else`, each branch computing its own instructions
        /// before it takes its operand.
        fn branch(&mut self, n: usize) {
            let inlined = self.inlined;
            let Some(flat) = inlined.instrs.get(n) else { return };
            let mask = flat_name(flat.args[0], ElemTy::U32, flat.at);
            let _ = writeln!(self.src, "    var t{n}: {};\n    if ({mask} != 0u) {{", flat.op.out_ty().wgsl());
            for (side, slot) in [(Region::Taken(n as u32), 1), (Region::Skipped(n as u32), 2)] {
                if slot == 2 {
                    self.src.push_str("    } else {\n");
                }
                let outer = (self.defined.clone(), self.loaded.clone());
                self.region(side);
                if let Ref::Input(k) = flat.args[slot] {
                    self.load(k, flat.at);
                }
                let value = flat_name(flat.args[slot], flat.op.operand_ty(slot), flat.at);
                let _ = writeln!(self.src, "    t{n} = {value};");
                (self.defined, self.loaded) = outer;
            }
            self.src.push_str("    }\n");
        }
    }

    let mut members: FxMap<Region, Vec<u32>> = FxMap::default();
    for (n, &region) in inlined.region.iter().enumerate() {
        members.entry(region).or_default().push(n as u32);
    }
    let mut emit = Emit {
        key,
        inlined,
        names,
        index,
        staging,
        src: String::new(),
        defined: vec![false; inlined.ctxs.len()],
        loaded: Vec::new(),
        members,
    };
    emit.region(Region::Top);
    emit.src
}

/// The workgroup-memory staging of the sub-programs of a store kernel: each is computed once
/// per cell of its region (the tile plus a halo) and the shifts read it there. Part of the
/// [`FusedKey`] because the generated source bakes in the offsets, which for a
/// [`ShiftKind::Linear`] shift come from a scalar value.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Tile {
    /// Width and height of the array (`h == 1` for 1D).
    pub(crate) dims: (u32, u32),
    /// Output cells per workgroup.
    pub(crate) tile: (u32, u32),
    /// Per sub-program, the halo (cells per side and axis) its region extends beyond the tile.
    pub(crate) need: Vec<(u32, u32)>,
    /// Per program (0 = main, `1 + k` = `subs[k]`) and instruction, the cell offset (x, y) a
    /// shift of a sub-program reads; `None` for every other instruction.
    pub(crate) offsets: Vec<Vec<Option<(i32, i32)>>>,
}

/// The staging of `key` on arrays of `len` elements, if its sub-programs are read through
/// small shifts only and their regions fit in workgroup memory. `None` keeps the sub-programs
/// as functions called per shift.
pub(super) fn plan_tile(key: &FusedKey, scalars: &[u32], len: usize, tuning: &Tuning) -> Option<Tile> {
    // Staged values are 4 bytes wide; programs of the 8-byte types call their sub-programs.
    if key.tail != Tail::Store || key.subs.is_empty() || !key.spans.is_empty() || !key.narrow() {
        return None;
    }
    let programs: Vec<&[Instr]> = std::iter::once(key.instrs.as_slice()).chain(key.subs.iter().map(Vec::as_slice)).collect();
    let mut dims = None;
    for instr in programs.iter().copied().flatten() {
        if let OpKind::Shift(ShiftKind::Dims { dims: d, .. }) = instr.op.kind {
            if d[2] != 1 || d[3] != 1 || dims.is_some_and(|known| known != (d[0], d[1])) {
                return None;
            }
            dims = Some((d[0], d[1]));
        }
    }
    let (w, h) = dims.unwrap_or((u32::try_from(len).ok()?, 1));
    if u64::from(w) * u64::from(h) != len as u64 || w == 0 || h == 0 {
        return None;
    }

    let mut offsets = Vec::with_capacity(programs.len());
    for (p, instrs) in programs.iter().enumerate() {
        let mut row = Vec::with_capacity(instrs.len());
        for instr in *instrs {
            let skip = usize::from(matches!(instr.op.kind, OpKind::Shift(_)));
            if instr.args.iter().skip(skip).any(|arg| matches!(arg, Ref::Sub(_))) {
                return None;
            }
            let at = match (instr.op.kind, instr.args[0]) {
                // Sub-programs come before the programs reading them.
                (OpKind::Shift(_), Ref::Sub(j)) if p > 0 && j as usize + 1 >= p || j as usize >= key.subs.len() => return None,
                (OpKind::Shift(kind), Ref::Sub(_)) => {
                    let (dx, dy) = match kind {
                        ShiftKind::View { .. } | ShiftKind::Take { .. } | ShiftKind::ParamView { .. } | ShiftKind::ParamTake { .. } => return None,
                        ShiftKind::Dims { shifts, .. } => (signed(shifts[0], w), signed(shifts[1], h)),
                        ShiftKind::Linear => {
                            let Ref::Scalar(s) = instr.args[1] else { return None };
                            let amount = *scalars.get(s as usize)?;
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
                    if ox.unsigned_abs() > STENCIL_RADIUS || oy.unsigned_abs() > STENCIL_RADIUS {
                        return None;
                    }
                    Some((ox, oy))
                }
                _ => None,
            };
            row.push(at);
        }
        offsets.push(row);
    }

    // Consumers first: the main program, then the sub-programs from the last (dependencies come first).
    let mut need = vec![(0u32, 0u32); key.subs.len()];
    for p in std::iter::once(0).chain((1..programs.len()).rev()) {
        let own = if p == 0 { (0, 0) } else { need[p - 1] };
        for (instr, at) in programs[p].iter().zip(&offsets[p]) {
            if let (Some((ox, oy)), Ref::Sub(j)) = (at, instr.args[0]) {
                let slot = &mut need[j as usize];
                *slot = (slot.0.max(own.0 + ox.unsigned_abs()), slot.1.max(own.1 + oy.unsigned_abs()));
            }
        }
    }

    let (mut tx, mut ty) = if h == 1 { (tuning.stencil_tile_1d, 1) } else { tuning.stencil_tile_2d };
    let bytes = |tx: u32, ty: u32| need.iter().map(|&(nx, ny)| (tx + 2 * nx) * (ty + 2 * ny) * 4).sum::<u32>();
    while bytes(tx, ty) > tuning.stencil_tile_bytes {
        if h == 1 || tx >= ty {
            tx /= 2;
        } else {
            ty /= 2;
        }
        if tx < 4 || (h > 1 && ty < 4) {
            return None;
        }
    }
    Some(Tile { dims: (w, h), tile: (tx, ty), need, offsets })
}

/// The kernel of `key` with its sub-programs staged as `tile` says, running one workgroup per
/// tile of the array. `based` and `stores` are those of the plain store kernel.
fn tiled_main(key: &FusedKey, tile: &Tile, based: bool, stores: &str, names: &Names) -> String {
    let ((w, h), (tx, ty)) = (tile.dims, tile.tile);
    let region = |j: usize| (tx + 2 * tile.need[j].0, ty + 2 * tile.need[j].1);
    let (halo_x, halo_y) = tile.need.iter().fold((0, 0), |(x, y), &(nx, ny)| (x.max(nx), y.max(ny)));
    let (index, global) = if based { ("gi", "        let gi = p.base + i;\n") } else { ("i", "") };
    // Cell (lx, ly) relative to the tile origin, as a global element.
    let element = if h == 1 {
        format!("wrap(i32(tile_x) + lx, {w}u)")
    } else {
        format!("wrap(i32(tile_x) + lx, {w}u) + wrap(i32(tile_y) + ly, {h}u) * {w}u")
    };
    let staged = |p: usize| {
        move |at: usize, instr: &Instr| -> Option<String> {
            let (ox, oy) = tile.offsets[p][at]?;
            let Ref::Sub(j) = instr.args[0] else { return None };
            let j = j as usize;
            let ((nx, ny), (lw, _)) = (tile.need[j], region(j));
            let row = if h == 1 { String::new() } else { format!(" + u32(ly + ({})) * {lw}u", oy + ny as i32) };
            Some(format!("w{j}[u32(lx + ({})){row}]", ox + nx as i32))
        }
    };

    let mut src = String::new();
    for (k, instrs) in key.subs.iter().enumerate() {
        let (lw, lh) = region(k);
        let ty = instrs.last().map_or(ElemTy::F32, |instr| instr.op.out_ty()).wgsl();
        let _ = writeln!(src, "var<workgroup> w{k}: array<{ty}, {}>;", lw * lh);
    }
    let _ = write!(
        src,
        "
{}
@compute @workgroup_size({WORKGROUP})
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let g = wid.x + wid.y * nwg.x;
    let tile_x = (g % {tiles_x}u) * {tx}u;
    let tile_y = (g / {tiles_x}u) * {ty}u;
",
        wrap_source(w >= tx + 2 * halo_x && (h == 1 || h >= ty + 2 * halo_y)),
        tiles_x = w.div_ceil(tx),
    );
    for (k, instrs) in key.subs.iter().enumerate() {
        let ((nx, ny), (lw, lh)) = (tile.need[k], region(k));
        let last = instrs.len().saturating_sub(1);
        let _ = write!(
            src,
            "    for (var e = lid; e < {}u; e = e + {WORKGROUP}u) {{
        let lx = i32(e % {lw}u) - {nx};
        let ly = i32(e / {lw}u) - {ny};
        let i = {element};
{global}{}        w{k}[e] = t{last};
    }}
    workgroupBarrier();
",
            lw * lh,
            body_with(instrs, &key.input_tys, &Names { views: names.views, in_place: &[] }, index, false, &staged(k + 1)),
        );
    }
    let _ = write!(
        src,
        "    for (var e = lid; e < {}u; e = e + {WORKGROUP}u) {{
        let lx = i32(e % {tx}u);
        let ly = i32(e / {tx}u);
        let x = tile_x + u32(lx);
        let y = tile_y + u32(ly);
        if (x < {w}u && y < {h}u) {{
            let i = x + y * {w}u;
{global}{}{stores}        }}
    }}
}}
",
        tx * ty,
        body_with(&key.instrs, &key.input_tys, names, index, false, &staged(0)),
    );
    src
}

/// The kernel of `key` with its sub-programs staged as `staging` says: one workgroup per tile of
/// output cells. Every level of staged sub-programs fills its workgroup-memory arrays `w{s}`
/// (each cell of its region, strided over the invocations; `programs.groups` are the lowered
/// bodies) before a barrier, then the main program reads them. `based` and `stores` are those
/// of the plain store kernel.
fn staged_main(key: &FusedKey, staging: &Staging, programs: &StagedPrograms, based: bool, stores: &str, names: &Names) -> String {
    let [w, h] = staging.dims;
    let [tx, ty] = staging.tile;
    let (index, global) = if based { ("gi", "            let gi = p.base + i;\n") } else { ("i", "") };
    let sub_names = Names { views: names.views, in_place: &[] };
    let mut src = String::new();
    for entry in &staging.subs {
        let ty = key.subs.get(entry.sub as usize).and_then(|program| program.last()).map_or(ElemTy::F32, |instr| instr.op.out_ty()).wgsl();
        let _ = writeln!(src, "var<workgroup> w{}: array<{ty}, {}>;", entry.sub, entry.cells());
    }
    let _ = write!(
        src,
        "
@compute @workgroup_size({WORKGROUP})
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let g = wid.x + wid.y * nwg.x;
    let tile_x = (g % {tiles_x}u) * {tx}u;
    let tile_y = (g / {tiles_x}u) * {ty}u;
",
        tiles_x = w.div_ceil(tx),
    );
    for level in staging.levels.windows(2) {
        for group in programs.groups.iter().filter(|group| group.members.first().is_some_and(|&first| (level[0] as usize..level[1] as usize).contains(&first))) {
            let Some(entry) = group.members.first().and_then(|&first| staging.subs.get(first)) else { continue };
            let [dw, dh] = entry.dims;
            let stores: String = group
                .members
                .iter()
                .zip(&group.values)
                .filter_map(|(&member, last)| Some(format!("            w{}[e] = t{last};\n", staging.subs.get(member)?.sub)))
                .collect();
            let _ = write!(
                src,
                "    for (var e = lid; e < {}u; e = e + {WORKGROUP}u) {{
        let lx = i32(e % {rw}u) + ({});
        let ly = i32(e / {rw}u) + ({});
        let x = i32(tile_x) + lx;
        let y = i32(tile_y) + ly;
        if (x >= 0 && x < {dw} && y >= 0 && y < {dh}) {{
            let i = u32(x) + u32(y) * {dw}u;
{global}{}{stores}        }}
    }}
",
                entry.cells(),
                entry.lo[0],
                entry.lo[1],
                lowered_body(key, &group.body, &sub_names, index, Some(staging)),
                rw = entry.width(),
            );
        }
        src.push_str("    workgroupBarrier();\n");
    }
    let _ = write!(
        src,
        "    for (var e = lid; e < {}u; e = e + {WORKGROUP}u) {{
        let lx = i32(e % {tx}u);
        let ly = i32(e / {tx}u);
        let x = tile_x + u32(lx);
        let y = tile_y + u32(ly);
        if (x < {w}u && y < {h}u) {{
            let i = x + y * {w}u;
{global}{}{stores}        }}
    }}
}}
",
        tx * ty,
        lowered_body(key, &programs.main, names, index, Some(staging)),
    );
    src
}

/// One function `sub{k}(i)` per sub-program: the value of a shifted expression at element `i`.
/// Programs with sub-programs never run segmented.
fn subs(key: &FusedKey, views: &[(ViewMap, Edge)], in_place: &[(u32, u32)]) -> String {
    let mut src = String::new();
    for (k, instrs) in key.subs.iter().enumerate() {
        let ty = instrs.last().map_or(ElemTy::F32, |instr| instr.op.out_ty()).wgsl();
        let last = instrs.len().saturating_sub(1);
        let _ = write!(
            src,
            "\nfn sub{k}(i: u32) -> {ty} {{\n{}    return t{last};\n}}\n",
            body(instrs, &key.input_tys, &Names { views, in_place }, "i", false)
        );
    }
    src
}
