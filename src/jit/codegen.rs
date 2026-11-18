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

use crate::builtin::{
    DF_MATH_LIBRARY, ReduceElem, class_operand, df_library, reduce_main, substitute,
};
use crate::gpu::hash::FxMap;
use crate::tuning::{STENCIL_RADIUS, Tuning};

use luma_core::kir::KernelModule;

use crate::error::Result;

use super::inline::{Inlined, Lowered, Region};
use super::kirgen;
use super::params;
use super::program::{FusedKey, Instr, Ref, Tail, calls_df_library, calls_df_math};
use super::staging::Staging;
use super::stencil::signed;
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
            ElemTy::F32 | ElemTy::U32 | ElemTy::I32 | ElemTy::Bool => format!("p.s{k}"),
        },
        Ref::Tmp(k) => format!("t{k}"),
        Ref::Sub(k) => format!("sub{k}"),
    }
}

/// The WGSL element type of a read-only storage buffer holding elements of `ty`.
///
/// With [`out_storage_ty`], [`load`], [`load_out`] and [`store`] this is the one place in the
/// code generator that knows how elements are laid out in memory (`ElemTy::layout`): a `bool` is
/// one bit of a `u32` word, element `i` being bit `i % 32` of word `i / 32` (and the bits past
/// the end of the array are 0), every other type is stored as itself. Registers hold
/// `ty.wgsl()`: a bool is a `u32` of 0 or 1 there.
pub(super) fn storage_ty(ty: ElemTy) -> &'static str {
    match ty {
        ElemTy::Bool => "u32",
        ElemTy::F32 | ElemTy::U32 | ElemTy::I32 | ElemTy::F64 | ElemTy::Df32 => ty.wgsl(),
    }
}

/// The WGSL element type of a read-write storage buffer holding elements of `ty`: the bits of a
/// `bool` array are written by atomic operations, because the elements of a word are written by
/// different invocations.
pub(super) fn out_storage_ty(ty: ElemTy) -> &'static str {
    match ty {
        ElemTy::Bool => "atomic<u32>",
        _ => storage_ty(ty),
    }
}

/// `index` as an operand of a binary operator.
fn operand(index: &str) -> String {
    if index.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') { index.to_owned() } else { format!("({index})") }
}

/// The WGSL expression of the element `index` of the read-only buffer `buffer`, as a register
/// value of type `ty` (see [`storage_ty`]).
pub(super) fn load(ty: ElemTy, buffer: &str, index: &str) -> String {
    match ty {
        ElemTy::Bool => {
            let j = operand(index);
            format!("(({buffer}[{j} >> 5u] >> ({j} & 31u)) & 1u)")
        }
        ElemTy::F32 | ElemTy::U32 | ElemTy::I32 | ElemTy::F64 | ElemTy::Df32 => format!("{buffer}[{index}]"),
    }
}

/// [`load`] from a read-write buffer (see [`out_storage_ty`]): a kernel that overwrites an input
/// reads it from the output buffer.
pub(super) fn load_out(ty: ElemTy, buffer: &str, index: &str) -> String {
    match ty {
        ElemTy::Bool => {
            let j = operand(index);
            format!("((atomicLoad(&{buffer}[{j} >> 5u]) >> ({j} & 31u)) & 1u)")
        }
        _ => load(ty, buffer, index),
    }
}

/// The WGSL expression of the element `index` of input `k`, read from where `names` says the
/// input is: its own buffer, or, for an input the kernel overwrites, the output buffer.
fn input_load(names: &Names, ty: ElemTy, k: u32, index: &str) -> String {
    match names.in_place.iter().find(|&&(input, _)| input == k) {
        Some(&(_, j)) => load_out(ty, &format!("out{j}"), index),
        None => load(ty, &format!("in{k}"), index),
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
        OpKind::Powf if ty == ElemTy::Df32 => format!("df_pow({a}, {b})"),
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
        Ref::Input(k) => input_load(names, instr.op.ty, k, &at),
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
pub(super) fn views_of_inlined(key: &FusedKey, lowered: &[&Inlined]) -> Vec<(ViewMap, Edge)> {
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
fn luma_wrap_i(c: i32, n: i32) -> i32 { let r = c % n; return select(r, r + n, r < 0); }
fn luma_wrap_mask(c: i32, mask: u32) -> i32 { return i32(u32(c) & mask); }
";

/// The WGSL functions of the division by a divisor that is only known when the kernel runs, from
/// the numbers of [`params::divisor_words`]. `luma_udiv` is for a dividend below `2^23` (an index
/// into an array of fewer elements): one `f32` product and a correction of one, with the
/// conversions between integer and `f32` made by adding `2^23`, which the shader compiler shares
/// between reads of several views as it does the multiplication by a constant.
/// A branch between this and the exact division for any dividend, or the division instruction
/// of a variable divisor (a software routine), costs 20 to 60 % of an arithmetic-bound kernel
/// (`euler/per_step` on 2000 x 2000 cells, RTX 3090) because the compiler no longer shares them;
/// so the kernel is made for one or the other: `luma_udiv_wide` is exact for any dividend. No
/// WGSL integer operation gives the high word of a product, so it is made of 16-bit halves
/// (about 20 operations in all, about 3 times `luma_udiv`).
const DIV_HELPERS: &str = "fn luma_udiv(n: u32, d: u32, r: u32) -> u32 {
    let f = bitcast<f32>(n | 0x4B000000u) - 8388608.0;
    let q = bitcast<u32>(f * bitcast<f32>(r) + 8388608.0) & 0x7FFFFFu;
    return q - select(0u, 1u, i32(n - q * d) < 0);
}
fn luma_urem(n: u32, d: u32, r: u32) -> u32 { return n - luma_udiv(n, d, r) * d; }
fn luma_mulhi(a: u32, b: u32) -> u32 {
    let a0 = a & 0xFFFFu; let a1 = a >> 16u;
    let b0 = b & 0xFFFFu; let b1 = b >> 16u;
    let m1 = a1 * b0; let m2 = a0 * b1;
    let carry = (((a0 * b0) >> 16u) + (m1 & 0xFFFFu) + (m2 & 0xFFFFu)) >> 16u;
    return a1 * b1 + (m1 >> 16u) + (m2 >> 16u) + carry;
}
fn luma_udiv_wide(n: u32, m: u32, s: u32) -> u32 { let t = luma_mulhi(n, m); return (t + ((n - t) >> 1u)) >> s; }
fn luma_urem_wide(n: u32, d: u32, m: u32, s: u32) -> u32 { return n - luma_udiv_wide(n, m, s) * d; }
";

/// `n / d` for the divisor whose words start at uniform slot `slot`; `wide` for any `n`, else
/// `n < 2^23`.
pub(super) fn udiv(n: &str, slot: u32, wide: bool) -> String {
    if wide {
        format!("luma_udiv_wide({n}, p.s{}, p.s{})", slot + 1, slot + 2)
    } else {
        format!("luma_udiv({n}, p.s{slot}, p.s{})", slot + 3)
    }
}

/// `n % d`, see [`udiv`].
pub(super) fn urem(n: &str, slot: u32, wide: bool) -> String {
    if wide {
        format!("luma_urem_wide({n}, p.s{slot}, p.s{}, p.s{})", slot + 1, slot + 2)
    } else {
        format!("luma_urem({n}, p.s{slot}, p.s{})", slot + 3)
    }
}

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
        (ElemTy::U32, ElemTy::F32) | (ElemTy::I32, ElemTy::F32) | (ElemTy::Bool, ElemTy::F32) => format!("f32({value})"),
        (ElemTy::F32, ElemTy::U32) => format!("select(0u, u32(clamp({value}, 0.0, 4294967040.0)), {value} == {value})"),
        (ElemTy::F32, ElemTy::I32) => format!("select(0i, i32(clamp({value}, -2147483648.0, 2147483520.0)), {value} == {value})"),
        (ElemTy::U32, ElemTy::I32) => format!("bitcast<i32>({value})"),
        (ElemTy::I32, ElemTy::U32) => format!("bitcast<u32>({value})"),
        (ElemTy::Bool, ElemTy::U32) => value.to_owned(),
        (ElemTy::Bool, ElemTy::I32) => format!("bitcast<i32>({value})"),
        (ElemTy::Bool, ElemTy::F64) => format!("f64({value})"),
        (ElemTy::Bool, ElemTy::Df32) => format!("vec2<f32>(f32({value}), 0.0)"),
        _ if from == to => value.to_owned(),
        _ => unreachable!("cast {from:?} -> {to:?}"),
    }
}

/// The double-float library a kernel of `key` defines (a reduction of `Df32` brings its own,
/// see `reduce_main`): behind the guard uniform when the program has one, else with a constant
/// guard, for the functions that never read it (ordering, minimum, maximum, absolute value).
pub(super) fn library_for(key: &FusedKey, tuning: &Tuning) -> String {
    let library = |guard: &str| df_library(guard, tuning.df_fma, tuning.df_sloppy_add);
    let instrs = || key.instrs.iter().chain(key.subs.iter().flatten());
    let math = if instrs().any(calls_df_math) { DF_MATH_LIBRARY } else { "" };
    match key.guard {
        Some(guard) => library(&format!("p.s{guard}")) + math,
        None if instrs().any(calls_df_library) => library("0u") + math,
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
            ElemTy::Bool => unreachable!("bool has no generators"),
            ElemTy::I32 => format!("bitcast<i32>({index})"),
            ElemTy::F32 | ElemTy::F64 | ElemTy::Df32 => format!("f32({index})"),
        },
        Gen::RangePow2 => match ty {
            ElemTy::U32 => format!("(({index} >> {a}) & {b})"),
            ElemTy::Bool => unreachable!("bool has no generators"),
            ElemTy::I32 => format!("bitcast<i32>(({index} >> {a}) & {b})"),
            ElemTy::F32 | ElemTy::F64 | ElemTy::Df32 => format!("f32(({index} >> {a}) & {b})"),
        },
        Gen::RangeConst { .. } | Gen::ParamRange { .. } => {
            let coord = match kind {
                Gen::RangeConst { stride: 1, size, .. } => format!("({index} % {size}u)"),
                Gen::RangeConst { stride, size, .. } => format!("(({index} / {stride}u) % {size}u)"),
                Gen::ParamRange { kind, wide, stride, size } => match kind {
                    params::RangeKind::Zero => "0u".to_owned(),
                    params::RangeKind::Mod => urem(index, size, wide),
                    params::RangeKind::DivMod => urem(&udiv(index, stride, wide), size, wide),
                },
                _ => unreachable!(),
            };
            match ty {
                ElemTy::U32 => coord,
                ElemTy::Bool => unreachable!("bool has no generators"),
                ElemTy::I32 => format!("bitcast<i32>({coord})"),
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
        ShiftKind::ParamDims { axes } => {
            let mut terms = Vec::new();
            for axis in axes {
                if !axis.used {
                    continue;
                }
                let [stride_slot, rem_slot, shift_slot] = axis.slots;
                // The stride the coordinate is scaled back by is a shift for a power of two.
                let (quotient, scale) = match axis.stride {
                    params::DivRemSpec::None => (i.to_owned(), None),
                    params::DivRemSpec::Pow2 => (format!("({i} >> p.s{stride_slot})"), Some(format!("<< p.s{stride_slot}"))),
                    params::DivRemSpec::Runtime { wide } => (udiv(i, stride_slot, wide), Some(format!("* p.s{stride_slot}"))),
                };
                let (coord, dim) = match (axis.rem, axis.shifted) {
                    (params::DivRemSpec::None, false) => (quotient, String::new()),
                    (params::DivRemSpec::Pow2, _) => (format!("({quotient} & p.s{rem_slot})"), format!("(p.s{rem_slot} + 1u)")),
                    (params::DivRemSpec::Runtime { wide }, _) => (urem(&quotient, rem_slot, wide), format!("p.s{rem_slot}")),
                    (params::DivRemSpec::None, true) => (quotient, format!("p.s{rem_slot}")),
                };
                let wrapped = if axis.shifted {
                    let shift = format!("p.s{shift_slot}");
                    format!("select({coord} - {shift}, {coord} + ({dim} - {shift}), {coord} < {shift})")
                } else {
                    coord
                };
                terms.push(match scale {
                    Some(scale) => format!("({wrapped} {scale})"),
                    None => wrapped,
                });
            }
            if terms.is_empty() {
                "0u".to_owned()
            } else {
                terms.join(" + ")
            }
        }
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
    instrs.iter().any(|instr| reads_directly(instr, k))
}

/// Whether `instr` loads input `k` at the element being computed.
fn reads_directly(instr: &Instr, k: u32) -> bool {
    let skip = usize::from(matches!(instr.op.kind, OpKind::Shift(_)));
    instr.args.iter().skip(skip).any(|arg| *arg == Ref::Input(k))
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
pub(super) fn stack_stages(key: &FusedKey, wg: u32, ty: ElemTy) -> bool {
    let Tail::Stack { total, first: 0, block: 1 } = key.tail else { return false };
    // The bits of a `bool` result are written one at a time, not as the words they are in.
    let bytes = match ty {
        ElemTy::Bool => return false,
        ElemTy::F32 | ElemTy::U32 | ElemTy::I32 => 4,
        ElemTy::F64 | ElemTy::Df32 => 8,
    };
    key.outputs.len() == total as usize && total > 1 && u64::from(wg) * u64::from(total) * bytes <= STACK_STAGE_BYTES
}

/// What a kernel of a key is built as: the pure tails (a store, a region write, a scatter, an
/// atomic scatter, a stack) as Kernel IR, built directly (see [`super::kirgen`]); the composite
/// ones, which reduce, multiply, sort or solve, as the WGSL source of the kernel.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Built {
    Module(KernelModule),
    Wgsl(String),
}

/// The kernel for `key`, with the workgroup shapes `tuning` gives.
/// `segment_len` (a power of two) is the elements per segment of the arrays the segmented
/// variant (`key.spans` not empty) runs on. `lowered` is the lowering of `key` (see
/// `jit::inline`) if it [`inlines`](FusedKey::inlines): the body then computes every
/// sub-program at the indices it is read at, and no `sub{k}` functions exist; for a
/// [`Lowered::Staged`] key (`key.staging` set) the staged sub-programs are computed into
/// workgroup memory instead.
pub(crate) fn build(key: &FusedKey, tuning: &Tuning, segment_len: u32, lowered: Option<&Lowered>) -> Result<Built> {
    match key.tail {
        Tail::Store
        | Tail::StoreMapped(_)
        | Tail::ParamStoreMapped { .. }
        | Tail::Scatter { .. }
        | Tail::ParamScatter { .. }
        | Tail::ScatterAdd { .. }
        | Tail::Stack { .. }
        | Tail::ParamStack { .. } => Ok(Built::Module(kirgen::store_module(key, tuning, segment_len, lowered)?)),
        Tail::Reduce(_) | Tail::AxisReduce(_) | Tail::Matmul { .. } | Tail::Solve(_) | Tail::Radix(_) => {
            Ok(Built::Wgsl(source(key, tuning, segment_len, lowered)))
        }
    }
}

/// The WGSL source of the kernel of a composite tail (see [`build`]).
fn source(key: &FusedKey, tuning: &Tuning, segment_len: u32, lowered: Option<&Lowered>) -> String {
    let mut src = kernel_source(key, tuning, segment_len, lowered);
    if src.contains("luma_udiv") || src.contains("luma_urem") {
        src.push_str(DIV_HELPERS);
    }
    src
}

fn kernel_source(key: &FusedKey, tuning: &Tuning, segment_len: u32, lowered: Option<&Lowered>) -> String {
    let inlined = match lowered {
        Some(Lowered::Plain(inlined)) => Some(inlined),
        Some(Lowered::Staged(programs)) => Some(&programs.main),
        None => None,
    };
    let wg = tuning.elementwise_workgroup;
    let reduce = matches!(key.tail, Tail::Reduce(_));
    let matmul = matches!(key.tail, Tail::Matmul { .. });
    let axis = matches!(key.tail, Tail::AxisReduce(_));
    let radix = match key.tail {
        Tail::Radix(pass) => Some(pass),
        _ => None,
    };
    let solve = match key.tail {
        Tail::Solve(spec) => Some(spec),
        _ => None,
    };
    let segmented = !key.spans.is_empty();
    let slots = |k: usize| key.spans.get(k).copied().unwrap_or(0);
    // A product reads its operands at their own element indices: nothing is relative to a segment.
    let based = generates(key) && !segmented && !matmul && !axis && radix.is_none() && solve.is_none();
    let uniform = reduce || matmul || axis || radix.is_some() || solve.is_some() || (!key.scalar_tys.is_empty() && reads_scalars(key)) || segmented || based || key.runs;
    let mut src = String::new();

    if uniform {
        src.push_str("struct Params { ");
        if reduce {
            src.push_str("n: u32, ");
        }
        if matmul {
            src.push_str("m: u32, n: u32, k: u32, kc: u32, ");
        }
        if axis {
            src.push_str(crate::scan::AXIS_PARAMS);
        }
        if radix.is_some() {
            src.push_str(crate::sort::PARAMS);
        }
        if solve.is_some() {
            src.push_str("n: u32, stride: u32, lines: u32, ");
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
        if key.runs {
            src.push_str("run: u32, words: u32, ");
        }
        for (k, ty) in key.scalar_tys.iter().enumerate() {
            let _ = write!(src, "s{k}: {}, ", ty.wgsl());
        }
        if let Tail::Reduce(ops) = key.tail {
            src.push_str(&ops.epilogue().fields());
        }
        src.push_str("}\n");
    }
    let out_elem = |at: usize| key.instrs.get(at).map_or(ElemTy::F32, |instr| instr.op.out_ty());
    let out_ty = |at: usize| out_elem(at).wgsl();
    let outputs: Vec<usize> = match key.tail {
        Tail::Store | Tail::StoreMapped(_) | Tail::ParamStoreMapped { .. } => {
            key.outputs.iter().map(|&at| at as usize).collect()
        }
        Tail::Scatter { .. }
        | Tail::ParamScatter { .. }
        | Tail::ScatterAdd { .. }
        | Tail::Stack { .. }
        | Tail::ParamStack { .. } => key.outputs.iter().take(1).map(|&at| at as usize).collect(),
        Tail::Reduce(_) => vec![key.instrs.len().saturating_sub(1)],
        // The roots are the operands, read by `luma_lhs` / `luma_rhs`; the output is `c`.
        // The roots of a sort are its keys, read by `luma_key`; its buffers are declared below.
        // The roots of a line solve are its parts, read by `luma_cell`; its buffers are declared below.
        Tail::Matmul { .. } | Tail::Radix(_) | Tail::Solve(_) => Vec::new(),
        // The roots are the cells of the pass, read by `luma_eval`; the outputs are its own.
        Tail::AxisReduce(_) => Vec::new(),
    };
    let axis_outputs = match key.tail {
        Tail::AxisReduce(pass) => {
            src.push_str(&pass.output_bindings(out_ty(key.instrs.len().saturating_sub(1))));
            pass.outputs()
        }
        _ => 0,
    };
    if matmul {
        src.push_str("@group(1) @binding(0) var<storage, read_write> c: array<f32>;\n");
    }
    let mut binding = outputs.len() + usize::from(matmul) + axis_outputs;
    if let Some(pass) = radix {
        // The roots are the keys, read by `luma_key`; the kernel's own buffers go around the inputs.
        pass.declare_writes(&mut src, &mut binding);
    }
    if let Some(spec) = solve {
        spec.declare_writes(&mut src, &mut binding);
    }
    for (j, &at) in outputs.iter().enumerate() {
        // A reduction's single output is `out`, as `reduce_main` expects.
        let name = if reduce { "out".to_owned() } else { format!("out{j}") };
        // An atomic scatter binds its target as atomics (floats as their bits). A reduction
        // writes the values it folds, whatever their buffer layout.
        let elem = match (key.tail, out_ty(at)) {
            (Tail::ScatterAdd { .. }, "i32") => "atomic<i32>",
            (Tail::ScatterAdd { .. }, _) => "atomic<u32>",
            (Tail::Reduce(_), ty) => ty,
            _ => out_storage_ty(out_elem(at)),
        };
        let _ = writeln!(src, "@group(1) @binding({j}) var<storage, read_write> {name}: array<{elem}>;");
    }
    for (k, ty) in key.input_tys.iter().enumerate() {
        if key.in_place.iter().any(|&(paired, _)| paired == k as u32) {
            continue;
        }
        if !segmented || direct(&key.instrs, k as u32) {
            let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, read> in{k}: array<{}>;", storage_ty(*ty));
            binding += 1;
        }
        for s in 0..slots(k) {
            let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, read> in{k}_{s}: array<{}>;", storage_ty(*ty));
            binding += 1;
        }
    }
    if let Tail::Matmul { lhs, rhs, .. } = key.tail {
        // The operands that are buffers, after the inputs of the others.
        for (operand, name) in [(lhs, "a"), (rhs, "b")] {
            if !operand.fused() {
                let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, read> {name}: array<f32>;");
                binding += 1;
            }
        }
    }
    if let Some(pass) = radix {
        pass.declare_reads(&mut src, &mut binding);
    }
    if let Some(spec) = solve {
        spec.declare_reads(&mut src, &mut binding);
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
            let per = segment_len / (ty.stride() as u32 / 4).max(1);
            let (log, mask) = (per.trailing_zeros(), per.wrapping_sub(1));
            let _ = writeln!(src, "fn ld{k}(g: u32) -> {} {{\n    let c = g >> {log}u; let o = g & {mask}u;", ty.wgsl());
            for s in 0..m - 1 {
                let _ = writeln!(src, "    if (c == p.c{k}_{s}) {{ return {}; }}", load(*ty, &format!("in{k}_{s}"), "o"));
            }
            let _ = writeln!(src, "    return {};\n}}", load(*ty, &format!("in{k}_{}", m - 1), "o"));
        }
    }
    let views = match lowered {
        Some(Lowered::Plain(inlined)) => views_of_inlined(key, &[inlined]),
        Some(Lowered::Staged(programs)) => views_of_inlined(key, &programs.bodies().collect::<Vec<_>>()),
        None => views_of(key),
    };
    // A reduction of `Df32` brings the library itself (see `reduce_main`), but not its math
    // functions: those come with the program that calls them.
    let reduces_df = reduce && out_ty(outputs[0]) == "vec2<f32>";
    if !reduces_df {
        src.push_str(&library_for(key, tuning));
    } else if key.instrs.iter().chain(key.subs.iter().flatten()).any(calls_df_math) {
        src.push_str(DF_MATH_LIBRARY);
    }
    src.push_str(&generator_helpers(std::iter::once(key.instrs.as_slice()).chain(key.subs.iter().map(Vec::as_slice))));
    src.push_str(&view_helpers(&views));
    if key.params.is_some() {
        let bodies: Vec<&Inlined> = match lowered {
            Some(Lowered::Plain(inlined)) => vec![inlined],
            Some(Lowered::Staged(programs)) => programs.bodies().collect(),
            None => Vec::new(),
        };
        src.push_str(&params::view_helpers(&params::views_used(key, &bodies)));
    }
    if key.tile.is_none() && lowered.is_none() {
        src.push_str(&subs(key, &views, &key.in_place));
    }

    let index = match (segmented, based) {
        (true, _) | (false, true) => "gi",
        (false, false) => "i",
    };
    match key.tail {
        Tail::Store
        | Tail::StoreMapped(_)
        | Tail::ParamStoreMapped { .. }
        | Tail::Scatter { .. }
        | Tail::ParamScatter { .. }
        | Tail::ScatterAdd { .. }
        | Tail::Stack { .. }
        | Tail::ParamStack { .. } => unreachable!("the store tails are built as Kernel IR (see `kirgen::store_module`)"),
        Tail::Reduce(ops) => {
            let (ty, last) = (out_ty(outputs[0]), outputs[0]);
            let guard = format!("p.s{}", key.guard.unwrap_or(0));
            let elem = if reduces_df { ReduceElem::Df32 { guard: &guard, fma: tuning.df_fma } } else { ReduceElem::Plain(ty) };
            let names = Names { views: &views, in_place: &[] };
            src.push_str(&load_function("luma_eval", key, last, inlined, &names, index, segmented));
            src.push_str(&reduce_main(elem, ops, false, "luma_eval", tuning));
        }
        Tail::AxisReduce(pass) => {
            let names = Names { views: &views, in_place: &[] };
            let last = key.instrs.len().saturating_sub(1);
            src.push_str(&load_function("luma_eval", key, last, inlined, &names, index, segmented));
            src.push_str(&crate::scan::reduce_body(pass, out_ty(last), "luma_eval", wg));
        }
        Tail::Matmul { tile, aligned, split, lhs, rhs } => {
            let names = Names { views: &views, in_place: &[] };
            let mut roots = key.outputs.iter().map(|&at| at as usize);
            for (operand, function) in [(lhs, "luma_lhs"), (rhs, "luma_rhs")] {
                if operand.fused() && let Some(root) = roots.next() {
                    src.push_str(&load_function(function, key, root, inlined, &names, index, segmented));
                }
            }
            src.push_str(&crate::matmul::body(tile, aligned, split, lhs, rhs));
        }
        Tail::Solve(spec) => {
            let names = Names { views: &views, in_place: &[] };
            let per = crate::solve::Io::parts_per_system(usize::from(spec.block));
            let mut roots = key.outputs.iter().map(|&at| at as usize);
            for s in 0..usize::from(spec.systems) {
                let parts: Vec<Part> = (0..per)
                    .map(|k| if spec.is_fused(s * per + k) { Part::Root(roots.next().unwrap_or(0)) } else { Part::Buffer(s * per + k) })
                    .collect();
                src.push_str(&load_cell(&format!("luma_cell{s}"), key, &parts, inlined, &names, index, segmented));
            }
            src.push_str(&spec.cell_dispatch());
            src.push_str(&spec.body());
        }
        Tail::Radix(pass) => {
            let names = Names { views: &views, in_place: &[] };
            let root = key.outputs.first().map_or(0, |&at| at as usize);
            src.push_str(&load_function("luma_key", key, root, inlined, &names, index, segmented));
            src.push_str(&crate::sort::fused_body(pass, out_ty(root)));
        }
    }
    src
}

/// `fn {name}(i: u32) -> T { ...; return t{out}; }`: the value of instruction `out` of the main
/// program of `key` at element `i`, computed by the instructions it needs
/// (those of the lowered program `inlined` for a lowered key). The program's loads, views and library are the kernel's
/// ([`source`] declares them before this); `index` and `segmented` are those of the kernel's
/// other uses of [`body`] (the index `gi` is the element's place in the whole array: the
/// function first computes it). A kernel that reads
/// an expression elsewhere than at the element it computes (a reduction's operand, a GEMM's
/// loads, a sort's keys) calls it as `{name}(index)`.
pub(super) fn load_function(
    name: &str,
    key: &FusedKey,
    out: usize,
    inlined: Option<&Inlined>,
    names: &Names,
    index: &str,
    segmented: bool,
) -> String {
    let global = if index == "gi" { "    let gi = p.base + i;\n" } else { "" };
    let ty = key.instrs.get(out).map_or(ElemTy::F32, |instr| instr.op.out_ty()).wgsl();
    let (computed, values) = computed_values(key, &[out], inlined, names, index, segmented);
    format!("\nfn {name}(i: u32) -> {ty} {{\n{global}{computed}    return t{};\n}}\n", values[0])
}

/// One value a line solve reads at a cell: the root `at` of the program, or the buffer `pr{k}`.
pub(super) enum Part {
    Root(usize),
    Buffer(usize),
}

/// `fn {name}(i: u32) -> array<f32, N> { ...; return array(...); }`: the `f32` values of `parts`
/// at element `i`, in order, the roots computed together by the instructions they need (see
/// [`load_function`]).
pub(super) fn load_cell(
    name: &str,
    key: &FusedKey,
    parts: &[Part],
    inlined: Option<&Inlined>,
    names: &Names,
    index: &str,
    segmented: bool,
) -> String {
    let global = if index == "gi" { "    let gi = p.base + i;\n" } else { "" };
    let roots: Vec<usize> = parts.iter().filter_map(|part| if let Part::Root(at) = part { Some(*at) } else { None }).collect();
    let (computed, values) = computed_values(key, &roots, inlined, names, index, segmented);
    let mut values = values.into_iter();
    let returned: Vec<String> = parts
        .iter()
        .map(|part| match part {
            Part::Root(_) => format!("t{}", values.next().unwrap_or(0)),
            Part::Buffer(k) => format!("pr{k}[i]"),
        })
        .collect();
    format!("\nfn {name}(i: u32) -> array<f32, {n}> {{\n{global}{computed}    return array<f32, {n}>({});\n}}\n", returned.join(", "), n = parts.len())
}

/// The statements computing the instructions that the roots `outs` of the main program of `key`
/// need (those of the lowered program `inlined` for a lowered key), and the instruction
/// each root's value is in afterwards.
fn computed_values(
    key: &FusedKey,
    outs: &[usize],
    inlined: Option<&Inlined>,
    names: &Names,
    index: &str,
    segmented: bool,
) -> (String, Vec<usize>) {
    let values: Vec<usize> = outs.iter().map(|&out| inlined.and_then(|inlined| inlined.main.get(out)).map_or(out, |&lowered| lowered as usize)).collect();
    let computed = match inlined {
        Some(inlined) => lowered_body(key, inlined, names, index, None, Some(&needed_lowered(inlined, &values))),
        None => body_with(&key.instrs, &key.input_tys, names, index, segmented, Some(&needed(&key.instrs, outs)), &|_, _| None),
    };
    (computed, values)
}

/// Which instructions of the lowered program `inlined` the lowered instructions `roots` need,
/// themselves included: their operands, and those of the shifts that define the indices of the
/// contexts they are computed at and read at.
fn needed_lowered(inlined: &Inlined, roots: &[usize]) -> Vec<bool> {
    let tmp = |arg: &Ref| if let Ref::Tmp(t) = arg { Some(*t as usize) } else { None };
    let mut keep = vec![false; inlined.instrs.len()];
    let mut work = roots.to_vec();
    while let Some(n) = work.pop() {
        let Some(slot) = keep.get_mut(n) else { continue };
        if std::mem::replace(slot, true) {
            continue;
        }
        let flat = &inlined.instrs[n];
        work.extend(flat.args.iter().filter_map(tmp));
        for mut c in [flat.at, flat.child] {
            while c != 0 {
                let Some(ctx) = inlined.ctxs.get(c as usize) else { break };
                work.extend(ctx.args.iter().filter_map(tmp));
                c = ctx.parent;
            }
        }
    }
    keep
}

/// Which instructions of `instrs` the instructions `outs` need, themselves included.
fn needed(instrs: &[Instr], outs: &[usize]) -> Vec<bool> {
    let mut keep = vec![false; instrs.len()];
    for &out in outs {
        if let Some(root) = keep.get_mut(out) {
            *root = true;
        }
    }
    let last = outs.iter().copied().max().unwrap_or(0);
    for at in (0..=last.min(instrs.len().saturating_sub(1))).rev() {
        if keep[at] {
            for arg in instrs[at].args {
                if let Ref::Tmp(t) = arg
                    && let Some(needed) = keep.get_mut(t as usize)
                {
                    *needed = true;
                }
            }
        }
    }
    keep
}

/// The loads of the inputs at element `i` and the instructions, as `let` bindings. The inputs
/// `names.in_place` lists are read from the output buffers, before they are overwritten.
/// `index` and `segmented` are those of [`expression`].
fn body(instrs: &[Instr], input_tys: &[ElemTy], names: &Names, index: &str, segmented: bool) -> String {
    body_with(instrs, input_tys, names, index, segmented, None, &|_, _| None)
}

/// [`body`], where `staged(k, instr)` may replace the expression of instruction `k`, and only the
/// instructions `keep` marks (all, if `None`) are written, with the inputs they load.
fn body_with(
    instrs: &[Instr],
    input_tys: &[ElemTy],
    names: &Names,
    index: &str,
    segmented: bool,
    keep: Option<&[bool]>,
    staged: &dyn Fn(usize, &Instr) -> Option<String>,
) -> String {
    let kept = |k: usize| keep.is_none_or(|keep| keep.get(k).copied().unwrap_or(false));
    let mut src = String::new();
    // Inputs read only through shifts are not loaded at `i`.
    for (k, ty) in input_tys.iter().enumerate() {
        if instrs.iter().enumerate().any(|(at, instr)| kept(at) && reads_directly(instr, k as u32)) {
            let _ = writeln!(src, "    let a{k}: {} = {};", ty.wgsl(), input_load(names, *ty, k as u32, "i"));
        }
    }
    for (k, instr) in instrs.iter().enumerate().filter(|&(k, _)| kept(k)) {
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
fn lowered_body(key: &FusedKey, inlined: &Inlined, names: &Names, index: &str, staging: Option<&Staging>, keep: Option<&[bool]>) -> String {
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
            let ty = self.key.input_tys.get(k as usize).copied().unwrap_or(ElemTy::F32);
            if c == 0 {
                let _ = writeln!(self.src, "    let a{k}: {} = {};", ty.wgsl(), input_load(self.names, ty, k, "i"));
            } else {
                let _ = writeln!(self.src, "    let a{k}_{c}: {} = {};", ty.wgsl(), input_load(self.names, ty, k, &format!("i{c}")));
            }
        }

        /// Writes the instructions computed in `region`.
        fn region(&mut self, region: Region) {
            for n in self.members.remove(&region).unwrap_or_default() {
                self.instruction(n as usize);
            }
        }

        /// The offset of the cell instruction `n` (a read of a staged sub-program through `kind`,
        /// computed at context `at`) reads from the cell of its program: the starts of the
        /// translation views on the way and of the read itself, where the lowering holds them.
        fn staged_offset(&self, n: usize, kind: ShiftKind, at: u32) -> Option<[i32; 2]> {
            if let Some(read) = self.inlined.staged_reads.get(n) {
                return *read;
            }
            let ShiftKind::View { map, .. } = kind else {
                return None;
            };
            let base = self.inlined.translation(at);
            Some([map.axes[0].start + base[0], map.axes[1].start + base[1]])
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
                    if !staged_read {
                        self.define(flat.child);
                    }
                    let at = self.index_of(flat.child);
                    let read = match flat.args[0] {
                        Ref::Input(k) => input_load(self.names, op.ty, k, &at),
                        Ref::Tmp(v) => format!("t{v}"),
                        Ref::Sub(s) => match self
                            .staging
                            .and_then(|layout| layout.find(s))
                            .zip(self.staged_offset(n, kind, flat.at))
                        {
                            // A read from workgroup memory: the region's cell `(lx, ly)` plus the
                            // offsets of the contexts the read is in and of the read itself.
                            Some((entry, [ox, oy])) => {
                                let (dx, dy) = (ox - entry.lo[0], oy - entry.lo[1]);
                                format!(
                                    "w{s}[u32(lx + ({dx})) + u32(ly + ({dy})) * {}u]",
                                    entry.width()
                                )
                            }
                            None => format!("{}[{at}]", name(Ref::Sub(s), ElemTy::U32)),
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
    for (n, &region) in inlined.region.iter().enumerate().filter(|&(n, _)| keep.is_none_or(|keep| keep.get(n).copied().unwrap_or(false))) {
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
#[derive(Clone, Debug)]
pub(crate) struct Tile {
    /// Width and height of the array (`h == 1` for 1D). Part of the key unless `slots` is set:
    /// then the source reads them from the uniform and they only tell the launch how many
    /// workgroups to run.
    pub(crate) dims: (u32, u32),
    /// Where the uniform has the width, the height and the tiles per row, once the key has been
    /// rewritten by [`params::apply_tile`] (see [`TileSlots`]).
    pub(crate) slots: Option<TileSlots>,
    /// Output cells per workgroup.
    pub(crate) tile: (u32, u32),
    /// Per sub-program, the halo (cells per side and axis) its region extends beyond the tile.
    pub(crate) need: Vec<(u32, u32)>,
    /// Per program (0 = main, `1 + k` = `subs[k]`) and instruction, the cell offset (x, y) a
    /// shift of a sub-program reads; `None` for every other instruction.
    pub(crate) offsets: Vec<Vec<Option<(i32, i32)>>>,
}

/// What the source of a tiled kernel whose array extents are in the uniform depends on.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct TileSlots {
    /// The slot of the width; the height and the tiles per row follow.
    pub(crate) base: u32,
    /// The array is 1-dimensional.
    pub(crate) one_d: bool,
    /// A tile and its halo fit in the array along every axis, so that a halo cell is at most
    /// one period away (see `Lower::declare_wrap`).
    pub(crate) small: bool,
}

impl PartialEq for Tile {
    fn eq(&self, other: &Tile) -> bool {
        self.slots == other.slots
            && (self.slots.is_some() || self.dims == other.dims)
            && self.tile == other.tile
            && self.need == other.need
            && self.offsets == other.offsets
    }
}

impl Eq for Tile {}

impl std::hash::Hash for Tile {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.slots.hash(state);
        if self.slots.is_none() {
            self.dims.hash(state);
        }
        self.tile.hash(state);
        self.need.hash(state);
        self.offsets.hash(state);
    }
}

impl Tile {
    /// Workgroups of a launch.
    pub(crate) fn tiles(&self) -> u32 {
        self.dims.0.div_ceil(self.tile.0) * self.dims.1.div_ceil(self.tile.1)
    }

    /// The halo of the widest region along each axis.
    pub(crate) fn halo(&self) -> (u32, u32) {
        self.need.iter().fold((0, 0), |(x, y), &(nx, ny)| (x.max(nx), y.max(ny)))
    }

    /// Whether a tile and its halo fit in the array along every axis.
    pub(crate) fn fits(&self) -> bool {
        let ((w, h), (tx, ty), (halo_x, halo_y)) = (self.dims, self.tile, self.halo());
        w >= tx + 2 * halo_x && (h == 1 || h >= ty + 2 * halo_y)
    }
}

/// The staging of `key` on arrays of `len` elements, if its sub-programs are read through
/// small shifts only and their regions fit in workgroup memory. `None` keeps the sub-programs
/// as functions called per shift.
pub(super) fn plan_tile(key: &FusedKey, scalars: &[u32], len: usize, tuning: &Tuning) -> Option<Tile> {
    // Staged values are 4 bytes wide; programs of the 8-byte types call their sub-programs.
    if key.tail != Tail::Store || key.subs.is_empty() || !key.spans.is_empty() || !key.narrow() || key.writes_bits() {
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
                        ShiftKind::View { .. } | ShiftKind::Take { .. } | ShiftKind::ParamView { .. } | ShiftKind::ParamTake { .. } | ShiftKind::ParamDims { .. } => return None,
                        ShiftKind::Dims { shifts, .. } => (signed(shifts[0], w), signed(shifts[1], h)),
                        ShiftKind::Linear => {
                            let Ref::Scalar(s) = instr.args[1] else { return None };
                            let amount = *scalars.get(s as usize)?;
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
    Some(Tile { dims: (w, h), slots: None, tile: (tx, ty), need, offsets })
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
