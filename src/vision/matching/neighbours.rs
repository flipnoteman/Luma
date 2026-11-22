//! Nearest neighbours: for every query feature the `k` closest train features by the sum of
//! absolute differences, of squared differences or the Hamming distance.
//!
//! Two kernels: the distances of a block of 64 queries to every train feature are computed like
//! the product of two matrices (`dist.rs` style: both operands staged 16 elements at a time in
//! workgroup memory, every invocation accumulating 4 x 4 distances in registers) into a scratch
//! matrix of `[train, query]`; a workgroup per query then keeps the `k` smallest in a sorted
//! list in workgroup memory, merging in blocks of 256 distances (bitonic sort of the block that
//! beat the list's largest entry, then a bitonic merge with the list). Queries are processed in
//! chunks so that the scratch matrix stays within a fixed budget.

use std::fmt::Write as _;

use crate::array::Array;
use crate::builtin::Builtin;
use crate::context::{RawArg, Scratch};
use crate::error::{LumaError, Result};
use crate::shape::Shape;

use super::super::{to_u32, Kernel as Any};
use super::{DistType, Neighbour};

/// Features per block along the query and train axes of the distance kernel, and the elements
/// of each feature staged per step.
const BQ: u32 = 64;
const BT: u32 = 64;
const KL: u32 = 16;

/// The element type of the features (and with it of the distances).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Elem {
    F32,
    U32,
    I32,
}

impl Elem {
    fn wgsl(self) -> &'static str {
        match self {
            Elem::F32 => "f32",
            Elem::U32 => "u32",
            Elem::I32 => "i32",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Kernel {
    /// The distances of a block of queries to all train features.
    Distances { elem: Elem, metric: DistType, along: bool },
    /// The `cap` smallest distances of every query (`cap` a power of two, at least `k`).
    Select { elem: Elem, cap: u32 },
}

pub(crate) fn source(kernel: Kernel) -> String {
    match kernel {
        Kernel::Distances { elem, metric, along } => distances_source(elem, metric, along),
        Kernel::Select { elem, cap } => select_source(elem, cap),
    }
}

/// Arguments: query, train, the distances (`[train, query]` of the chunk, as words), uniform
/// `{nq, nt, len, q0, qend}`. `along`: the elements of a feature are contiguous (`dist_dim` 0).
fn distances_source(elem: Elem, metric: DistType, along: bool) -> String {
    let ty = elem.wgsl();
    let (zero, acc_zero) = match elem {
        Elem::F32 => ("0.0", "0.0"),
        Elem::U32 => ("0u", "0u"),
        Elem::I32 => ("0", "0u"),
    };
    // The accumulation of the distance `acc` of the pair `a`, `b` (`d` is a name for a temporary).
    let step = |a: &str, b: &str, acc: &str, d: &str| -> String {
        match (elem, metric) {
            (Elem::F32, DistType::Sad) => format!("{acc} += abs({a} - {b});"),
            (Elem::F32, DistType::Ssd) => format!("let {d} = {a} - {b}; {acc} = fma({d}, {d}, {acc});"),
            (Elem::U32, DistType::Sad) => format!("{acc} += max({a}, {b}) - min({a}, {b});"),
            (Elem::U32, DistType::Ssd) => format!("let {d} = max({a}, {b}) - min({a}, {b}); {acc} += {d} * {d};"),
            (Elem::U32, DistType::Shd) => format!("{acc} += countOneBits({a} ^ {b});"),
            (Elem::I32, DistType::Sad) => format!("{acc} += bitcast<u32>(max({a}, {b})) - bitcast<u32>(min({a}, {b}));"),
            (Elem::I32, DistType::Ssd) => format!("let {d} = bitcast<u32>(max({a}, {b})) - bitcast<u32>(min({a}, {b})); {acc} += {d} * {d};"),
            _ => unreachable!("rejected by the caller"),
        }
    };
    let mut s = String::new();
    s.push_str("struct P { nq: u32, nt: u32, len: u32, q0: u32, qend: u32, a: u32, b: u32, c: u32 }\n");
    let _ = writeln!(s, "@group(1) @binding(0) var<storage, read> qv: array<{ty}>;");
    let _ = writeln!(s, "@group(1) @binding(1) var<storage, read> tv: array<{ty}>;");
    s.push_str("@group(1) @binding(2) var<storage, read_write> dm: array<u32>;\n");
    s.push_str("@group(1) @binding(3) var<uniform> p: P;\n");
    let _ = writeln!(s, "var<workgroup> qs: array<{ty}, {}>;", KL * BQ);
    let _ = writeln!(s, "var<workgroup> ts: array<{ty}, {}>;", KL * BT);
    let (qidx, tidx, f, e) = if along {
        ("gq * p.len + ge", "gt * p.len + ge", format!("i / {KL}u"), format!("i % {KL}u"))
    } else {
        ("gq + p.nq * ge", "gt + p.nt * ge", format!("i % {BQ}u"), format!("i / {BQ}u"))
    };
    let _ = write!(
        s,
        "@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let tx = t & 15u;
    let ty = t >> 4u;
    let t0 = wid.x * {BT}u;
    let qb = p.q0 + wid.y * {BQ}u;
"
    );
    for j in 0..4 {
        for i in 0..4 {
            let _ = writeln!(s, "    var acc{j}_{i} = {acc_zero};");
        }
    }
    let _ = write!(
        s,
        "    for (var e0 = 0u; e0 < p.len; e0 += {KL}u) {{
        workgroupBarrier();
        for (var i = t; i < {words}u; i += 256u) {{
            let f = {f};
            let e = {e};
            let ge = e0 + e;
            let gq = qb + f;
            let gt = t0 + f;
            var a = {zero};
            if (gq < p.qend && ge < p.len) {{ a = qv[{qidx}]; }}
            qs[e * {BQ}u + f] = a;
            var b = {zero};
            if (gt < p.nt && ge < p.len) {{ b = tv[{tidx}]; }}
            ts[e * {BT}u + f] = b;
        }}
        workgroupBarrier();
        for (var e = 0u; e < {KL}u; e++) {{
",
        words = KL * BQ
    );
    for j in 0..4 {
        let _ = writeln!(s, "            let a{j} = qs[e * {BQ}u + ty * 4u + {j}u];");
    }
    for i in 0..4 {
        let _ = writeln!(s, "            let b{i} = ts[e * {BT}u + tx + {}u];", 16 * i);
    }
    for j in 0..4 {
        for i in 0..4 {
            let _ = writeln!(s, "            {}", step(&format!("a{j}"), &format!("b{i}"), &format!("acc{j}_{i}"), &format!("d{j}_{i}")));
        }
    }
    s.push_str("        }\n    }\n");
    let word = |v: String| if elem == Elem::F32 { format!("bitcast<u32>({v})") } else { v };
    for j in 0..4 {
        let _ = writeln!(s, "    {{\n        let gq = qb + ty * 4u + {j}u;");
        for i in 0..4 {
            let _ = writeln!(s, "        {{ let gt = t0 + tx + {}u; if (gq < p.qend && gt < p.nt) {{ dm[gt + p.nt * (gq - p.q0)] = {}; }} }}", 16 * i, word(format!("acc{j}_{i}")));
        }
        s.push_str("    }\n");
    }
    s.push_str("}\n");
    s
}

/// Arguments: distances (`[train, query]` words), indices out, distances out (`[k, query]`),
/// uniform `{nt, k, q0}`.
fn select_source(elem: Elem, cap: u32) -> String {
    let key = match elem {
        Elem::F32 => "select(dm[base + j], 0xffffffffu, bitcast<f32>(dm[base + j]) != bitcast<f32>(dm[base + j]))",
        Elem::U32 => "dm[base + j]",
        Elem::I32 => "dm[base + j] ^ 0x80000000u",
    };
    let out = match elem {
        Elem::I32 => "ak[t] ^ 0x80000000u",
        _ => "ak[t]",
    };
    let mut s = String::new();
    s.push_str("struct P { nt: u32, k: u32, q0: u32, a: u32 }\n");
    s.push_str("@group(1) @binding(0) var<storage, read> dm: array<u32>;\n");
    s.push_str("@group(1) @binding(1) var<storage, read_write> oi: array<u32>;\n");
    s.push_str("@group(1) @binding(2) var<storage, read_write> od: array<u32>;\n");
    s.push_str("@group(1) @binding(3) var<uniform> p: P;\n");
    let _ = writeln!(s, "var<workgroup> ak: array<u32, {cap}>;\nvar<workgroup> ai: array<u32, {cap}>;");
    s.push_str("var<workgroup> bk: array<u32, 256>;\nvar<workgroup> bi: array<u32, 256>;\nvar<workgroup> npass: atomic<u32>;\nvar<workgroup> go: u32;\n");
    s.push_str("fn lt(k1: u32, i1: u32, k2: u32, i2: u32) -> bool { return k1 < k2 || (k1 == k2 && i1 < i2); }\n");
    let _ = write!(
        s,
        "@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let base = wid.x * p.nt;
    if (t < {cap}u) {{ ak[t] = 0xffffffffu; ai[t] = 0xffffffffu; }}
    for (var j0 = 0u; j0 < p.nt; j0 += 256u) {{
        let j = j0 + t;
        if (t == 0u) {{ atomicStore(&npass, 0u); }}
        workgroupBarrier();
        var key = 0xffffffffu;
        var id = 0xffffffffu;
        if (j < p.nt) {{
            let kk = {key};
            if (lt(kk, j, ak[{last}u], ai[{last}u])) {{ key = kk; id = j; atomicAdd(&npass, 1u); }}
        }}
        bk[t] = key;
        bi[t] = id;
        workgroupBarrier();
        if (t == 0u) {{ go = atomicLoad(&npass); }}
        workgroupBarrier();
        if (workgroupUniformLoad(&go) == 0u) {{ continue; }}
        for (var kk = 2u; kk <= 256u; kk = kk << 1u) {{
            for (var jj = kk >> 1u; jj > 0u; jj = jj >> 1u) {{
                let ix = t ^ jj;
                if (ix > t) {{
                    let k1 = bk[t]; let i1 = bi[t]; let k2 = bk[ix]; let i2 = bi[ix];
                    let asc = (t & kk) == 0u;
                    if ((asc && lt(k2, i2, k1, i1)) || (!asc && lt(k1, i1, k2, i2))) {{
                        bk[t] = k2; bi[t] = i2; bk[ix] = k1; bi[ix] = i1;
                    }}
                }}
                workgroupBarrier();
            }}
        }}
        var ck = 0u;
        var ci = 0u;
        if (t < {cap}u) {{
            let r = {last}u - t;
            if (lt(bk[r], bi[r], ak[t], ai[t])) {{ ck = bk[r]; ci = bi[r]; }} else {{ ck = ak[t]; ci = ai[t]; }}
        }}
        workgroupBarrier();
        if (t < {cap}u) {{ ak[t] = ck; ai[t] = ci; }}
        workgroupBarrier();
        for (var jj = {half}u; jj > 0u; jj = jj >> 1u) {{
            let ix = t ^ jj;
            if (t < {cap}u && ix > t) {{
                let k1 = ak[t]; let i1 = ai[t]; let k2 = ak[ix]; let i2 = ai[ix];
                if (lt(k2, i2, k1, i1)) {{ ak[t] = k2; ai[t] = i2; ak[ix] = k1; ai[ix] = i1; }}
            }}
            workgroupBarrier();
        }}
    }}
    if (t < p.k) {{
        let o = (p.q0 + wid.x) * p.k + t;
        oi[o] = ai[t];
        od[o] = {out};
    }}
}}
",
        last = cap - 1,
        half = cap / 2
    );
    s
}

/// The most words of the distance matrix a chunk of queries may take.
const MATRIX_WORDS: u64 = 64 << 20;

pub(crate) fn run<T: Neighbour>(query: &Array<T>, train: &Array<T>, dist_dim: usize, k: usize, metric: DistType) -> Result<(Array<u32>, Array<T::Dist>)> {
    let ctx = query.context();
    if !ctx.same_as(train.context()) {
        return Err(LumaError::ContextMismatch);
    }
    if dist_dim > 1 {
        return Err(LumaError::InvalidShape(format!("nearest_neighbour: dist_dim {dist_dim} is not 0 or 1")));
    }
    let (qd, td) = (query.shape().dims(), train.shape().dims());
    if qd[2] != 1 || qd[3] != 1 || td[2] != 1 || td[3] != 1 {
        return Err(LumaError::InvalidShape(format!("nearest_neighbour takes 2-D feature matrices, got {qd:?} and {td:?}")));
    }
    // Features along the other axis than the one the distance runs over.
    let (nq, len) = if dist_dim == 0 { (qd[1], qd[0]) } else { (qd[0], qd[1]) };
    let (nt, tlen) = if dist_dim == 0 { (td[1], td[0]) } else { (td[0], td[1]) };
    if len != tlen {
        return Err(LumaError::InvalidShape(format!("nearest_neighbour: query and train describe features with {len} and {tlen} elements along axis {dist_dim}")));
    }
    if k == 0 || k > nt || k > 256 {
        return Err(LumaError::InvalidShape(format!("nearest_neighbour: n_dist {k} is not in 1..=min(256, {nt}), the number of train features")));
    }
    if !T::supports(metric) {
        return Err(LumaError::Unsupported(format!("nearest_neighbour: {metric:?} is not defined for {} features", T::NAME)));
    }
    let to = |v: usize, what: &str| to_u32(v, what).map_err(crate::array::Failure::into_error);
    let (nq32, nt32, len32, k32) = (to(nq, "query features")?, to(nt, "train features")?, to(len, "elements")?, to(k, "n_dist")?);
    let idx = Array::<u32>::alloc(ctx, Shape::new([k, nq, 1, 1]))?;
    let dist = Array::<T::Dist>::alloc(ctx, Shape::new([k, nq, 1, 1]))?;
    let (q_pin, q_bytes) = query.ready().map_err(crate::array::Failure::into_error)?;
    let (t_pin, t_bytes) = train.ready().map_err(crate::array::Failure::into_error)?;
    let (idx_pin, idx_bytes) = idx.ready().map_err(crate::array::Failure::into_error)?;
    let (dist_pin, dist_bytes) = dist.ready().map_err(crate::array::Failure::into_error)?;
    // Queries per chunk: the distance matrix of a chunk is `nt` words per query.
    let budget = (MATRIX_WORDS.min(ctx.max_buffer_bytes() / 4)) / u64::from(nt32);
    let max_groups = u64::from(ctx.max_workgroups_per_dimension());
    let per_chunk = budget.min(max_groups * u64::from(BQ)).min(max_groups) / u64::from(BQ) * u64::from(BQ);
    let per_chunk = if (nq as u64) <= budget.min(max_groups) { nq as u64 } else { per_chunk };
    if per_chunk == 0 || u64::from(nt32).div_ceil(u64::from(BT)) > max_groups {
        return Err(LumaError::Unsupported(format!("nearest_neighbour: {nt} train features are too many for one launch")));
    }
    let elem = T::ELEM;
    let cap = k32.next_power_of_two();
    let mut scratch = Scratch::new(ctx);
    let matrix_bytes = per_chunk * u64::from(nt32) * 4;
    let matrix = scratch.take(matrix_bytes)?;
    let dist_kernel = ctx.builtin(Builtin::Vision(Any::Matching(super::Kernel::Neighbours(Kernel::Distances { elem, metric, along: dist_dim == 0 }))), T::NAME)?;
    let select_kernel = ctx.builtin(Builtin::Vision(Any::Matching(super::Kernel::Neighbours(Kernel::Select { elem, cap }))), T::NAME)?;
    let mut q0 = 0u64;
    while q0 < nq as u64 {
        let qc = per_chunk.min(nq as u64 - q0);
        let qend = (q0 + qc) as u32;
        let params = [nq32, nt32, len32, q0 as u32, qend, 0, 0, 0];
        let args = [
            RawArg::read(q_pin.buffer(), q_bytes),
            RawArg::read(t_pin.buffer(), t_bytes),
            RawArg::write(&matrix, matrix_bytes),
            RawArg::Uniform(bytemuck::bytes_of(&params)),
        ];
        ctx.launch(&dist_kernel, [nt32.div_ceil(BT), (qc as u32).div_ceil(BQ), 1], [nt32, qc as u32, 1], &args)?;
        let sel = [nt32, k32, q0 as u32, 0];
        let args = [
            RawArg::read(&matrix, matrix_bytes),
            RawArg::write(idx_pin.buffer(), idx_bytes),
            RawArg::write(dist_pin.buffer(), dist_bytes),
            RawArg::Uniform(bytemuck::bytes_of(&sel)),
        ];
        ctx.launch(&select_kernel, [qc as u32, 1, 1], [qc as u32, 1, 1], &args)?;
        q0 += qc;
    }
    Ok((idx, dist))
}
