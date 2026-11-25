//! Runs of equal keys: the scan by key (a segmented scan) and the reductions by key.
//!
//! A *run* is a maximal stretch of consecutive elements whose keys compare equal (`==`: so every
//! NaN key is a run of its own); keys are not sorted or grouped by any of this, equal keys that
//! are apart are different runs, as in ArrayFire's `scanByKey` and `sumByKey`.
//!
//! The scan by key works on lines of contiguous cells cut in chunks of [`CHUNK`]. A line of one
//! chunk is one kernel; longer lines take three:
//!
//! 1. *Fold*: a workgroup reads the keys of its chunk, writes one bit per cell (does a run start
//!    here?) and folds only the cells from the last run start in the chunk (all of them if none
//!    starts): the pair `(run starts in the chunk, value of the last run so far)`. The values
//!    before the last run start are not read.
//! 2. *Carry*: one workgroup per line scans the chunk pairs in order with the pair operator
//!    `(fa, a) . (fb, b) = (fa | fb, fb ? b : op(a, b))`, which is associative whenever `op` is:
//!    the value a run has reached at the end of every chunk.
//! 3. *Scan*: a workgroup scans its chunk of values in workgroup memory with the same pair
//!    operator, seeded with the carry of the chunk before it, reading the head bits instead of the
//!    keys.
//!
//! Chunks of a line are combined in a fixed order, so the results are the same on every run.

use std::fmt::Write;
use std::panic::Location;
use std::sync::Arc;


use crate::array::{Array, Failure};
use crate::builtin::BinOp;
use crate::context::{Context, RawArg, Scratch};
use crate::error::LumaError;
use crate::gpu_type::{ElemLayout, Numeric, ScalarType};
use crate::jit::structured::boundary::ScanAttrs;
use crate::jit::structured::strategy::{candidates, note, Boundary, BoundaryAttrs, FamilyTag, Params};
use crate::jit::{ElemTy, Operator};
use crate::ops::{fold_workgroups, Outcome, Rhs};
use crate::scan::ScanOp;
use crate::scatter::AtomicElem;

/// Invocations of a workgroup of the kernels.
const WORKGROUP: u32 = 256;
/// Cells an invocation owns in the fold and scan kernels.
const PER_THREAD: u32 = 16;
/// Cells of a chunk.
pub(crate) const CHUNK: u32 = WORKGROUP * PER_THREAD;
/// Words of head bits of a chunk.
const CHUNK_WORDS: u32 = CHUNK / 32;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pass {
    Fold,
    Carry,
    /// The scan of a chunk; `bits`: the run starts come from the bits the fold wrote (else the
    /// kernel reads the keys, for a line of one chunk).
    Scan { bits: bool },
}

impl Pass {
    fn label(self) -> &'static str {
        match self {
            Pass::Fold => "ScanByKeyFold",
            Pass::Carry => "ScanByKeyCarry",
            Pass::Scan { .. } => "ScanByKeyScan",
        }
    }
}

/// The WGSL of one pass for values of `ty`, keys of `kty` and the operator `op`.
fn source(pass: Pass, ty: &str, kty: &str, op: &Operator) -> String {
    let mut src = String::from("struct Params { n: u32, lines: u32, nb: u32, ws: u32, ");
    for (k, scalar) in op.scalar_tys.iter().enumerate() {
        let _ = write!(src, "s{k}: {}, ", scalar.wgsl());
    }
    src.push_str("}\n");
    let binding = |src: &mut String, at: u32, access: &str, name: &str, elem: &str| {
        let _ = writeln!(src, "@group(1) @binding({at}) var<storage, {access}> {name}: array<{elem}>;");
    };
    let uniform = match pass {
        Pass::Fold => {
            binding(&mut src, 0, "read_write", "agg_f", "u32");
            binding(&mut src, 1, "read_write", "agg_v", ty);
            binding(&mut src, 2, "read_write", "hbits", "u32");
            binding(&mut src, 3, "read", "keys", kty);
            binding(&mut src, 4, "read", "a", ty);
            5
        }
        Pass::Carry => {
            binding(&mut src, 0, "read_write", "incl", ty);
            binding(&mut src, 1, "read", "agg_f", "u32");
            binding(&mut src, 2, "read", "agg_v", ty);
            3
        }
        Pass::Scan { bits } => {
            binding(&mut src, 0, "read_write", "out", ty);
            if bits {
                binding(&mut src, 1, "read", "hbits", "u32");
            } else {
                binding(&mut src, 1, "read", "keys", kty);
            }
            binding(&mut src, 2, "read", "a", ty);
            binding(&mut src, 3, "read", "incl", ty);
            4
        }
    };
    let _ = writeln!(src, "@group(1) @binding({uniform}) var<uniform> p: Params;");
    src.push_str(&op.source);
    let (w, k) = (WORKGROUP, PER_THREAD);
    let tile = CHUNK + CHUNK / 32 + 1;
    if pass == Pass::Carry {
        let _ = write!(
            src,
            "var<workgroup> part_v: array<{ty}, {w}>;
var<workgroup> part_f: array<u32, {w}>;
var<workgroup> part_h: array<u32, {w}>;
@compute @workgroup_size({w})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let l = wg.x + wg.y * nwg.x;
    if (l >= p.lines) {{ return; }}
    let per = (p.nb + {w}u - 1u) / {w}u;
    let lo = min(t * per, p.nb);
    let hi = min(lo + per, p.nb);
    let row = l * p.nb;
    var f = 0u;
    var h = 0u;
    var v: {ty};
    for (var c = lo; c < hi; c++) {{
        let cf = agg_f[row + c];
        let cv = agg_v[row + c];
        if (h == 0u) {{ v = cv; f = cf; h = 1u; }} else {{
            if (cf == 0u) {{ v = op(v, cv); }} else {{ v = cv; }}
            f = f | cf;
        }}
    }}
    part_v[t] = v;
    part_f[t] = f;
    part_h[t] = h;
    for (var d = 1u; d < {w}u; d <<= 1u) {{
        workgroupBarrier();
        var nv = part_v[t];
        var nf = part_f[t];
        var nh = part_h[t];
        if (t >= d && part_h[t - d] != 0u) {{
            let pv = part_v[t - d];
            let pf = part_f[t - d];
            if (nh == 0u) {{ nv = pv; nf = pf; nh = 1u; }} else {{
                if (nf == 0u) {{ nv = op(pv, nv); }}
                nf = nf | pf;
            }}
        }}
        workgroupBarrier();
        part_v[t] = nv;
        part_f[t] = nf;
        part_h[t] = nh;
    }}
    workgroupBarrier();
    var have = false;
    var run: {ty};
    if (t > 0u && part_h[t - 1u] != 0u) {{ have = true; run = part_v[t - 1u]; }}
    for (var c = lo; c < hi; c++) {{
        let cf = agg_f[row + c];
        let cv = agg_v[row + c];
        if (!have || cf != 0u) {{ run = cv; }} else {{ run = op(run, cv); }}
        have = true;
        incl[row + c] = run;
    }}
}}
"
        );
        return src;
    }
    let from_bits = matches!(pass, Pass::Scan { bits: true });
    let _ = writeln!(src, "var<workgroup> tile: array<{ty}, {tile}>;");
    if !from_bits {
        let _ = writeln!(src, "var<workgroup> ktile: array<{kty}, {tile}>;\nvar<workgroup> prevk: {kty};");
    }
    let _ = write!(
        src,
        "var<workgroup> part_v: array<{ty}, {w}>;
var<workgroup> part_f: array<u32, {w}>;
fn ix(j: u32) -> u32 {{ return j + (j >> 5u); }}
"
    );
    if !from_bits {
        src.push_str(
            "fn is_head(j: u32, c: u32) -> bool {
    if (j == 0u) { return c == 0u || ktile[ix(0u)] != prevk; }
    return ktile[ix(j)] != ktile[ix(j - 1u)];
}
",
        );
    }
    if pass == Pass::Fold {
        let _ = write!(
            src,
            "var<workgroup> part_h: array<u32, {w}>;
var<workgroup> part_l: array<u32, {w}>;
@compute @workgroup_size({w})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let wid = wg.x + wg.y * nwg.x;
    if (wid >= p.lines * p.nb) {{ return; }}
    let l = wid / p.nb;
    let c = wid - l * p.nb;
    let base = l * p.n;
    let s0 = c * {CHUNK}u;
    let len = min(p.n - s0, {CHUNK}u);
    if (t == 0u && c > 0u) {{ prevk = keys[base + s0 - 1u]; }}
    for (var j = t; j < len; j += {w}u) {{ ktile[ix(j)] = keys[base + s0 + j]; }}
    workgroupBarrier();
    let lo = t * {k}u;
    let hi = min(len, lo + {k}u);
    var last = 0u;
    for (var j = lo; j < hi; j++) {{ if (is_head(j, c)) {{ last = j + 1u; }} }}
    part_l[t] = last;
    if (t < {words}u && t * 32u < len) {{
        var word = 0u;
        let end = min(32u, len - t * 32u);
        for (var i = 0u; i < end; i++) {{ if (is_head(t * 32u + i, c)) {{ word |= 1u << i; }} }}
        hbits[l * p.ws + c * {words}u + t] = word;
    }}
    for (var d = {half}u; d > 0u; d >>= 1u) {{
        workgroupBarrier();
        if (t < d) {{ part_l[t] = max(part_l[t], part_l[t + d]); }}
    }}
    workgroupBarrier();
    let top = part_l[0];
    var start = 0u;
    if (top > 0u) {{ start = top - 1u; }}
    for (var j = start + t; j < len; j += {w}u) {{ tile[ix(j)] = a[base + s0 + j]; }}
    workgroupBarrier();
    let lo2 = max(lo, start);
    var acc: {ty};
    var has = 0u;
    for (var j = lo2; j < hi; j++) {{
        let x = tile[ix(j)];
        if (has == 0u) {{ acc = x; has = 1u; }} else {{ acc = op(acc, x); }}
    }}
    part_v[t] = acc;
    part_h[t] = has;
    for (var d = 1u; d < {w}u; d <<= 1u) {{
        workgroupBarrier();
        var nv = part_v[t];
        var nh = part_h[t];
        if (t >= d && part_h[t - d] != 0u) {{
            if (nh == 0u) {{ nv = part_v[t - d]; nh = 1u; }} else {{ nv = op(part_v[t - d], nv); }}
        }}
        workgroupBarrier();
        part_v[t] = nv;
        part_h[t] = nh;
    }}
    workgroupBarrier();
    if (t == 0u) {{
        let last_thread = (len - 1u) / {k}u;
        agg_f[wid] = select(0u, 1u, top > 0u);
        agg_v[wid] = part_v[last_thread];
    }}
}}
",
            words = CHUNK_WORDS,
            half = WORKGROUP / 2,
        );
        return src;
    }
    // The scan of a chunk.
    let head_of = if from_bits { "((hb >> (j - lo)) & 1u) != 0u" } else { "is_head(j, c)" };
    let _ = write!(
        src,
        "@compute @workgroup_size({w})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let wid = wg.x + wg.y * nwg.x;
    if (wid >= p.lines * p.nb) {{ return; }}
    let l = wid / p.nb;
    let c = wid - l * p.nb;
    let base = l * p.n;
    let s0 = c * {CHUNK}u;
    let len = min(p.n - s0, {CHUNK}u);
"
    );
    if from_bits {
        src.push_str("    var hb = 0u;
    {
        let lo0 = t * 16u;
        if (lo0 < len) { hb = (hbits[l * p.ws + c * 128u + (lo0 >> 5u)] >> (lo0 & 31u)) & 0xffffu; }
    }
");
        src.push_str("    for (var j = t; j < len; j += 256u) { tile[ix(j)] = a[base + s0 + j]; }
");
    } else {
        src.push_str("    if (t == 0u && c > 0u) { prevk = keys[base + s0 - 1u]; }
    for (var j = t; j < len; j += 256u) { tile[ix(j)] = a[base + s0 + j]; ktile[ix(j)] = keys[base + s0 + j]; }
");
    }
    let _ = write!(
        src,
        "    workgroupBarrier();
    let lo = t * {k}u;
    let hi = min(len, lo + {k}u);
    var f = 0u;
    var acc: {ty};
    if (lo < hi) {{
        var have = false;
        for (var j = lo; j < hi; j++) {{
            let x = tile[ix(j)];
            if ({head_of}) {{ f = 1u; acc = x; have = true; }}
            else if (have) {{ acc = op(acc, x); }}
            else {{ acc = x; have = true; }}
        }}
    }}
    part_v[t] = acc;
    part_f[t] = f;
    for (var d = 1u; d < {w}u; d <<= 1u) {{
        workgroupBarrier();
        var nv = part_v[t];
        var nf = part_f[t];
        if (t >= d) {{
            let pv = part_v[t - d];
            let pf = part_f[t - d];
            if (nf == 0u) {{ nv = op(pv, nv); }}
            nf = nf | pf;
        }}
        workgroupBarrier();
        part_v[t] = nv;
        part_f[t] = nf;
    }}
    workgroupBarrier();
    var have = false;
    var run: {ty};
    if (t > 0u) {{
        have = true;
        run = part_v[t - 1u];
        if (part_f[t - 1u] == 0u && c > 0u) {{ run = op(incl[wid - 1u], run); }}
    }} else if (c > 0u) {{
        have = true;
        run = incl[wid - 1u];
    }}
    for (var j = lo; j < hi; j++) {{
        let x = tile[ix(j)];
        if ({head_of} || !have) {{ run = x; have = true; }} else {{ run = op(run, x); }}
        tile[ix(j)] = run;
    }}
    workgroupBarrier();
    for (var j = t; j < len; j += {w}u) {{ out[base + s0 + j] = tile[ix(j)]; }}
}}
"
    );
    src
}

/// Which cells of `shape` start a run along `axis`: the first of every line, and every cell whose
/// key differs from the one before it.
fn heads<K: ScalarType>(keys: &Array<K>, axis: usize) -> Array<bool> {
    let mut offsets = [0isize; 4];
    offsets[axis] = 1;
    let previous = keys.shift(offsets);
    keys.ne(&previous) | keys.index(axis).eq(0u32)
}

impl<T: Numeric> Array<T> {
    /// The inclusive scan of `op` along `axis` restarted at every run of equal `keys` (see the
    /// module documentation): element `i` of the result folds, with `op`, the elements from the
    /// first of the run that `i` belongs to up to `i`. A run never continues from one line along
    /// `axis` into the next. `keys` has the shape of this array and the type `u32`, `i32` or
    /// `f32` (keys compare with `==`, so NaN keys are runs of one element). `op` is as for
    /// [`Array::scan`]: associative, an elementwise expression of its two arguments. The results
    /// are the same on every run. Arrays are evaluated first.
    ///
    /// Along an axis other than the first the lines are moved to be contiguous by a tiled
    /// transpose and back again.
    #[track_caller]
    pub fn scan_by_key<K: ScalarType>(&self, keys: &Array<K>, axis: usize, op: impl Fn(&Array<T>, &Array<T>) -> Array<T>) -> Array<T> {
        let location = Location::caller();
        let result = self.run_scan_by_key(keys, axis, &op);
        Array::resolve(self.context(), self.shape(), result, "scan_by_key", location)
    }

    /// The exclusive scan by key: element `i` folds the elements of its run before `i`, and the
    /// first element of every run is `identity` (the value `op(identity, x) == x`, 0 for a sum).
    #[track_caller]
    pub fn scan_by_key_exclusive<K: ScalarType>(&self, keys: &Array<K>, axis: usize, identity: T, op: impl Fn(&Array<T>, &Array<T>) -> Array<T>) -> Array<T> {
        let location = Location::caller();
        let result = self.run_scan_by_key(keys, axis, &op).and_then(|inclusive| {
            let mut offsets = [0isize; 4];
            offsets[axis] = 1;
            let first = heads(keys, axis);
            let fill = Array::<T>::full_on(self.context(), self.shape(), identity)?;
            Ok(crate::ops::select_at(&first, &fill, inclusive.shift_fill(offsets, identity), location))
        });
        Array::resolve(self.context(), self.shape(), result, "scan_by_key_exclusive", location)
    }

    /// The running sum along `axis` that restarts at every run of equal `keys` (a
    /// [`scan_by_key`](Array::scan_by_key) with `+`).
    #[track_caller]
    pub fn cumsum_by_key<K: ScalarType>(&self, keys: &Array<K>, axis: usize) -> Array<T> {
        let location = Location::caller();
        let result = self.run_scan_by_key(keys, axis, &crate::scan::add);
        Array::resolve(self.context(), self.shape(), result, "cumsum_by_key", location)
    }

    fn run_scan_by_key<K: ScalarType>(&self, keys: &Array<K>, axis: usize, op: &ScanOp<'_, T>) -> Outcome<Array<T>> {
        let ctx = self.context();
        let shape = self.shape();
        if axis >= 4 {
            return Err(LumaError::InvalidShape(format!("axis {axis} of shape {shape}")).into());
        }
        if !ctx.same_as(keys.context()) {
            return Err(LumaError::ContextMismatch.into());
        }
        if shape != keys.shape() {
            return Err(LumaError::ShapeMismatch { lhs: shape.dims(), rhs: keys.shape().dims() }.into());
        }
        if T::LAYOUT != ElemLayout::plain(4) || K::LAYOUT != ElemLayout::plain(4) {
            return Err(LumaError::Unsupported("scan_by_key needs values and keys of 4 bytes (u32, i32, f32)".into()).into());
        }
        let dims = shape.dims();
        let (n, stride) = (dims[axis], dims[..axis].iter().product::<usize>());
        let outer = self.len() / (n * stride);
        if stride > 1 {
            // Lines made contiguous: `[stride, n, outer]` to `[n, stride, outer]`.
            let to_lines = |a: &Array<T>| a.reshape([stride, n, outer]).swap_axes(0, 1);
            let values = to_lines(self);
            let keys = keys.reshape([stride, n, outer]).swap_axes(0, 1);
            values.request().map_err(Failure::New)?;
            keys.request().map_err(Failure::New)?;
            let scanned = values.run_scan_by_key(&keys, 0, op)?;
            let back = scanned.reshape([n, stride, outer]).swap_axes(0, 1).reshape(shape);
            back.request().map_err(Failure::New)?;
            return Ok(back);
        }
        let operator = self.trace_operator(op)?;
        let (pin, bytes) = self.ready()?;
        let (kpin, kbytes) = keys.ready()?;
        let out = Array::<T>::alloc(ctx, shape)?;
        let out_storage = Arc::clone(out.storage()?);
        let out_pin = out_storage.pin_single()?;
        let lines = u32::try_from(self.len() / n).map_err(|_| LumaError::DimensionTooLarge { shape: dims })?;
        let n32 = u32::try_from(n).map_err(|_| LumaError::DimensionTooLarge { shape: dims })?;
        let boundary = Boundary::new(FamilyTag::Scan, "scan_by_key", Location::caller(), [n32, 1, lines, 0], BoundaryAttrs::Scan(ScanAttrs { grouping_free: false, by_key: true }));
        let (chosen, params) = candidates(ctx, &boundary).next().expect("a by-key scan strategy always fits");
        note(ctx, &boundary, chosen, &params, &[]);
        let nb = match params {
            Params::Scan(scan) => scan.nb,
            _ => unreachable!("a scan strategy runs with scan parameters"),
        };
        let units = u64::from(lines) * u64::from(nb);
        let ws = n32.div_ceil(32);
        let mut scratch = Scratch::new(ctx);
        let tiny = scratch.take(16)?;
        let (agg_f, agg_v, incl, hbits) = if nb > 1 {
            let words = units * 4;
            (scratch.take(words)?, scratch.take(words)?, scratch.take(words)?, scratch.take(u64::from(lines) * u64::from(ws) * 4)?)
        } else {
            (tiny.clone(), tiny.clone(), tiny.clone(), tiny.clone())
        };
        let ty = T::wgsl_type();
        let kty = K::wgsl_type();
        let mut words = vec![n32, lines, nb, ws];
        words.extend_from_slice(&operator.scalars);
        let uniform = RawArg::Uniform(bytemuck::cast_slice(&words));
        let kernel = |pass: Pass| ctx.jit().stencil_kernel(ctx, source(pass, ty, kty, &operator), pass.label());
        let units32 = u32::try_from(units).map_err(|_| LumaError::InvalidShape("too many chunks".into()))?;
        let grid = fold_workgroups(ctx, units32)?;
        let (unit_bytes, input) = (units * 4, RawArg::read(pin.buffer(), bytes));
        let keys_arg = RawArg::read(kpin.buffer(), kbytes);
        let hbits_bytes = if nb > 1 { u64::from(lines) * u64::from(ws) * 4 } else { 16 };
        if nb > 1 {
            let (fold, carry) = (kernel(Pass::Fold)?, kernel(Pass::Carry)?);
            ctx.launch(&fold, grid, [units32, 1, 1], &[RawArg::write(&agg_f, unit_bytes), RawArg::write(&agg_v, unit_bytes), RawArg::write(&hbits, hbits_bytes), keys_arg, input, uniform])?;
            ctx.launch(&carry, fold_workgroups(ctx, lines)?, [lines, 1, 1], &[RawArg::write(&incl, unit_bytes), RawArg::read(&agg_f, unit_bytes), RawArg::read(&agg_v, unit_bytes), uniform])?;
        }
        let incl_bytes = if nb > 1 { unit_bytes } else { 16 };
        let scan = kernel(Pass::Scan { bits: nb > 1 })?;
        let second = if nb > 1 { RawArg::read(&hbits, hbits_bytes) } else { keys_arg };
        ctx.launch(&scan, grid, [units32, 1, 1], &[RawArg::write(out_pin.buffer(), out.byte_len()), second, input, RawArg::read(&incl, incl_bytes), uniform])?;
        drop(scratch);
        crate::jit::check_finite(ctx, &out_storage, ElemTy::of::<T>()?)?;
        Ok(out)
    }
}

/// The offsets (start of every run, then the length) and the first key of every run of the
/// elements of `keys` in flat order; reads the number of runs back, so it synchronizes once.
fn runs<K: ScalarType>(ctx: &Context, keys: &Array<K>) -> Outcome<(Array<u32>, Array<K>)> {
    let n = keys.len();
    let flat = keys.reshape(n);
    let starts = heads(&flat, 0).nonzero().map_err(Failure::New)?.expect("the first element starts a run");
    let first_keys = flat.take(&starts);
    let end = Array::<u32>::full_on(ctx, 1usize, u32::try_from(n).map_err(|_| LumaError::DimensionTooLarge { shape: [n, 1, 1, 1] })?)?;
    let offsets = crate::view::concat_slice(0, &[starts, end]);
    offsets.request().map_err(Failure::New)?;
    Ok((offsets, first_keys))
}

fn mul<T: ScalarType>(x: &Array<T>, y: &Array<T>) -> Array<T> {
    x.binary(Rhs::Array(y), BinOp::Mul, false, "mul")
}

/// `reduce(values, offsets)` over the runs of `keys`: the first key of every run and the result.
fn by_key<K: ScalarType, V: ScalarType, R: ScalarType>(
    values: &Array<V>,
    keys: &Array<K>,
    reduce: impl FnOnce(&Array<V>, &Array<u32>, usize) -> Outcome<Array<R>>,
) -> crate::Result<(Array<K>, Array<R>)> {
    let run = || -> Outcome<(Array<K>, Array<R>)> {
        let ctx = values.context();
        if !ctx.same_as(keys.context()) {
            return Err(LumaError::ContextMismatch.into());
        }
        if values.shape() != keys.shape() {
            return Err(LumaError::ShapeMismatch { lhs: values.shape().dims(), rhs: keys.shape().dims() }.into());
        }
        let (offsets, first_keys) = runs(ctx, keys)?;
        let runs = first_keys.len();
        let reduced = reduce(&values.reshape(values.len()), &offsets, runs)?;
        Ok((first_keys, reduced))
    };
    run().map_err(Failure::into_error)
}

/// Reductions of every run of equal consecutive keys (see the module documentation): the first
/// key of every run and the reduction of the values of the run, both one-dimensional arrays of
/// one element per run, in order. The number of runs depends on the data, so these read it back and
/// **synchronize** once; `keys` has the shape of the values and the type `u32`, `i32` or `f32`.
/// Pending arrays are evaluated first.
impl<V: AtomicElem> Array<V> {
    /// The sum of the values of every run (as [`Array::sum_segments`]: integers wrap, `f32` sums
    /// are the same on every run).
    pub fn sum_by_key<K: ScalarType>(&self, keys: &Array<K>) -> crate::Result<(Array<K>, Array<V>)> {
        by_key(self, keys, |values, offsets, _| Ok(values.sum_segments(offsets)))
    }

    /// The minimum of the values of every run.
    pub fn min_by_key<K: ScalarType>(&self, keys: &Array<K>) -> crate::Result<(Array<K>, Array<V>)> {
        by_key(self, keys, |values, offsets, _| Ok(values.min_segments(offsets)))
    }

    /// The maximum of the values of every run.
    pub fn max_by_key<K: ScalarType>(&self, keys: &Array<K>) -> crate::Result<(Array<K>, Array<V>)> {
        by_key(self, keys, |values, offsets, _| Ok(values.max_segments(offsets)))
    }
}

impl<V: Numeric> Array<V> {
    /// The product of the values of every run, from a [`scan_by_key`](Array::scan_by_key) with
    /// `*` read at the last element of each run.
    pub fn product_by_key<K: ScalarType>(&self, keys: &Array<K>) -> crate::Result<(Array<K>, Array<V>)> {
        by_key(self, keys, |values, offsets, runs| {
            let scanned = values.run_scan_by_key(&keys.reshape(keys.len()), 0, &mul)?;
            let last = &offsets.slice(0, 1..runs + 1) - 1u32;
            Ok(scanned.take(&last))
        })
    }
}

impl<V: ScalarType> Array<V> {
    /// The number of non-zero values of every run, `u32` (ArrayFire's `countByKey`).
    pub fn count_by_key<K: ScalarType>(&self, keys: &Array<K>) -> crate::Result<(Array<K>, Array<u32>)> {
        by_key(self, keys, |values, offsets, _| {
            let zero = Array::<V>::zeros_on(values.context(), values.shape())?;
            Ok(values.ne(&zero).cast_at::<u32>(Location::caller()).sum_segments(offsets))
        })
    }

    /// Whether any value of every run is non-zero (ArrayFire's `anyTrueByKey`).
    pub fn any_true_by_key<K: ScalarType>(&self, keys: &Array<K>) -> crate::Result<(Array<K>, Array<bool>)> {
        by_key(self, keys, |values, offsets, _| {
            let zero = Array::<V>::zeros_on(values.context(), values.shape())?;
            Ok(values.ne(&zero).cast_at::<u32>(Location::caller()).max_segments(offsets).ne(0u32))
        })
    }

    /// Whether all values of every run are non-zero (ArrayFire's `allTrueByKey`).
    pub fn all_true_by_key<K: ScalarType>(&self, keys: &Array<K>) -> crate::Result<(Array<K>, Array<bool>)> {
        by_key(self, keys, |values, offsets, _| {
            let zero = Array::<V>::zeros_on(values.context(), values.shape())?;
            Ok(values.ne(&zero).cast_at::<u32>(Location::caller()).min_segments(offsets).ne(0u32))
        })
    }
}
