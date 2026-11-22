//! Digital filters along dimension 0: [`Array::fir`] (direct form, with the FFT convolution for
//! long filters), [`Array::iir`] and [`Array::iir_sos`] (recursive filters, computed in parallel
//! over the length of the signal).
//!
//! The kernels are written in WGSL and run through [`crate::Kernel`], so they work on every
//! backend. Their inputs must fit one buffer ([`LumaError::Chunked`] otherwise).
//!
//! # The recursive filter in parallel
//!
//! A filter `y[i] = v[i] - a1 y[i-1] - .. - ap y[i-p]` is a linear recurrence on the state
//! `(y[i], .., y[i-p+1])`: `s[i] = A s[i-1] + (v[i], 0, ..)`. A segment of the signal started
//! from the zero state ends in some state `e`; started from a state `s` it ends in `A^len s + e`,
//! so the end state of a longer stretch is found by folding the segments' `e`s with the constant
//! matrix power `A^len` (computed once on the host in `f64`). Three kernels use that:
//!
//! 1. every workgroup takes a chunk, every invocation a piece of it, runs the recurrence over
//!    its piece from zero and folds the pieces' end states: the chunk's end state from zero;
//! 2. one workgroup per signal folds the chunk end states the same way, into the state entering
//!    each chunk (every invocation takes a run of chunks, one invocation combines the runs);
//! 3. every workgroup runs step 1's fold again, seeds the pieces with their true entering states
//!    and runs the recurrence once more, writing the output.
//!
//! The signal is read twice and written once, and the longest sequential chain is a piece, a
//! run of chunks, or the 64 pieces / runs of a workgroup, whichever is longest.

use std::fmt::Write as _;
use std::panic::Location;

use crate::array::{Array, Failure};
use crate::error::LumaError;
use crate::kernel::{uniform, KernelArg};
use crate::ops::{fold_workgroups, Outcome};
use crate::shape::Shape;

/// The longest filter the direct FIR kernel runs; longer ones use the FFT convolution.
const FIR_DIRECT_MAX: usize = 256;
/// Filters at least this long use the FFT convolution when the signal is a single vector (see
/// `Array::fir`).
const FIR_FFT_MIN: usize = 4096;
/// Outputs per workgroup of the FIR kernel (`FIR_WORKGROUP` invocations of eight outputs).
const FIR_TILE: u32 = 1024;
const FIR_WORKGROUP: u32 = 128;
/// The largest recursion order (`a.len() - 1`) of the IIR kernels.
pub const IIR_MAX_ORDER: usize = 8;

#[derive(crate::encase::ShaderType)]
struct FirParams {
    n: u32,
    k: u32,
    tiles: u32,
    /// Signals (the product of the batch extents).
    batch: u32,
    /// Distance between the signals of `x` (`n`, or 0 when one signal is shared).
    x_stride: u32,
    /// Distance between the filters (`k`, or 0 when one filter is shared).
    b_stride: u32,
}

/// The shared-memory address of element `a`: one pad word per 32, so that invocations reading
/// every eighth word (their own run of eight outputs) meet 32 different banks.
fn pad(a: usize) -> usize {
    a + (a >> 5)
}

/// The direct FIR kernel. A workgroup computes `FIR_TILE` outputs of one signal; every invocation
/// eight consecutive ones, in registers, over blocks of eight taps: the eight inputs a block
/// brings in (the window of the 15 inputs that eight outputs and eight taps touch slides by
/// eight per block) and the eight taps are all it loads from workgroup memory for 64
/// multiply-adds. The outputs go through workgroup memory so that the global stores are
/// coalesced.
fn fir_source() -> String {
    // Window variable of the input at offset `d` (-7..=7) from the block's first output.
    let var = |d: i32| match d {
        0 => "x0".to_string(),
        d if d < 0 => format!("xm{}", -d),
        d => format!("xp{d}"),
    };
    let mut w = String::new();
    let _ = write!(
        w,
        r#"
struct P {{ n: u32, k: u32, tiles: u32, batch: u32, x_stride: u32, b_stride: u32 }}
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read> b: array<f32>;
@group(1) @binding(2) var<storage, read_write> y: array<f32>;
@group(1) @binding(3) var<uniform> p: P;

const TILE: u32 = {tile}u;
const WG: u32 = {wg}u;
var<workgroup> xs: array<f32, {xs}>;
var<workgroup> bs: array<f32, {kmax}>;

fn pad(a: u32) -> u32 {{ return a + (a >> 5u); }}

@compute @workgroup_size({wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) t: u32) {{
    let g = wid.x + wid.y * nwg.x;
    if (g >= p.tiles * p.batch) {{ return; }}
    let sig = g / p.tiles;
    let t0 = (g % p.tiles) * TILE;
    // The taps in blocks of eight; the padding taps are 0.
    let kp = (p.k + 7u) & 0xfffffff8u;
    let xo = sig * p.x_stride;
    let bo = sig * p.b_stride;
    for (var j = t; j < kp; j += WG) {{
        var c = 0.0;
        if (j < p.k) {{ c = b[bo + j]; }}
        bs[j] = c;
    }}
    let span = TILE + kp - 1u;
    for (var j = t; j < span; j += WG) {{
        let gi = i32(t0) + i32(j) - i32(kp - 1u);
        var v = 0.0;
        if (gi >= 0 && gi < i32(p.n)) {{ v = x[xo + u32(gi)]; }}
        xs[pad(j)] = v;
    }}
    workgroupBarrier();
    let o = t * 8u;
    // The first input of the block's first output; it moves down by 8 per block.
    var a0 = o + kp - 1u;
"#,
        tile = FIR_TILE,
        wg = FIR_WORKGROUP,
        kmax = FIR_DIRECT_MAX,
        xs = pad(FIR_TILE as usize + FIR_DIRECT_MAX + 8) + 1,
    );
    for r in 0..8 {
        let _ = writeln!(w, "    var acc{r} = 0.0;");
    }
    for d in -7..=7i32 {
        let _ = writeln!(w, "    var {} = xs[pad(a0 - 7u + {}u)];", var(d), (d + 7) as u32);
    }
    let _ = writeln!(w, "    let blocks = kp >> 3u;\n    for (var mb = 0u; mb < blocks; mb += 1u) {{");
    for u in 0..8 {
        let _ = writeln!(w, "        let b{u} = bs[mb * 8u + {u}u];");
    }
    for r in 0..8i32 {
        for u in 0..8i32 {
            let _ = writeln!(w, "        acc{r} = fma(b{u}, {}, acc{r});", var(r - u));
        }
    }
    let _ = writeln!(w, "        if (mb + 1u < blocks) {{\n            a0 -= 8u;");
    // Offsets 1..=7 of the next block are offsets -7..=-1 of this one.
    for d in 1..=7i32 {
        let _ = writeln!(w, "            {} = {};", var(d), var(d - 8));
    }
    for d in -7..=0i32 {
        let _ = writeln!(w, "            {} = xs[pad(a0 - 7u + {}u)];", var(d), (d + 7) as u32);
    }
    let _ = writeln!(w, "        }}\n    }}\n    workgroupBarrier();");
    for r in 0..8 {
        let _ = writeln!(w, "    xs[pad(o + {r}u)] = acc{r};");
    }
    let _ = writeln!(
        w,
        "    workgroupBarrier();
    for (var j = t; j < TILE; j += WG) {{
        if (t0 + j < p.n) {{ y[sig * p.n + t0 + j] = xs[pad(j)]; }}
    }}
}}"
    );
    w
}

/// The shapes of a filter operation on `x` (`[n, batch..]`) and coefficients (`[k, batch..]`):
/// the batch extents of the output and the strides between the signals.
fn batch_layout(x: Shape, b: Shape) -> Result<(Shape, u32, u32), LumaError> {
    let (xd, bd) = (x.dims(), b.dims());
    let xb: usize = xd[1..].iter().product();
    let bb: usize = bd[1..].iter().product();
    let out = if bb == 1 {
        x
    } else if xb == 1 {
        Shape::new([xd[0], bd[1], bd[2], bd[3]])
    } else if xd[1..] == bd[1..] {
        x
    } else {
        return Err(LumaError::InvalidShape(format!("fir: the batch extents {:?} of the signal and {:?} of the filter coefficients differ", &xd[1..], &bd[1..])));
    };
    let stride = |n: usize, batch: usize| if batch == 1 { 0 } else { n as u32 };
    Ok((out, stride(xd[0], xb), stride(bd[0], bb)))
}

fn fail<T: crate::gpu_type::GpuType>(ctx: &crate::Context, shape: Shape, error: LumaError, op: &'static str, location: &'static Location<'static>) -> Array<T> {
    let result: Outcome<Array<T>> = Err(Failure::New(error));
    Array::resolve(ctx, shape, result, op, location)
}

impl Array<f32> {
    /// The finite impulse response filter with coefficients `b` applied to this signal along
    /// dimension 0, like ArrayFire's `fir(b, x)`: `y[i] = b[0] x[i] + b[1] x[i-1] + ..` with
    /// zeros before the start; the result has the signal's shape.
    ///
    /// Batches: the signal `[n, ..]` can have batch dimensions 1..3, and `b` (`[k, ..]`) is either
    /// one filter for all of them, or has the same batch extents as the signal (one filter per
    /// signal); a signal that is a single vector with batched filters is repeated.
    ///
    /// Filters of up to 256 taps run as a direct kernel (inputs of a workgroup staged in workgroup
    /// memory). Longer filters of a single vector signal use the FFT convolution
    /// ([`Array::conv1d_fft`]) from 4096 taps on and otherwise
    /// a direct kernel per block of 256 taps; batched signals with long filters are filtered one
    /// by one.
    ///
    /// Fails (poisoned array) if `b` is empty or the batch extents do not match, and with
    /// [`LumaError::Chunked`] if an array is split over several buffers.
    ///
    /// ```
    /// use luma::Array;
    /// let x = Array::from_slice(&[1.0f32, 4.0, 2.0, 8.0], 4)?;
    /// let b = Array::from_slice(&[0.5f32, 0.25], 2)?;
    /// assert_eq!(x.fir(&b).to_vec()?, vec![0.5, 2.25, 2.0, 4.5]);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn fir(&self, b: &Array<f32>) -> Array<f32> {
        let location = Location::caller();
        let result = self.run_fir(b);
        Array::resolve(self.context(), self.shape(), result, "fir", location)
    }

    fn run_fir(&self, b: &Array<f32>) -> Outcome<Array<f32>> {
        let ctx = self.context();
        self.src()?;
        b.src()?;
        if !ctx.same_as(b.context()) {
            return Err(LumaError::ContextMismatch.into());
        }
        let (n, k) = (self.shape().dims()[0], b.shape().dims()[0]);
        let (out_shape, x_stride, b_stride) = batch_layout(self.shape(), b.shape())?;
        let to_u32 = |v: usize| u32::try_from(v).map_err(|_| LumaError::DimensionTooLarge { shape: self.shape().dims() });
        if k > FIR_DIRECT_MAX {
            return self.fir_long(b, out_shape);
        }
        let batch = to_u32(out_shape.len() / n)?;
        let tiles = to_u32(n.div_ceil(FIR_TILE as usize))?;
        let groups = tiles.checked_mul(batch).ok_or(LumaError::DimensionTooLarge { shape: out_shape.dims() })?;
        let grid = fold_workgroups(ctx, groups)?;
        let kernel = ctx.kernel(&fir_source()).build()?;
        let mut out = Array::<f32>::alloc(ctx, out_shape)?;
        let params = FirParams { n: to_u32(n)?, k: to_u32(k)?, tiles, batch, x_stride, b_stride };
        kernel.launch_raw(grid, [KernelArg::from(self), KernelArg::from(b), KernelArg::from(&mut out), uniform(&params)])?;
        Ok(out)
    }

    /// Filters longer than the direct kernel's limit: the FFT convolution for a single vector,
    /// else one filtering per signal.
    fn fir_long(&self, b: &Array<f32>, out_shape: Shape) -> Outcome<Array<f32>> {
        let (n, k) = (self.shape().dims()[0], b.shape().dims()[0]);
        let batch = out_shape.len() / n;
        let (xb, bb) = (self.len() / n, b.len() / k);
        let one = |x: &Array<f32>, b: &Array<f32>| -> Array<f32> {
            if k >= FIR_FFT_MIN {
                x.conv1d_fft(b).slice(0, 0..n)
            } else {
                // Blocks of taps: y = sum over blocks of fir(block) applied to the signal delayed
                // by the block's offset.
                let mut sum: Option<Array<f32>> = None;
                for start in (0..k).step_by(FIR_DIRECT_MAX) {
                    let end = k.min(start + FIR_DIRECT_MAX);
                    let part = x.fir(&b.slice(0, start..end).copy()).shift_fill(start as isize, 0.0);
                    sum = Some(match sum {
                        Some(s) => s + part,
                        None => part,
                    });
                }
                sum.expect("a filter has taps")
            }
        };
        if batch == 1 {
            return Ok(one(&self.reshape([n]), &b.reshape([k])).reshape(out_shape));
        }
        let x2 = self.reshape([n, xb]);
        let b2 = b.reshape([k, bb]);
        let parts: Vec<Array<f32>> = (0..batch)
            .map(|i| {
                let xi = x2.slice(1, if xb == 1 { 0..1 } else { i..i + 1 }).copy().reshape([n]);
                let bi = b2.slice(1, if bb == 1 { 0..1 } else { i..i + 1 }).copy().reshape([k]);
                one(&xi, &bi).reshape([n, 1])
            })
            .collect();
        Ok(crate::view::concat_slice(1, &parts).reshape(out_shape))
    }
}

// -- Recursive filters --

/// Invocations per workgroup of the recursive kernels.
const IIR_WORKGROUP: u32 = 64;
/// Samples per invocation: a chunk is `IIR_WORKGROUP * IIR_PIECE` samples.
const IIR_PIECE: u32 = 48;
const IIR_CHUNK: usize = (IIR_WORKGROUP * IIR_PIECE) as usize;

#[derive(crate::encase::ShaderType)]
struct IirParams {
    n: u32,
    chunks: u32,
    batch: u32,
    /// Chunks per invocation of the carry kernel.
    run: u32,
    a: [V4; 2],
    /// `A^piece`, `A^chunk` and `A^(chunk * run)`, row-major `order x order`, padded.
    m_piece: [V4; 16],
    m_chunk: [V4; 16],
    m_run: [V4; 16],
}

/// Four floats; the elements of the uniform arrays have the 16-byte stride uniform arrays need.
#[derive(crate::encase::ShaderType, Clone, Copy, Default)]
struct V4 {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

/// The component of float number `i` of a `V4` array.
fn comp(i: usize) -> &'static str {
    ["x", "y", "z", "w"][i % 4]
}

/// The `order x order` companion matrix of `y[i] = -sum a_j y[i-j]` on the state
/// `(y[i], .., y[i-order+1])`, row-major.
fn companion(a: &[f64]) -> Vec<f64> {
    let p = a.len();
    let mut m = vec![0.0; p * p];
    for j in 0..p {
        m[j] = -a[j];
    }
    for i in 1..p {
        m[i * p + i - 1] = 1.0;
    }
    m
}

fn mat_mul(a: &[f64], b: &[f64], p: usize) -> Vec<f64> {
    let mut c = vec![0.0; p * p];
    for i in 0..p {
        for k in 0..p {
            let v = a[i * p + k];
            for j in 0..p {
                c[i * p + j] += v * b[k * p + j];
            }
        }
    }
    c
}

fn mat_pow(m: &[f64], p: usize, mut e: u64) -> Vec<f64> {
    let mut result: Vec<f64> = (0..p * p).map(|i| if i / p == i % p { 1.0 } else { 0.0 }).collect();
    let mut base = m.to_vec();
    while e > 0 {
        if e & 1 == 1 {
            result = mat_mul(&result, &base, p);
        }
        base = mat_mul(&base, &base, p);
        e >>= 1;
    }
    result
}

fn pack<const N: usize>(values: &[f64]) -> [V4; N] {
    let mut out = [V4::default(); N];
    for (i, v) in values.iter().enumerate() {
        let c = &mut out[i / 4];
        let v = *v as f32;
        match i % 4 {
            0 => c.x = v,
            1 => c.y = v,
            2 => c.z = v,
            _ => c.w = v,
        }
    }
    out
}

/// The source text shared by the three recursive kernels: the uniform, constants and the code of
/// one sample step and of a matrix-vector product, for order `p`.
fn iir_prelude(p: usize) -> String {
    let mut w = String::new();
    let _ = writeln!(
        w,
        "struct V4 {{ x: f32, y: f32, z: f32, w: f32 }}
struct U {{ n: u32, chunks: u32, batch: u32, run: u32, a: array<V4, 2>, ms: array<V4, 16>, mc: array<V4, 16>, mr: array<V4, 16> }}
const P: u32 = {p}u;
const WG: u32 = {wg}u;
const S: u32 = {piece}u;
const PITCH: u32 = {pitch}u;
const CH: u32 = {chunk}u;",
        wg = IIR_WORKGROUP,
        piece = IIR_PIECE,
        pitch = IIR_PIECE + 1,
        chunk = IIR_CHUNK,
    );
    w
}

/// `let r0 = ..; let r1 = ..;` of `M * s` for the matrix `m` (an array in the uniform `u`) and
/// the state variables `s0..`, then the names `r0..`.
fn matvec(w: &mut String, m: &str, src: &str, dst: &str, p: usize, indent: &str) {
    for i in 0..p {
        let terms: Vec<String> = (0..p).map(|j| format!("u.{m}[{}].{} * {src}{j}", (i * p + j) / 4, comp(i * p + j))).collect();
        let _ = writeln!(w, "{indent}let {dst}{i} = {};", terms.join(" + "));
    }
}

/// One recurrence step on the state `s0..`: `y = v - a0*s0 - ..`, shift, `s0 = y`.
fn step(w: &mut String, v: &str, p: usize, indent: &str) {
    let terms: String = (0..p).map(|j| format!(" - a{j} * s{j}")).collect();
    let _ = writeln!(w, "{indent}let y = {v}{terms};");
    for j in (1..p).rev() {
        let _ = writeln!(w, "{indent}s{j} = s{};", j - 1);
    }
    let _ = writeln!(w, "{indent}s0 = y;");
}

fn coefficients(w: &mut String, p: usize) {
    for j in 0..p {
        let _ = writeln!(w, "    let a{j} = u.a[{}].{};", j / 4, comp(j));
    }
}

fn state_vars(w: &mut String, p: usize) {
    for j in 0..p {
        let _ = writeln!(w, "    var s{j} = 0.0;");
    }
}

/// The cooperative, coalesced load of chunk `c` of signal `sig` into the padded workgroup array.
const LOAD: &str = "
    for (var i = t; i < CH; i += WG) {
        let idx = c * CH + i;
        var v = 0.0;
        if (idx < u.n) { v = x[sig * u.n + idx]; }
        sh[(i / S) * PITCH + i % S] = v;
    }
    workgroupBarrier();";

/// Kernel 1: the end state of every chunk from the zero state.
fn iir_totals_source(p: usize) -> String {
    let mut w = iir_prelude(p);
    let _ = write!(
        w,
        "
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> tot: array<f32>;
@group(1) @binding(2) var<uniform> u: U;
var<workgroup> sh: array<f32, {sh}>;
var<workgroup> es: array<f32, {es}>;

@compute @workgroup_size({wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) t: u32) {{
    let g = wid.x + wid.y * nwg.x;
    if (g >= u.chunks * u.batch) {{ return; }}
    let sig = g / u.chunks;
    let c = g % u.chunks;
",
        sh = IIR_WORKGROUP * (IIR_PIECE + 1),
        es = IIR_WORKGROUP as usize * p,
        wg = IIR_WORKGROUP,
    );
    w.push_str(LOAD);
    w.push('\n');
    coefficients(&mut w, p);
    state_vars(&mut w, p);
    let _ = writeln!(w, "    for (var j = 0u; j < S; j += 1u) {{\n        let v = sh[t * PITCH + j];");
    step(&mut w, "v", p, "        ");
    let _ = writeln!(w, "    }}");
    for j in 0..p {
        let _ = writeln!(w, "    es[t * P + {j}u] = s{j};");
    }
    let _ = writeln!(w, "    workgroupBarrier();\n    if (t == 0u) {{");
    for j in 0..p {
        let _ = writeln!(w, "        var z{j} = 0.0;");
    }
    let _ = writeln!(w, "        for (var q = 0u; q < WG; q += 1u) {{");
    matvec(&mut w, "ms", "z", "m", p, "            ");
    for j in 0..p {
        let _ = writeln!(w, "            z{j} = m{j} + es[q * P + {j}u];");
    }
    let _ = writeln!(w, "        }}");
    for j in 0..p {
        let _ = writeln!(w, "        tot[g * P + {j}u] = z{j};");
    }
    let _ = writeln!(w, "    }}\n}}");
    w
}

/// Kernel 2: the state entering every chunk, one workgroup per signal.
fn iir_carries_source(p: usize) -> String {
    let mut w = iir_prelude(p);
    let _ = write!(
        w,
        "
@group(1) @binding(0) var<storage, read> tot: array<f32>;
@group(1) @binding(1) var<storage, read_write> carry: array<f32>;
@group(1) @binding(2) var<uniform> u: U;
var<workgroup> fs: array<f32, {es}>;

@compute @workgroup_size({wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) t: u32) {{
    let sig = wid.x + wid.y * nwg.x;
    if (sig >= u.batch) {{ return; }}
    let first = min(t * u.run, u.chunks);
    let last = min(first + u.run, u.chunks);
    let base = sig * u.chunks;
",
        es = IIR_WORKGROUP as usize * p,
        wg = IIR_WORKGROUP,
    );
    // The total of this invocation's run.
    for j in 0..p {
        let _ = writeln!(w, "    var z{j} = 0.0;");
    }
    let _ = writeln!(w, "    for (var c = first; c < last; c += 1u) {{");
    matvec(&mut w, "mc", "z", "m", p, "        ");
    for j in 0..p {
        let _ = writeln!(w, "        z{j} = m{j} + tot[(base + c) * P + {j}u];");
    }
    let _ = writeln!(w, "    }}");
    for j in 0..p {
        let _ = writeln!(w, "    fs[t * P + {j}u] = z{j};");
    }
    let _ = writeln!(w, "    workgroupBarrier();\n    if (t == 0u) {{");
    // The state entering each run, stored in `fs` over the totals (the run's own total is not
    // needed after the one before it was folded in).
    for j in 0..p {
        let _ = writeln!(w, "        var r{j} = 0.0;");
    }
    let _ = writeln!(w, "        for (var q = 0u; q < WG; q += 1u) {{");
    for j in 0..p {
        let _ = writeln!(w, "            let f{j} = fs[q * P + {j}u];\n            fs[q * P + {j}u] = r{j};");
    }
    matvec(&mut w, "mr", "r", "m", p, "            ");
    for j in 0..p {
        let _ = writeln!(w, "            r{j} = m{j} + f{j};");
    }
    let _ = writeln!(w, "        }}\n    }}\n    workgroupBarrier();");
    for j in 0..p {
        let _ = writeln!(w, "    var c{j} = fs[t * P + {j}u];");
    }
    let _ = writeln!(w, "    for (var c = first; c < last; c += 1u) {{");
    for j in 0..p {
        let _ = writeln!(w, "        carry[(base + c) * P + {j}u] = c{j};");
    }
    matvec(&mut w, "mc", "c", "m", p, "        ");
    for j in 0..p {
        let _ = writeln!(w, "        c{j} = m{j} + tot[(base + c) * P + {j}u];");
    }
    let _ = writeln!(w, "    }}\n}}");
    w
}

/// Kernel 3: the output.
fn iir_apply_source(p: usize) -> String {
    let mut w = iir_prelude(p);
    let _ = write!(
        w,
        "
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read> carry: array<f32>;
@group(1) @binding(2) var<storage, read_write> out: array<f32>;
@group(1) @binding(3) var<uniform> u: U;
var<workgroup> sh: array<f32, {sh}>;
var<workgroup> es: array<f32, {es}>;

@compute @workgroup_size({wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) t: u32) {{
    let g = wid.x + wid.y * nwg.x;
    if (g >= u.chunks * u.batch) {{ return; }}
    let sig = g / u.chunks;
    let c = g % u.chunks;
",
        sh = IIR_WORKGROUP * (IIR_PIECE + 1),
        es = IIR_WORKGROUP as usize * p,
        wg = IIR_WORKGROUP,
    );
    w.push_str(LOAD);
    w.push('\n');
    coefficients(&mut w, p);
    state_vars(&mut w, p);
    // End state of this piece from zero.
    let _ = writeln!(w, "    for (var j = 0u; j < S; j += 1u) {{\n        let v = sh[t * PITCH + j];");
    step(&mut w, "v", p, "        ");
    let _ = writeln!(w, "    }}");
    for j in 0..p {
        let _ = writeln!(w, "    es[t * P + {j}u] = s{j};");
    }
    let _ = writeln!(w, "    workgroupBarrier();\n    if (t == 0u) {{");
    for j in 0..p {
        let _ = writeln!(w, "        var z{j} = carry[g * P + {j}u];");
    }
    let _ = writeln!(w, "        for (var q = 0u; q < WG; q += 1u) {{");
    for j in 0..p {
        let _ = writeln!(w, "            let e{j} = es[q * P + {j}u];\n            es[q * P + {j}u] = z{j};");
    }
    matvec(&mut w, "ms", "z", "m", p, "            ");
    for j in 0..p {
        let _ = writeln!(w, "            z{j} = m{j} + e{j};");
    }
    let _ = writeln!(w, "        }}\n    }}\n    workgroupBarrier();");
    for j in 0..p {
        let _ = writeln!(w, "    s{j} = es[t * P + {j}u];");
    }
    let _ = writeln!(w, "    for (var j = 0u; j < S; j += 1u) {{\n        let v = sh[t * PITCH + j];");
    step(&mut w, "v", p, "        ");
    let _ = writeln!(w, "        sh[t * PITCH + j] = y;\n    }}\n    workgroupBarrier();");
    let _ = writeln!(
        w,
        "    for (var i = t; i < CH; i += WG) {{
        let idx = c * CH + i;
        if (idx < u.n) {{ out[sig * u.n + idx] = sh[(i / S) * PITCH + i % S]; }}
    }}
}}"
    );
    w
}

/// Checks the coefficients of a recursive filter and returns `b / a0` and `a[1..] / a0` in `f64`.
fn normalize(b: &[f32], a: &[f32]) -> Result<(Vec<f64>, Vec<f64>), LumaError> {
    if b.is_empty() || a.is_empty() {
        return Err(LumaError::InvalidShape("iir: the coefficient lists are empty".into()));
    }
    if a.len() - 1 > IIR_MAX_ORDER {
        return Err(LumaError::InvalidShape(format!("iir: feedback order {} exceeds {IIR_MAX_ORDER}", a.len() - 1)));
    }
    if a[0] == 0.0 || b.iter().chain(a).any(|v| !v.is_finite()) {
        return Err(LumaError::InvalidShape("iir: coefficients must be finite and a[0] not 0".into()));
    }
    let a0 = f64::from(a[0]);
    Ok((b.iter().map(|&v| f64::from(v) / a0).collect(), a[1..].iter().map(|&v| f64::from(v) / a0).collect()))
}

impl Array<f32> {
    /// The all-pole filter `y[i] = v[i] - a[0] y[i-1] - .. - a[p-1] y[i-p]` along dimension 0
    /// (zero initial state), `1 <= p <= IIR_MAX_ORDER`.
    fn run_all_pole(&self, a: &[f64]) -> Outcome<Array<f32>> {
        let ctx = self.context();
        self.src()?;
        let p = a.len();
        let shape = self.shape();
        let n = shape.dims()[0];
        let batch = shape.len() / n;
        let to_u32 = |v: usize| u32::try_from(v).map_err(|_| LumaError::DimensionTooLarge { shape: shape.dims() });
        let chunks = n.div_ceil(IIR_CHUNK);
        let run = chunks.div_ceil(IIR_WORKGROUP as usize);
        let companion_matrix = companion(a);
        let params = IirParams {
            n: to_u32(n)?,
            chunks: to_u32(chunks)?,
            batch: to_u32(batch)?,
            run: to_u32(run)?,
            a: pack(a),
            m_piece: pack(&mat_pow(&companion_matrix, p, u64::from(IIR_PIECE))),
            m_chunk: pack(&mat_pow(&companion_matrix, p, IIR_CHUNK as u64)),
            m_run: pack(&mat_pow(&companion_matrix, p, (IIR_CHUNK * run) as u64)),
        };
        let groups = to_u32(chunks * batch)?;
        let grid = fold_workgroups(ctx, groups)?;
        let grid_signals = fold_workgroups(ctx, to_u32(batch)?)?;
        let totals_kernel = ctx.kernel(&iir_totals_source(p)).build()?;
        let carries_kernel = ctx.kernel(&iir_carries_source(p)).build()?;
        let apply_kernel = ctx.kernel(&iir_apply_source(p)).build()?;
        let mut tot = Array::<f32>::alloc(ctx, Shape::from(chunks * batch * p))?;
        let mut carry = Array::<f32>::alloc(ctx, Shape::from(chunks * batch * p))?;
        let mut out = Array::<f32>::alloc(ctx, shape)?;
        totals_kernel.launch_raw(grid, [KernelArg::from(self), KernelArg::from(&mut tot), uniform(&params)])?;
        carries_kernel.launch_raw(grid_signals, [KernelArg::from(&tot), KernelArg::from(&mut carry), uniform(&params)])?;
        apply_kernel.launch_raw(grid, [KernelArg::from(self), KernelArg::from(&carry), KernelArg::from(&mut out), uniform(&params)])?;
        Ok(out)
    }

    /// The linear recursive (IIR) filter with feedforward coefficients `b` and feedback
    /// coefficients `a` along dimension 0, like ArrayFire's `iir(b, a, x)`:
    /// `a[0] y[i] = b[0] x[i] + .. + b[nb-1] x[i-nb+1] - a[1] y[i-1] - .. - a[na-1] y[i-na+1]`
    /// with a zero initial state; the result has the signal's shape, and batch dimensions (1..3)
    /// are filtered independently. The coefficients are host values (read at call time, so
    /// nothing waits for the GPU); `a[0]` must not be 0 and `a.len() - 1` at most
    /// [`IIR_MAX_ORDER`].
    ///
    /// The feedforward part is a fused elementwise expression for up to 32 taps; the feedback part runs as the
    /// three kernels of the module documentation, in parallel over the length of the signal
    /// (a signal that is one long vector uses the whole GPU). The state is `f32`: poles near the
    /// unit circle accumulate rounding error as in a sequential implementation, but the error of the parallel chunking is
    /// a few ulps, since the chunk transitions are computed in `f64`.
    ///
    /// Fails (poisoned array) on invalid coefficients and with [`LumaError::Chunked`] if the
    /// signal is split over several buffers.
    ///
    /// ```
    /// use luma::Array;
    /// let x = Array::from_slice(&[1.0f32, 0.0, 0.0, 0.0], 4)?;
    /// // y[i] = x[i] + 0.5 y[i-1]: the impulse response 1, 0.5, 0.25, ..
    /// assert_eq!(x.iir(&[1.0], &[1.0, -0.5]).to_vec()?, vec![1.0, 0.5, 0.25, 0.125]);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn iir(&self, b: &[f32], a: &[f32]) -> Array<f32> {
        let location = Location::caller();
        let result = normalize(b, a).map_err(Failure::from).and_then(|(b, a)| self.run_iir(&b, &a));
        Array::resolve(self.context(), self.shape(), result, "iir", location)
    }

    fn run_iir(&self, b: &[f64], a: &[f64]) -> Outcome<Array<f32>> {
        self.src()?;
        let v = if b.len() == 1 {
            self * b[0] as f32
        } else if b.len() <= 32 {
            let mut sum = self * b[0] as f32;
            for (k, &c) in b.iter().enumerate().skip(1) {
                sum = sum + self.shift_fill(k as isize, 0.0) * c as f32;
            }
            sum
        } else {
            let taps: Vec<f32> = b.iter().map(|&c| c as f32).collect();
            self.fir(&Array::from_slice_on(self.context(), &taps, b.len())?)
        };
        if a.is_empty() {
            return Ok(v);
        }
        v.run_all_pole(a)
    }

    /// A cascade of second-order sections: each `[b0, b1, b2, a0, a1, a2]` is a biquad filter
    /// (as [`Array::iir`] with three coefficients each, `a0` normally 1) applied to the output of
    /// the one before. Fails (poisoned array) on invalid coefficients or an empty cascade.
    #[track_caller]
    pub fn iir_sos(&self, sections: &[[f32; 6]]) -> Array<f32> {
        let location = Location::caller();
        if sections.is_empty() {
            return fail(self.context(), self.shape(), LumaError::InvalidShape("iir_sos: no sections".into()), "iir_sos", location);
        }
        let mut y = self.clone();
        for s in sections {
            let result = normalize(&s[..3], &s[3..]).map_err(Failure::from).and_then(|(b, a)| y.run_iir(&b, &a));
            y = Array::resolve(self.context(), self.shape(), result, "iir_sos", location);
        }
        y
    }
}
