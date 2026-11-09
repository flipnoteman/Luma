//! FFT convolution: forward transforms of the zero-padded input and kernel, a pointwise product,
//! and the inverse transform, as a few passes over the data.
//!
//! The forward transform is decimation in frequency (natural order in, bit-reversed order out)
//! and the inverse is decimation in time (bit-reversed in, natural out), so no pass ever
//! permutes the data. Every pass moves one tile of [`TILE`] complex values through workgroup
//! memory and applies several radix-2 stages to it:
//!
//! * a *local* pass works on contiguous blocks and applies the stages of the smallest distances
//!   (up to `TILE / 2`). In the convolution's middle pass it also multiplies by the kernel's
//!   spectrum and runs the first inverse stages, all without leaving workgroup memory;
//! * a *columns* pass applies up to [`MAX_COLUMN_STAGES`] stages of larger distances: a tile is
//!   a few adjacent columns (so loads stay coalesced) of the rows those stages combine.
//!
//! A 2^25-point convolution is two column passes, the fused local pass and two inverse column
//! passes for the input, and three passes for the kernel.

use crate::builtin::Builtin;
use crate::context::{Context, RawArg, Scratch};
use crate::error::LumaError;
use crate::ops::fold_workgroups;

/// Complex values one workgroup holds in workgroup memory (`2^TILE_LOG`).
const TILE_LOG: u32 = 11;
const TILE: u32 = 1 << TILE_LOG;
const WORKGROUP: u32 = 256;
/// Most stages of one columns pass; a tile then has `TILE >> MAX_COLUMN_STAGES` columns.
const MAX_COLUMN_STAGES: u32 = 8;

/// One FFT pass kernel.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct FftPass {
    pub(crate) kind: PassKind,
    /// Loads the real array `src` (zero beyond `in_len`) instead of the complex `data`.
    pub(crate) real_in: bool,
    /// Stores the real parts, scaled by `inv_n`, into `dst` (the first `out_len` only) instead
    /// of the complex `data`.
    pub(crate) real_out: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum PassKind {
    /// Stages of distance `2^s` up to `2^(s + k - 1)`, forward or inverse.
    Columns { inverse: bool },
    /// The `k` forward stages within contiguous blocks of `2^k`; with `multiply`, followed by the
    /// product with the spectrum `h` and the `k` inverse stages.
    Local { multiply: bool },
}

impl FftPass {
    /// Whether the complex buffer `data` is bound: everything but a local pass going from the
    /// real input straight to the real output.
    fn has_data(self) -> bool {
        !(self.real_in && self.real_out)
    }

    fn has_h(self) -> bool {
        matches!(self.kind, PassKind::Local { multiply: true })
    }
}

/// Uniform of every pass.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    s: u32,
    k: u32,
    in_len: u32,
    out_len: u32,
    inv_n: f32,
}

/// The WGSL source of `pass`. Bindings in order: `data`, `src`, `h`, `dst` (those the pass
/// uses), then the parameters.
pub(crate) fn source(pass: FftPass) -> String {
    let mut bindings = Vec::new();
    if pass.has_data() {
        bindings.push("var<storage, read_write> data: array<vec2<f32>>");
    }
    if pass.real_in {
        bindings.push("var<storage, read> src: array<f32>");
    }
    if pass.has_h() {
        bindings.push("var<storage, read> h: array<vec2<f32>>");
    }
    if pass.real_out {
        bindings.push("var<storage, read_write> dst: array<f32>");
    }
    bindings.push("var<uniform> p: Params");
    let mut src =
        String::from("struct Params { s: u32, k: u32, in_len: u32, out_len: u32, inv_n: f32 }\n");
    for (i, binding) in bindings.iter().enumerate() {
        src.push_str(&format!("@group(1) @binding({i}) {binding};\n"));
    }
    let load = if pass.real_in {
        "if (g < p.in_len) { return vec2<f32>(src[g], 0.0); } return vec2<f32>(0.0);"
    } else {
        "return data[g];"
    };
    let store = if pass.real_out {
        "if (g < p.out_len) { dst[g] = v.x * p.inv_n; }"
    } else {
        "data[g] = v;"
    };
    src.push_str(&format!(
        "
var<workgroup> sh: array<vec2<f32>, {TILE}>;

fn load(g: u32) -> vec2<f32> {{ {load} }}
fn store(g: u32, v: vec2<f32>) {{ {store} }}
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}}
// e^(sign * 2 pi i * num / 2^log_den)
fn twiddle(num: u32, log_den: u32, sign: f32) -> vec2<f32> {{
    let a = sign * 6.283185307179586 * (f32(num) / f32(1u << log_den));
    return vec2<f32>(cos(a), sin(a));
}}
"
    ));
    match pass.kind {
        PassKind::Local { multiply } => src.push_str(&local_main(multiply)),
        PassKind::Columns { inverse } => src.push_str(&columns_main(inverse)),
    }
    src
}

/// Butterflies of a stage pairing `lo` with `hi`, the twiddle exponent `num / 2^den` (forward:
/// decimation in frequency; inverse: decimation in time).
fn butterfly(inverse: bool) -> &'static str {
    if inverse {
        "let a = sh[lo]; let b = cmul(sh[hi], twiddle(num, den, 1.0)); sh[lo] = a + b; sh[hi] = a - b;"
    } else {
        "let a = sh[lo]; let b = sh[hi]; sh[lo] = a + b; sh[hi] = cmul(a - b, twiddle(num, den, -1.0));"
    }
}

fn local_main(multiply: bool) -> String {
    let stage = |inverse: bool| {
        let header = if inverse {
            "for (var t = 1u; t <= p.k; t = t + 1u)"
        } else {
            "for (var t = p.k; t > 0u; t = t - 1u)"
        };
        format!(
            "    {header} {{
        let d = 1u << (t - 1u);
        for (var q = lid; q < half; q = q + {WORKGROUP}u) {{
            let num = q & (d - 1u);
            let den = t;
            let lo = ((q >> (t - 1u)) << t) | num;
            let hi = lo + d;
            {}
        }}
        workgroupBarrier();
    }}
",
            butterfly(inverse)
        )
    };
    let middle = if multiply {
        format!(
            "    for (var e = lid; e < size; e = e + {WORKGROUP}u) {{ sh[e] = cmul(sh[e], h[base + e]); }}
    workgroupBarrier();
{}",
            stage(true)
        )
    } else {
        String::new()
    };
    format!(
        "
@compute @workgroup_size({WORKGROUP})
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let size = 1u << p.k;
    let half = size >> 1u;
    let base = (wid.x + wid.y * nwg.x) * size;
    for (var e = lid; e < size; e = e + {WORKGROUP}u) {{ sh[e] = load(base + e); }}
    workgroupBarrier();
{}{middle}    for (var e = lid; e < size; e = e + {WORKGROUP}u) {{ store(base + e, sh[e]); }}
}}
",
        stage(false)
    )
}

fn columns_main(inverse: bool) -> String {
    let header = if inverse {
        "for (var tl = 1u; tl <= p.k; tl = tl + 1u)"
    } else {
        "for (var tl = p.k; tl > 0u; tl = tl - 1u)"
    };
    format!(
        "
@compute @workgroup_size({WORKGROUP})
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    // The tile: `cols` adjacent columns of the `2^k` rows, `2^s` apart, that the stages combine.
    let log_c = {TILE_LOG}u - p.k;
    let cols = 1u << log_c;
    let g = wid.x + wid.y * nwg.x;
    let per_block = (1u << p.s) >> log_c;
    let c0 = (g % per_block) << log_c;
    let origin = ((g / per_block) << (p.s + p.k)) + c0;
    for (var e = lid; e < {TILE}u; e = e + {WORKGROUP}u) {{
        sh[e] = load(origin + ((e >> log_c) << p.s) + (e & (cols - 1u)));
    }}
    workgroupBarrier();
    {header} {{
        let d = 1u << (tl - 1u);
        for (var q = lid; q < {half}u; q = q + {WORKGROUP}u) {{
            let col = q & (cols - 1u);
            let pair = q >> log_c;
            let j = pair & (d - 1u);
            let lo = ((((pair >> (tl - 1u)) << tl) | j) << log_c) | col;
            let hi = lo + (d << log_c);
            let num = (j << p.s) + c0 + col;
            let den = p.s + tl;
            {}
        }}
        workgroupBarrier();
    }}
    for (var e = lid; e < {TILE}u; e = e + {WORKGROUP}u) {{
        store(origin + ((e >> log_c) << p.s) + (e & (cols - 1u)), sh[e]);
    }}
}}
",
        butterfly(inverse),
        half = TILE / 2,
    )
}

/// The buffers of one convolution: real input and kernel, real output, each with the bytes
/// bound and the number of elements.
pub(crate) struct Operands<'a> {
    pub(crate) input: (&'a wgpu::Buffer, u64, u32),
    pub(crate) kernel: (&'a wgpu::Buffer, u64, u32),
    pub(crate) output: (&'a wgpu::Buffer, u64, u32),
}

/// Records the full linear convolution of `ops.input` with `ops.kernel` into `ops.output`, with
/// transforms of `n` (a power of two of at least the output length) points.
pub(crate) fn convolve(
    ctx: &Context,
    ops: &Operands<'_>,
    n: u32,
    scratch: &mut Scratch<'_>,
) -> Result<(), LumaError> {
    let log_n = n.trailing_zeros();
    let local = log_n.min(TILE_LOG);
    let rest = log_n - local;
    // Column passes from the largest distances down, as (s, k).
    let count = rest.div_ceil(MAX_COLUMN_STAGES);
    let mut columns = Vec::new();
    let mut top = log_n;
    for i in 0..count {
        let k = rest / count + u32::from(i < rest % count);
        columns.push((top - k, k));
        top -= k;
    }

    let complex_bytes = u64::from(n) * 8;
    let spectrum = scratch.take(complex_bytes)?;
    let work = scratch.take(complex_bytes)?;
    let inv_n = 1.0 / n as f32;
    let run = |kind: PassKind,
               (s, k): (u32, u32),
               data: &wgpu::Buffer,
               real_in: Option<(&wgpu::Buffer, u64, u32)>,
               real_out: Option<(&wgpu::Buffer, u64, u32)>|
     -> Result<(), LumaError> {
        let pass = FftPass {
            kind,
            real_in: real_in.is_some(),
            real_out: real_out.is_some(),
        };
        let core = ctx.builtin(Builtin::Fft(pass), "f32")?;
        let params = Params {
            s,
            k,
            in_len: real_in.map_or(0, |(_, _, len)| len),
            out_len: real_out.map_or(0, |(_, _, len)| len),
            inv_n,
        };
        let mut args = Vec::with_capacity(5);
        if pass.has_data() {
            args.push(RawArg::write(data, complex_bytes));
        }
        if let Some((buffer, bytes, _)) = real_in {
            args.push(RawArg::read(buffer, bytes));
        }
        if pass.has_h() {
            args.push(RawArg::read(&spectrum, complex_bytes));
        }
        if let Some((buffer, bytes, _)) = real_out {
            args.push(RawArg::write(buffer, bytes));
        }
        args.push(RawArg::Uniform(bytemuck::bytes_of(&params)));
        // Every pass moves one tile per workgroup (one block for a transform smaller than a tile).
        ctx.launch(&core, fold_workgroups(ctx, n >> local)?, [n, 1, 1], &args)
    };
    let forward =
        |source: (&wgpu::Buffer, u64, u32), into: &wgpu::Buffer| -> Result<(), LumaError> {
            for (i, &pass) in columns.iter().enumerate() {
                run(
                    PassKind::Columns { inverse: false },
                    pass,
                    into,
                    (i == 0).then_some(source),
                    None,
                )?;
            }
            Ok(())
        };

    // The kernel's spectrum.
    forward(ops.kernel, &spectrum)?;
    run(
        PassKind::Local { multiply: false },
        (0, local),
        &spectrum,
        columns.is_empty().then_some(ops.kernel),
        None,
    )?;

    // The input: forward, product and inverse in workgroup memory, then the inverse columns.
    forward(ops.input, &work)?;
    run(
        PassKind::Local { multiply: true },
        (0, local),
        &work,
        columns.is_empty().then_some(ops.input),
        columns.is_empty().then_some(ops.output),
    )?;
    for (i, &pass) in columns.iter().enumerate().rev() {
        run(
            PassKind::Columns { inverse: true },
            pass,
            &work,
            None,
            (i == 0).then_some(ops.output),
        )?;
    }
    Ok(())
}
