//! FFT convolution as a few passes over the data.
//!
//! The forward transform is decimation in frequency (natural order in, bit-reversed order out)
//! and the inverse is decimation in time (bit-reversed in, natural out), so no pass ever
//! permutes the data. Every pass moves one tile of complex values (as many as fit in the
//! device's workgroup memory, see [`tile_log`]) through workgroup memory and applies several
//! radix-2 stages to it:
//!
//! * a *local* pass works on contiguous blocks and applies the stages of the smallest distances
//!   (up to half a tile). In the convolution's middle pass it also multiplies by the kernel's
//!   spectrum and runs the first inverse stages, all without leaving workgroup memory;
//! * a *columns* pass applies up to [`MAX_COLUMN_STAGES`] stages of larger distances: a tile is
//!   a few adjacent columns (so loads stay coalesced) of the rows those stages combine.
//!
//! The passes transform a batch of independent `2^log_b`-point blocks laid out one after the
//! other. A convolution either transforms the whole zero-padded input as one block, or (usually)
//! runs *overlap-save*: the input is cut into overlapping blocks of a few times the kernel
//! length, two of them packed into the real and imaginary part of one complex block (the kernel
//! is real, so they stay separate through the product). [`convolve`] picks whichever moves
//! fewer bytes.

use crate::builtin::Builtin;
use crate::context::{Context, RawArg, Scratch};
use crate::error::LumaError;
use crate::ops::fold_workgroups;

/// Tiles of `2^TILE_LOG` complex values (32 KiB) where the device allows, else half that.
const TILE_LOG: u32 = 12;
const WORKGROUP: u32 = 256;
/// Most stages of one columns pass; a tile then has `2^(tile_log - MAX_COLUMN_STAGES)` columns.
const MAX_COLUMN_STAGES: u32 = 8;

/// One FFT pass kernel.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct FftPass {
    pub(crate) kind: PassKind,
    pub(crate) input: Io,
    pub(crate) output: Io,
    /// A tile is `2^tile_log` complex values.
    pub(crate) tile_log: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum PassKind {
    /// Stages of distance `2^s` up to `2^(s + k - 1)`, forward or inverse.
    Columns { inverse: bool },
    /// The `k` forward stages within contiguous blocks of `2^k`; with `multiply`, followed by the
    /// product with the spectrum `h` and the `k` inverse stages.
    Local { multiply: bool },
}

/// Where a pass loads from or stores to.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Io {
    /// The complex work buffer `data`.
    Complex,
    /// The real array `src`/`dst`: loads read zero beyond `in_len`; stores keep the real part,
    /// scaled by `inv_n`, of the first `out_len` values.
    Real,
    /// Overlap-save blocks of the real `src`/`dst`: complex block `q` holds real blocks `2q`
    /// (real part) and `2q + 1` (imaginary part). Real block `b` loads input
    /// `b * step - lead ..`, and stores its values from `lead` on as outputs `b * step ..`.
    Blocks,
}

impl FftPass {
    fn has_data(self) -> bool {
        self.input == Io::Complex || self.output == Io::Complex
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
    lead: u32,
    step: u32,
    /// Size of one transformed block, as a power of two.
    log_b: u32,
}

/// The WGSL source of `pass`. Bindings in order: `data`, `src`, `h`, `dst` (those the pass
/// uses), then the parameters.
pub(crate) fn source(pass: FftPass) -> String {
    let mut bindings = Vec::new();
    if pass.has_data() {
        bindings.push("var<storage, read_write> data: array<vec2<f32>>");
    }
    if pass.input != Io::Complex {
        bindings.push("var<storage, read> src: array<f32>");
    }
    if pass.has_h() {
        bindings.push("var<storage, read> h: array<vec2<f32>>");
    }
    if pass.output != Io::Complex {
        bindings.push("var<storage, read_write> dst: array<f32>");
    }
    bindings.push("var<uniform> p: Params");
    let mut src = String::from(
        "struct Params { s: u32, k: u32, in_len: u32, out_len: u32, inv_n: f32, lead: u32, step: u32, log_b: u32 }\n",
    );
    for (i, binding) in bindings.iter().enumerate() {
        src.push_str(&format!("@group(1) @binding({i}) {binding};\n"));
    }
    let load = match pass.input {
        Io::Complex => "return data[g];",
        Io::Real => "return vec2<f32>(input_at(i32(g)), 0.0);",
        Io::Blocks => {
            "let e = g & ((1u << p.log_b) - 1u);
    let at = i32((g >> p.log_b) * 2u * p.step + e) - i32(p.lead);
    return vec2<f32>(input_at(at), input_at(at + i32(p.step)));"
        }
    };
    let store = match pass.output {
        Io::Complex => "data[g] = v;",
        Io::Real => "if (g < p.out_len) { dst[g] = v.x * p.inv_n; }",
        Io::Blocks => {
            "let e = g & ((1u << p.log_b) - 1u);
    // The first `lead` values of a circular convolution wrap around; the rest are valid.
    if (e < p.lead) { return; }
    let o = (g >> p.log_b) * 2u * p.step + e - p.lead;
    let w = v * p.inv_n;
    if (o < p.out_len) { dst[o] = w.x; }
    if (o + p.step < p.out_len) { dst[o + p.step] = w.y; }"
        }
    };
    let input_at = if pass.input == Io::Complex {
        ""
    } else {
        "fn input_at(g: i32) -> f32 {
    if (g >= 0 && u32(g) < p.in_len) { return src[u32(g)]; }
    return 0.0;
}
"
    };
    src.push_str(&format!(
        "
var<workgroup> sh: array<vec2<f32>, {tile}>;

{input_at}fn load(g: u32) -> vec2<f32> {{
    {load}
}}
fn store(g: u32, v: vec2<f32>) {{
    {store}
}}
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}}
fn conj(a: vec2<f32>) -> vec2<f32> {{ return vec2<f32>(a.x, -a.y); }}
// e^(sign * 2 pi i * num / 2^log_den)
fn twiddle(num: u32, log_den: u32, sign: f32) -> vec2<f32> {{
    let a = sign * 6.283185307179586 * (f32(num) / f32(1u << log_den));
    return vec2<f32>(cos(a), sin(a));
}}
",
        tile = 1u32 << pass.tile_log,
    ));
    match pass.kind {
        PassKind::Local { multiply } => src.push_str(&local_main(multiply)),
        PassKind::Columns { inverse } => src.push_str(&columns_main(inverse, pass.tile_log)),
    }
    src
}

/// Butterflies of a stage pairing `lo` with `hi` (forward: decimation in frequency; inverse:
/// decimation in time). `w` is the forward twiddle `e^(-2 pi i num / 2^den)`; the inverse uses
/// its conjugate.
fn butterfly(inverse: bool, w: &str) -> String {
    if inverse {
        format!("let w = {w}; let a = sh[lo]; let b = cmul(sh[hi], vec2<f32>(w.x, -w.y)); sh[lo] = a + b; sh[hi] = a - b;")
    } else {
        format!(
            "let w = {w}; let a = sh[lo]; let b = sh[hi]; sh[lo] = a + b; sh[hi] = cmul(a - b, w);"
        )
    }
}

/// The `k` stages of a whole tile in workgroup memory, two at a time (radix 4: one read and one
/// write of workgroup memory and one barrier per two stages), plus one radix-2 stage if `k` is
/// odd. Forward stages run from the largest distance down (decimation in frequency), inverse
/// ones from the smallest up (decimation in time).
fn local_stages(inverse: bool) -> String {
    let radix4 = if inverse {
        // Stages t (distance d) and t + 1 (distance 2d) on x0..x3 = base + {0, d, 2d, 3d}.
        "let d = 1u << (t - 1u);
            let j = q & (d - 1u);
            let i0 = ((q >> (t - 1u)) << (t + 1u)) | j;
            let wa = conj(twiddle(j, t, -1.0));
            let x0 = sh[i0]; let x1 = cmul(sh[i0 + d], wa);
            let x2 = sh[i0 + 2u * d]; let x3 = cmul(sh[i0 + 3u * d], wa);
            let a0 = x0 + x1; let a1 = x0 - x1; let a2 = x2 + x3; let a3 = x2 - x3;
            let b2 = cmul(a2, conj(twiddle(j, t + 1u, -1.0)));
            let b3 = cmul(a3, conj(twiddle(j + d, t + 1u, -1.0)));
            sh[i0] = a0 + b2; sh[i0 + 2u * d] = a0 - b2;
            sh[i0 + d] = a1 + b3; sh[i0 + 3u * d] = a1 - b3;"
    } else {
        // Stages t (distance d) and t - 1 (distance d / 2) on x0..x3 = base + {0, h, d, d + h}.
        "let d = 1u << (t - 1u);
            let hd = d >> 1u;
            let j = q & (hd - 1u);
            let i0 = ((q >> (t - 2u)) << t) | j;
            let x0 = sh[i0]; let x1 = sh[i0 + hd]; let x2 = sh[i0 + d]; let x3 = sh[i0 + d + hd];
            let a0 = x0 + x2; let a2 = cmul(x0 - x2, twiddle(j, t, -1.0));
            let a1 = x1 + x3; let a3 = cmul(x1 - x3, twiddle(j + hd, t, -1.0));
            let w = twiddle(j, t - 1u, -1.0);
            sh[i0] = a0 + a1; sh[i0 + hd] = cmul(a0 - a1, w);
            sh[i0 + d] = a2 + a3; sh[i0 + d + hd] = cmul(a2 - a3, w);"
    };
    let radix2 = format!(
        "let d = 1u << (t - 1u);
            let num = q & (d - 1u);
            let lo = ((q >> (t - 1u)) << t) | num;
            let hi = lo + d;
            let den = t;
            {}",
        butterfly(inverse, "twiddle(num, den, -1.0)")
    );
    if inverse {
        format!(
            "    {{
    var t = 1u;
    for (; t + 1u <= p.k; t = t + 2u) {{
        for (var q = lid; q < size / 4u; q = q + {WORKGROUP}u) {{
            {radix4}
        }}
        workgroupBarrier();
    }}
    if (t == p.k) {{
        for (var q = lid; q < half; q = q + {WORKGROUP}u) {{
            {radix2}
        }}
        workgroupBarrier();
    }}
    }}
"
        )
    } else {
        format!(
            "    {{
    var t = p.k;
    for (; t >= 2u; t = t - 2u) {{
        for (var q = lid; q < size / 4u; q = q + {WORKGROUP}u) {{
            {radix4}
        }}
        workgroupBarrier();
    }}
    if (t == 1u) {{
        for (var q = lid; q < half; q = q + {WORKGROUP}u) {{
            {radix2}
        }}
        workgroupBarrier();
    }}
    }}
"
        )
    }
}

fn local_main(multiply: bool) -> String {
    let middle = if multiply {
        format!(
            "    let mask = (1u << p.log_b) - 1u;
    for (var e = lid; e < size; e = e + {WORKGROUP}u) {{ sh[e] = cmul(sh[e], h[(base + e) & mask]); }}
    workgroupBarrier();
{}",
            local_stages(true)
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
        local_stages(false),
    )
}

fn columns_main(inverse: bool, tile_log: u32) -> String {
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
    let log_c = {tile_log}u - p.k;
    let cols = 1u << log_c;
    let g = wid.x + wid.y * nwg.x;
    let per_block = (1u << p.s) >> log_c;
    let c0 = (g % per_block) << log_c;
    let origin = ((g / per_block) << (p.s + p.k)) + c0;
    for (var e = lid; e < {tile}u; e = e + {WORKGROUP}u) {{
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
    for (var e = lid; e < {tile}u; e = e + {WORKGROUP}u) {{
        store(origin + ((e >> log_c) << p.s) + (e & (cols - 1u)), sh[e]);
    }}
}}
",
        butterfly(inverse, "twiddle(num, den, -1.0)"),
        half = (1u32 << tile_log) / 2,
        tile = 1u32 << tile_log,
    )
}

/// The buffers of one convolution: real input and kernel, real output, each with the bytes
/// bound and the number of elements.
pub(crate) struct Operands<'a> {
    pub(crate) input: (&'a wgpu::Buffer, u64, u32),
    pub(crate) kernel: (&'a wgpu::Buffer, u64, u32),
    pub(crate) output: (&'a wgpu::Buffer, u64, u32),
}

/// The tile size of the device: as large as its workgroup memory allows.
fn tile_log(ctx: &Context) -> u32 {
    let bytes = ctx.device().limits().max_compute_workgroup_storage_size;
    if bytes >= 8 << TILE_LOG {
        TILE_LOG
    } else {
        TILE_LOG - 1
    }
}

/// The passes of `2^log_b`-point transforms: the local pass's stage count and the column passes
/// from the largest distances down, as (s, k).
fn plan(log_b: u32, tile_log: u32) -> (u32, Vec<(u32, u32)>) {
    let local = log_b.min(tile_log);
    let rest = log_b - local;
    let count = rest.div_ceil(MAX_COLUMN_STAGES);
    let mut columns = Vec::new();
    let mut top = log_b;
    for i in 0..count {
        let k = rest / count + u32::from(i < rest % count);
        columns.push((top - k, k));
        top -= k;
    }
    (local, columns)
}

/// Passes over the data of a convolution with `2^log_b`-point transforms (the forward columns,
/// the fused local pass and the inverse columns).
fn passes(log_b: u32, tile_log: u32) -> u32 {
    1 + 2 * plan(log_b, tile_log).1.len() as u32
}

/// A batch of transforms to run: what the passes load and store, and their parameters.
struct Batch<'a> {
    log_b: u32,
    tile_log: u32,
    /// Complex values in the work buffer (blocks times block size).
    total: u32,
    work: &'a wgpu::Buffer,
    spectrum: &'a wgpu::Buffer,
    src: (&'a wgpu::Buffer, u64),
    input: Io,
    dst: Option<((&'a wgpu::Buffer, u64), Io)>,
    params: Params,
}

/// Records the forward transform of `batch`; with a `dst`, also the product with the spectrum
/// and the inverse transform into it.
fn run(ctx: &Context, batch: &Batch<'_>) -> Result<(), LumaError> {
    let (local, columns) = plan(batch.log_b, batch.tile_log);
    let work_bytes = u64::from(batch.total) * 8;
    let spectrum_bytes = u64::from(1u32 << batch.log_b) * 8;
    let launch =
        |kind: PassKind, (s, k): (u32, u32), input: Io, output: Io| -> Result<(), LumaError> {
            let pass = FftPass {
                kind,
                input,
                output,
                tile_log: batch.tile_log,
            };
            let core = ctx.builtin(Builtin::Fft(pass), "f32")?;
            let params = Params {
                s,
                k,
                ..batch.params
            };
            let mut args = Vec::with_capacity(5);
            if pass.has_data() {
                args.push(RawArg::write(batch.work, work_bytes));
            }
            if input != Io::Complex {
                args.push(RawArg::read(batch.src.0, batch.src.1));
            }
            if pass.has_h() {
                args.push(RawArg::read(batch.spectrum, spectrum_bytes));
            }
            if output != Io::Complex {
                let ((buffer, bytes), _) = batch
                    .dst
                    .ok_or_else(|| LumaError::Gpu("FFT pass without output".into()))?;
                args.push(RawArg::write(buffer, bytes));
            }
            args.push(RawArg::Uniform(bytemuck::bytes_of(&params)));
            // One tile per workgroup (one block for a transform smaller than a tile).
            ctx.launch(
                &core,
                fold_workgroups(ctx, batch.total >> local)?,
                [batch.total, 1, 1],
                &args,
            )
        };
    let source = |first: bool| if first { batch.input } else { Io::Complex };
    for (i, &stages) in columns.iter().enumerate() {
        launch(
            PassKind::Columns { inverse: false },
            stages,
            source(i == 0),
            Io::Complex,
        )?;
    }
    let Some((_, out)) = batch.dst else {
        return launch(
            PassKind::Local { multiply: false },
            (0, local),
            source(columns.is_empty()),
            Io::Complex,
        );
    };
    let local_out = if columns.is_empty() { out } else { Io::Complex };
    launch(
        PassKind::Local { multiply: true },
        (0, local),
        source(columns.is_empty()),
        local_out,
    )?;
    for (i, &stages) in columns.iter().enumerate().rev() {
        launch(
            PassKind::Columns { inverse: true },
            stages,
            Io::Complex,
            if i == 0 { out } else { Io::Complex },
        )?;
    }
    Ok(())
}

/// Records the full linear convolution of `ops.input` with `ops.kernel` into `ops.output`.
pub(crate) fn convolve(
    ctx: &Context,
    ops: &Operands<'_>,
    scratch: &mut Scratch<'_>,
) -> Result<(), LumaError> {
    let kernel_len = ops.kernel.2;
    let out_len = ops.output.2;
    let tile_log = tile_log(ctx);
    let log_n = out_len
        .checked_next_power_of_two()
        .ok_or(LumaError::DimensionTooLarge {
            shape: [out_len as usize, 1, 1, 1],
        })?
        .trailing_zeros();

    // Bytes moved per output, up to a constant: the transformed values per output times the
    // passes over them (plus, for the single transform, those of the kernel's spectrum).
    let full_cost = f64::from(1u32 << log_n) / f64::from(out_len)
        * f64::from(passes(log_n, tile_log) + passes(log_n, tile_log).div_ceil(2));
    let smallest = (2 * kernel_len)
        .next_power_of_two()
        .trailing_zeros()
        .max(tile_log);
    let blocks = (smallest..smallest + 3)
        .filter(|&log_b| log_b < log_n)
        .map(|log_b| {
            let step = (1u32 << log_b) - (kernel_len - 1);
            (
                log_b,
                f64::from(1u32 << log_b) / f64::from(step) / 2.0
                    * f64::from(passes(log_b, tile_log)),
            )
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .filter(|&(_, cost)| cost < full_cost);

    let base = Params {
        s: 0,
        k: 0,
        in_len: 0,
        out_len,
        inv_n: 0.0,
        lead: 0,
        step: 0,
        log_b: 0,
    };
    let (log_b, total, input, output, lead, step) = match blocks {
        Some((log_b, _)) => {
            let step = (1u32 << log_b) - (kernel_len - 1);
            let pairs = out_len.div_ceil(step).div_ceil(2);
            let total = pairs
                .checked_mul(1 << log_b)
                .ok_or(LumaError::DimensionTooLarge {
                    shape: [out_len as usize, 1, 1, 1],
                })?;
            (log_b, total, Io::Blocks, Io::Blocks, kernel_len - 1, step)
        }
        None => (log_n, 1 << log_n, Io::Real, Io::Real, 0, 0),
    };
    let block = 1u32 << log_b;
    let spectrum = scratch.take(u64::from(block) * 8)?;
    // A single-pass transform never touches the work buffer.
    let work = if plan(log_b, tile_log).1.is_empty() {
        spectrum.clone()
    } else {
        scratch.take(u64::from(total) * 8)?
    };

    // The kernel's spectrum: one block, zero-padded.
    run(
        ctx,
        &Batch {
            log_b,
            tile_log,
            total: block,
            work: &spectrum,
            spectrum: &spectrum,
            src: (ops.kernel.0, ops.kernel.1),
            input: Io::Real,
            dst: None,
            params: Params {
                in_len: kernel_len,
                log_b,
                ..base
            },
        },
    )?;
    run(
        ctx,
        &Batch {
            log_b,
            tile_log,
            total,
            work: &work,
            spectrum: &spectrum,
            src: (ops.input.0, ops.input.1),
            input,
            dst: Some(((ops.output.0, ops.output.1), output)),
            params: Params {
                in_len: ops.input.2,
                inv_n: 1.0 / block as f32,
                lead,
                step,
                log_b,
                ..base
            },
        },
    )
}
