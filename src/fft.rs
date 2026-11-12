//! FFT convolution as a few passes over the data.
//!
//! The forward transform is decimation in frequency (natural order in, bit-reversed order out)
//! and the inverse is decimation in time (bit-reversed in, natural out), so no pass ever
//! permutes the data. Every pass moves one tile of complex values (as many as fit in the
//! device's workgroup memory, see [`tile_log`]) and applies several radix-2 stages to it:
//!
//! * a *local* pass works on contiguous blocks and applies the stages of the smallest distances
//!   (up to half a tile), four at a time in registers (radix 16: one exchange through workgroup
//!   memory per four stages). In the convolution's middle pass it also multiplies by the
//!   kernel's spectrum and runs the first inverse stages, all without leaving the tile;
//! * a *columns* pass applies up to [`MAX_COLUMN_STAGES`] stages of larger distances, one at a
//!   time in workgroup memory: a tile is a few adjacent columns (so loads stay coalesced) of the
//!   rows those stages combine.
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
// A local pass's register groups need at most one thread each from a workgroup.
const _: () = assert!(TILE_LOG <= 12);
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
    Local { multiply: bool, k: u32 },
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
        matches!(self.kind, PassKind::Local { multiply: true, .. })
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
    /// Workgroups that have a tile; the grid may launch more.
    groups: u32,
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
        "struct Params { s: u32, k: u32, in_len: u32, out_len: u32, inv_n: f32, lead: u32, step: u32, log_b: u32, groups: u32 }\n",
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
        PassKind::Local { multiply, k } => src.push_str(&local_main(multiply, k)),
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

/// Stages applied to `2^r` values in registers, spaced `2^s` apart within blocks of `2^(s + r)`.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Group {
    s: u32,
    r: u32,
}

/// The `k` forward stages as groups of four, largest distances first, then the remainder.
fn forward_groups(k: u32) -> Vec<Group> {
    let mut groups: Vec<Group> = (0..k / 4)
        .map(|i| Group {
            s: k - 4 * (i + 1),
            r: 4,
        })
        .collect();
    if !k.is_multiple_of(4) {
        groups.push(Group { s: 0, r: k % 4 });
    }
    groups
}

/// The `k` inverse stages, smallest distances first: the remainder, then groups of four.
fn inverse_groups(k: u32) -> Vec<Group> {
    let tail = k % 4;
    let mut groups = Vec::new();
    if tail != 0 {
        groups.push(Group { s: 0, r: tail });
    }
    groups.extend((0..k / 4).map(|i| Group {
        s: tail + 4 * i,
        r: 4,
    }));
    groups
}

fn literal(x: f64) -> String {
    format!("{:?}", x as f32)
}

/// `v * e^(sign 2 pi i num / den)` for constants, as an expression.
fn rotate(v: &str, num: u32, den: u32, sign: f64) -> String {
    let angle = 2.0 * std::f64::consts::PI * f64::from(num) / f64::from(den);
    let (c, s) = (angle.cos(), sign * angle.sin());
    if num == 0 {
        v.to_string()
    } else if 2 * num == den {
        format!("(-{v})")
    } else if 4 * num == den || 4 * num == 3 * den {
        // Multiplication by +i or -i.
        if s > 0.0 {
            format!("vec2<f32>(-{v}.y, {v}.x)")
        } else {
            format!("vec2<f32>({v}.y, -{v}.x)")
        }
    } else if (8 * num).is_multiple_of(den) {
        // Odd multiples of pi/4: both parts have magnitude 1/sqrt(2).
        let sc = if c > 0.0 { "" } else { "-" };
        let re = format!("{sc}{v}.x {} {v}.y", if s > 0.0 { "-" } else { "+" });
        let im = format!("{sc}{v}.y {} {v}.x", if s > 0.0 { "+" } else { "-" });
        let r = literal(std::f64::consts::FRAC_1_SQRT_2);
        format!("(vec2<f32>({re}, {im}) * {r})")
    } else {
        format!("cmul({v}, vec2<f32>({}, {}))", literal(c), literal(s))
    }
}

fn bit_reverse(p: u32, r: u32) -> u32 {
    p.reverse_bits() >> (32 - r)
}

/// Where a region reads its values from or writes them to: global memory, or workgroup memory
/// at the `sx` swizzle of the index.
#[derive(Clone, Copy, PartialEq)]
enum Mem {
    Global,
    Swizzled,
}

/// One thread's step of a local pass: it reads its run of values (`group`) into registers,
/// applies the forward group if `fwd` (decimation in frequency: butterflies with constant
/// twiddles, then each value times `w^bit_reverse(position)`), the product with the spectrum
/// `h` if `multiply`, the inverse group if `inv` (the same steps mirrored), and writes the run
/// back. A region with both `fwd` and `inv` applies the same group in both directions.
struct Region {
    group: Group,
    fwd: bool,
    multiply: bool,
    inv: bool,
    read: Mem,
    write: Mem,
}

impl Region {
    /// The twiddle `w = e^(sign 2 pi i j / 2^(s + r))` of the thread's first value, applied to
    /// every value as `w^bit_reverse(position)`. Powers are products of two nearer ones, used
    /// as soon as they exist to keep few of them live.
    fn twiddles(g: Group, sign: f64) -> String {
        let mut out = format!("let p1 = twiddle(j, {}u, {sign:?});\n", g.s + g.r);
        for e in 1..1u32 << g.r {
            if e > 1 {
                out += &format!("let p{e} = cmul(p{}, p{});\n", e / 2, e - e / 2);
            }
            let m = bit_reverse(e, g.r);
            out += &format!("x{m} = cmul(x{m}, p{e});\n");
        }
        out
    }

    fn forward(g: Group) -> String {
        let n = 1u32 << g.r;
        let mut out = String::new();
        for a in 0..g.r {
            let dist = n >> (a + 1);
            for m in (0..n).filter(|m| m & dist == 0) {
                let rotated = rotate("d", m & (dist - 1), 2 * dist, -1.0);
                out += &format!(
                    "{{ let u = x{m}; let v = x{hi}; x{m} = u + v; let d = u - v; x{hi} = {rotated}; }}\n",
                    hi = m + dist
                );
            }
        }
        if g.s > 0 {
            out += &Self::twiddles(g, -1.0);
        }
        out
    }

    fn inverse(g: Group) -> String {
        let n = 1u32 << g.r;
        let mut out = String::new();
        if g.s > 0 {
            out += &Self::twiddles(g, 1.0);
        }
        for a in 0..g.r {
            let dist = 1u32 << a;
            for m in (0..n).filter(|m| m & dist == 0) {
                let hi = m + dist;
                let rotated = rotate(&format!("x{hi}"), m & (dist - 1), 2 * dist, 1.0);
                out += &format!("{{ let u = x{m}; let v = {rotated}; x{m} = u + v; x{hi} = u - v; }}\n");
            }
        }
        out
    }

    /// The statements for a tile of `2^k` values.
    fn source(&self, k: u32) -> String {
        let g = self.group;
        let n = 1u32 << g.r;
        let spacing = 1u32 << g.s;
        let at = |m: u32| format!("(i0 + {}u)", spacing * m);
        let mut body = String::new();
        for m in 0..n {
            body += &match self.read {
                Mem::Global => format!("var x{m} = load(base + {});\n", at(m)),
                Mem::Swizzled => format!("var x{m} = sh[sx({})];\n", at(m)),
            };
        }
        if self.fwd {
            body += &Self::forward(g);
        }
        if self.multiply {
            for m in 0..n {
                body += &format!("x{m} = cmul(x{m}, h[(base + {}) & mask]);\n", at(m));
            }
        }
        if self.inv {
            body += &Self::inverse(g);
        }
        for m in 0..n {
            body += &match self.write {
                Mem::Global => format!("store(base + {}, x{m});\n", at(m)),
                Mem::Swizzled => format!("sh[sx({})] = x{m};\n", at(m)),
            };
        }
        let threads = 1u32 << (k - g.r);
        let guard = if threads < WORKGROUP {
            format!("if (lid < {threads}u) ")
        } else {
            String::new()
        };
        let barrier = if self.write == Mem::Global {
            ""
        } else {
            "workgroupBarrier();\n"
        };
        format!(
            "{guard}{{
let q = lid;
let mask = (1u << p.log_b) - 1u;
let j = q & {}u;
let i0 = ((q >> {}u) << {}u) | j;
{body}}}
{barrier}",
            spacing - 1,
            g.s,
            g.s + g.r
        )
    }
}

/// The body of a local pass over a tile of `2^k` values. Values are exchanged through workgroup
/// memory between groups only; the last forward group and the first inverse group are the same
/// run per thread, so the product with the spectrum sits between them in registers.
///
/// Invariant: every region reads and writes workgroup memory at its own slots under one index
/// mapping (the `sx` swizzle, which also spreads runs of 16 values over distinct memory banks).
/// Mixing mappings inside a region makes a thread overwrite a slot another thread has not read
/// yet, which is a data race that only goes unnoticed where 16 or more invocations run in
/// lockstep.
fn local_body(multiply: bool, k: u32) -> String {
    if k == 0 {
        let value = if multiply {
            "cmul(load(base), h[base & ((1u << p.log_b) - 1u)])"
        } else {
            "load(base)"
        };
        return format!("if (lid == 0u) {{ store(base, {value}); }}\n");
    }
    let forward = forward_groups(k);
    let mut regions: Vec<Region> = Vec::new();
    for (i, &group) in forward.iter().enumerate() {
        // With the product, the last forward group and the first inverse group are one run.
        let fused = multiply && i + 1 == forward.len();
        regions.push(Region { group, fwd: true, multiply: fused, inv: fused, read: Mem::Swizzled, write: Mem::Swizzled });
    }
    if multiply {
        let inverse = inverse_groups(k);
        debug_assert_eq!(forward.last(), inverse.first());
        regions.extend(inverse.into_iter().skip(1).map(|group| Region {
            group,
            fwd: false,
            multiply: false,
            inv: true,
            read: Mem::Swizzled,
            write: Mem::Swizzled,
        }));
    }
    if let Some(first) = regions.first_mut() {
        first.read = Mem::Global;
    }
    if multiply && let Some(last) = regions.last_mut() {
        last.write = Mem::Global;
    }
    let mut body = String::new();
    for region in &regions {
        body += &region.source(k);
    }
    if !multiply {
        body += &format!(
            "for (var e = lid; e < {}u; e = e + {WORKGROUP}u) {{ store(base + e, sh[sx(e)]); }}\n",
            1u32 << k
        );
    }
    body
}

fn local_main(multiply: bool, k: u32) -> String {
    format!(
        "
// Swaps the low four bits with bits four to seven, so runs of 16 consecutive values hit
// distinct memory banks.
fn sx(i: u32) -> u32 {{ return i ^ ((i >> 4u) & 15u); }}

@compute @workgroup_size({WORKGROUP})
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let g = wid.x + wid.y * nwg.x;
    if (g >= p.groups) {{ return; }}
    let base = g << p.k;
{}}}
",
        local_body(multiply, k)
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
    if (g >= p.groups) {{ return; }}
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
            let groups = batch.total >> local;
            let params = Params {
                s,
                k,
                groups,
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
                fold_workgroups(ctx, groups)?,
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
            PassKind::Local {
                multiply: false,
                k: local,
            },
            (0, local),
            source(columns.is_empty()),
            Io::Complex,
        );
    };
    let local_out = if columns.is_empty() { out } else { Io::Complex };
    launch(
        PassKind::Local {
            multiply: true,
            k: local,
        },
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
        groups: 0,
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
