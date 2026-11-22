//! The kernels of the public FFTs: one pass transforms a tile of `cols` lines of up to a few
//! thousand points in workgroup memory, with any mix of radices (2, 4, 8, 16 as register
//! butterflies of radix 2, and the odd primes up to [`MAX_RADIX`] as symmetric-pair butterflies).
//!
//! A transform of `N = D_1 D_2 ... D_p` points is `p` passes of the Stockham autosort form of
//! decimation in frequency: pass `i` views the data as `[outer][D_i][mid][low]` (`low` the points
//! already transformed times the inner batch, `mid` the digits still to come), takes the
//! `D_i`-point transform along the first of them, multiplies by the twiddle
//! `W_(D_i mid)^(k m)` and writes `[outer][mid][D_i][low]`. The last pass has `mid = 1` and
//! leaves the data in natural order, so no pass permutes. A tile is `cols` adjacent `(m, l)`
//! positions, which makes every global access of a pass a run of consecutive elements (see
//! [`crate::spectral`] for how the strides are chosen).
//!
//! Inside the tile the transform is the in-place decimation in frequency: each stage applies a
//! radix-`R` butterfly to values `m` apart and the twiddles of its block, and the result is read
//! back in natural order through the mixed-radix digit reversal of the position.
//!
//! The inverse transform is `conj(F(conj(x)))`: every pass conjugates what it loads and stores
//! (the uniform `cj` is `-1` for the inverse), so one kernel serves both directions.

use std::fmt::Write;

/// Invocations per workgroup.
pub(crate) const WORKGROUP: u32 = 256;
/// Most stages (radices) of one pass.
pub(crate) const MAX_STAGES: usize = 12;
/// The largest radix: the odd primes up to this have a butterfly of their own.
pub(crate) const MAX_RADIX: u32 = 31;

/// What a pass reads from `src`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum LoadKind {
    /// Complex values.
    Complex,
    /// Real values (the imaginary part is zero).
    Real,
    /// Real values packed in pairs, `x[2n] + i x[2n + 1]`: the half-size complex transform of a
    /// real one.
    RealPair,
    /// The half spectrum `0..n_in` of a real signal of `n_full` points, completed to the whole
    /// spectrum by `X[n_full - n] = conj(X[n])`.
    Hermitian,
}

/// What a pass writes to `dst`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum StoreKind {
    /// Complex values.
    Complex,
    /// The real parts.
    Real,
    /// Real and imaginary part as two consecutive reals (the inverse of [`LoadKind::RealPair`]).
    RealPair,
}

/// A table of complex factors multiplied in while loading or storing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Table {
    None,
    /// Indexed by the position along the transformed axis.
    Line,
    /// Indexed by the offset within a `[N][inner]` slab (the same for every outer batch).
    Slab,
}

/// One pass kernel. Everything that is not here (strides, lengths, scale) is in the [`Params`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct DftPass {
    pub(crate) radices: [u8; MAX_STAGES],
    pub(crate) stages: u8,
    /// Lines (columns) per tile.
    pub(crate) cols: u32,
    /// The loads of a tile are runs along the columns (`in_rs == 1`), else along the line.
    pub(crate) in_fast: bool,
    /// The same for the stores.
    pub(crate) out_fast: bool,
    /// Multiply by the pass's twiddle factors (`mid > 1`).
    pub(crate) twiddle: bool,
    pub(crate) load: LoadKind,
    pub(crate) load_table: Table,
    /// Loads are zero where the position along the line is `n_lim` or more.
    pub(crate) load_limit: bool,
    pub(crate) store: StoreKind,
    pub(crate) store_table: Table,
    /// Stores skip the positions `n_out` and beyond.
    pub(crate) store_limit: bool,
    /// The last pass of a forward transform of a real signal's even and odd samples (`inner`
    /// 1, `low > 1`, `cols` even): a tile holds rows `l` and their mirrors `low - l`, so that
    /// the values `Z[nat]` and `Z[h - nat]` are both in workgroup memory, and the pass writes the
    /// half spectrum `X[nat] = E + W^nat O` (`h + 1` values) instead of `Z`.
    pub(crate) post: bool,
}

impl DftPass {
    /// The points of one line.
    pub(crate) fn points(&self) -> u32 {
        self.radices[..self.stages as usize].iter().map(|&r| u32::from(r)).product()
    }

    fn real_in(&self) -> bool {
        matches!(self.load, LoadKind::Real | LoadKind::RealPair)
    }

    fn real_out(&self) -> bool {
        matches!(self.store, StoreKind::Real | StoreKind::RealPair)
    }
}

/// The uniform of every pass (all four-byte words, a multiple of four of them).
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Params {
    /// Workgroups that have a tile.
    pub(crate) tiles: u32,
    /// Rows (lines) in all: `outer * j`.
    pub(crate) rows: u32,
    /// Rows per outer index: `mid * low`.
    pub(crate) j: u32,
    pub(crate) low: u32,
    pub(crate) mid: u32,
    pub(crate) in_rs: u32,
    pub(crate) in_ds: u32,
    pub(crate) in_os: u32,
    pub(crate) out_rs: u32,
    pub(crate) out_ds: u32,
    pub(crate) out_os: u32,
    /// `D * mid`, the denominator of the pass's twiddles.
    pub(crate) den: u32,
    /// The inner batch (elements between consecutive points of a line).
    pub(crate) inner: u32,
    /// `low / inner`: the points already transformed.
    pub(crate) prod: u32,
    pub(crate) n_in: u32,
    pub(crate) n_out: u32,
    pub(crate) n_full: u32,
    pub(crate) scale: f32,
    /// `1` forward, `-1` inverse (conjugation of loads and stores).
    pub(crate) cj: f32,
    /// Per row step of the position `m` (1 when the tile's rows are consecutive `m`, else 0).
    pub(crate) mrs: u32,
    /// With `load_limit`: the positions from here on load as zero.
    pub(crate) n_lim: u32,
    /// With `post`: tiles per outer index, and the rows `l <= low / 2` that are paired with their
    /// mirror `low - l`.
    pub(crate) tpo: u32,
    pub(crate) prodh: u32,
    pub(crate) pad: u32,
}

/// The `f32` literal for `x`.
fn lit(x: f64) -> String {
    let s = format!("{:?}", x as f32);
    if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("NaN") { s } else { format!("{s}.0") }
}

/// A source of fresh names.
struct Names(u32);

impl Names {
    fn next(&mut self) -> String {
        self.0 += 1;
        format!("t{}", self.0)
    }
}

/// `v * e^(-2 pi i num / den)` for constants, as an expression of `v`.
fn rotate(v: &str, num: u32, den: u32) -> String {
    let angle = 2.0 * std::f64::consts::PI * f64::from(num) / f64::from(den);
    let (c, s) = (angle.cos(), -angle.sin());
    if num == 0 {
        v.to_string()
    } else if 2 * num == den {
        format!("(-{v})")
    } else if 4 * num == den {
        format!("vec2<f32>({v}.y, -{v}.x)")
    } else if 4 * num == 3 * den {
        format!("vec2<f32>(-{v}.y, {v}.x)")
    } else {
        format!("cmul({v}, vec2<f32>({}, {}))", lit(c), lit(s))
    }
}

/// Radix-2 decimation in frequency of `v` (a power of two of values) with constant twiddles:
/// emits the code and returns the names of the outputs in natural order.
fn dif(code: &mut String, names: &mut Names, v: Vec<String>) -> Vec<String> {
    let n = v.len();
    if n == 1 {
        return v;
    }
    let h = n / 2;
    let (mut even, mut odd) = (Vec::new(), Vec::new());
    for j in 0..h {
        let s = names.next();
        let d = names.next();
        let _ = writeln!(code, "let {s} = {} + {};", v[j], v[j + h]);
        let _ = writeln!(code, "let {d} = {};", rotate(&format!("({} - {})", v[j], v[j + h]), j as u32, n as u32));
        even.push(s);
        odd.push(d);
    }
    let (e, o) = (dif(code, names, even), dif(code, names, odd));
    let mut out = Vec::with_capacity(n);
    for i in 0..h {
        out.push(e[i].clone());
        out.push(o[i].clone());
    }
    out
}

/// The `q`-point DFT (`q` an odd prime) by symmetric pairs; returns the outputs in natural order.
fn prime_dft(code: &mut String, names: &mut Names, v: &[String]) -> Vec<String> {
    let q = v.len();
    let h = q / 2;
    let (mut sums, mut diffs) = (Vec::new(), Vec::new());
    for j in 1..=h {
        let s = names.next();
        let d = names.next();
        let _ = writeln!(code, "let {s} = {} + {};", v[j], v[q - j]);
        let _ = writeln!(code, "let {d} = {} - {};", v[j], v[q - j]);
        sums.push(s);
        diffs.push(d);
    }
    let mut out = vec![String::new(); q];
    let y0 = names.next();
    let _ = writeln!(code, "let {y0} = {} + {};", v[0], sums.join(" + "));
    out[0] = y0;
    for k in 1..=h {
        let (a, b) = (names.next(), names.next());
        let angle = |j: usize| 2.0 * std::f64::consts::PI * ((j * k) % q) as f64 / q as f64;
        let cos_terms: Vec<String> = (1..=h).map(|j| format!("{} * {}", sums[j - 1], lit(angle(j).cos()))).collect();
        let sin_terms: Vec<String> = (1..=h).map(|j| format!("{} * {}", diffs[j - 1], lit(angle(j).sin()))).collect();
        let _ = writeln!(code, "let {a} = {} + {};", v[0], cos_terms.join(" + "));
        let _ = writeln!(code, "let {b} = {};", sin_terms.join(" + "));
        let (lo, hi) = (names.next(), names.next());
        let _ = writeln!(code, "let {lo} = vec2<f32>({a}.x + {b}.y, {a}.y - {b}.x);");
        let _ = writeln!(code, "let {hi} = vec2<f32>({a}.x - {b}.y, {a}.y + {b}.x);");
        out[k] = lo;
        out[q - k] = hi;
    }
    out
}

/// Code computing the `r`-point DFT of `x0 .. x{r-1}`; returns the names of the outputs in
/// natural order.
fn dft(code: &mut String, names: &mut Names, r: u32) -> Vec<String> {
    let inputs: Vec<String> = (0..r).map(|j| format!("x{j}")).collect();
    if r.is_power_of_two() { dif(code, names, inputs) } else { prime_dft(code, names, &inputs) }
}

/// The shared-memory index of position `pos` of row `r`.
fn at(pass: &DftPass, pos: &str, r: &str) -> String {
    if pass.in_fast {
        format!("sx({pos} * {}u + {r})", pass.cols)
    } else {
        format!("sx({r} * {}u + {pos})", pass.points())
    }
}

/// Stage `s`: every item (row, block, position in block) loads `R` values `m` apart, applies the
/// butterfly and the twiddles of its block, and writes them back.
fn stage(pass: &DftPass, s: usize) -> String {
    let d = pass.points();
    let radices = &pass.radices[..pass.stages as usize];
    let r = u32::from(radices[s]);
    let done: u32 = radices[..=s].iter().map(|&x| u32::from(x)).product();
    let m = d / done;
    let n = r * m;
    let qn = d / r;
    let c = pass.cols;
    let items = c * qn;
    let (row, q) = if pass.in_fast { (format!("w % {c}u"), format!("w / {c}u")) } else { (format!("w / {qn}u"), format!("w % {qn}u")) };
    let mut body = String::new();
    let _ = writeln!(body, "let row = {row}; let q = {q};");
    let _ = writeln!(body, "let blk = q / {m}u; let pp = q - blk * {m}u; let base = blk * {n}u + pp;");
    for j in 0..r {
        let _ = writeln!(body, "var x{j} = sh[{}];", at(pass, &format!("(base + {}u)", j * m), "row"));
    }
    let mut code = String::new();
    let out = dft(&mut code, &mut Names(0), r);
    body.push_str(&code);
    if m > 1 {
        // Twiddles of the block: w^k for the k-th output, e^(-2 pi i pp k / n).
        if r <= 4 {
            for k in 1..r {
                let _ = writeln!(body, "let w{k} = tw(pp * {k}u, {n}u);");
            }
        } else {
            let _ = writeln!(body, "let w1 = tw(pp, {n}u);");
            for k in 2..r {
                let _ = writeln!(body, "let w{k} = cmul(w{}, w{});", k / 2, k - k / 2);
            }
        }
    }
    for k in 0..r {
        let value = if m > 1 && k > 0 { format!("cmul({}, w{k})", out[k as usize]) } else { out[k as usize].clone() };
        let _ = writeln!(body, "sh[{}] = {value};", at(pass, &format!("(base + {}u)", k * m), "row"));
    }
    format!("for (var w = lid; w < {items}u; w = w + {WORKGROUP}u) {{\n{body}}}\nworkgroupBarrier();\n")
}

/// The load of one element: `gi` is its index in `src`, `n` its position along the line.
fn load_value(pass: &DftPass) -> String {
    let mut s = String::new();
    // `n`: the position along the transformed line (first pass only).
    let needs_n = pass.load_table != Table::None || pass.load_limit || matches!(pass.load, LoadKind::RealPair | LoadKind::Hermitian);
    if needs_n {
        s.push_str("let n = d * p.mid + mrow;\n");
    }
    let guard = if pass.load_limit { "if (n < p.n_lim) " } else { "" };
    let raw = match pass.load {
        LoadKind::Complex => "v = src[gi];".to_string(),
        LoadKind::Real => "v = vec2<f32>(src[gi], 0.0);".to_string(),
        LoadKind::RealPair => {
            "let u = gi - ob * p.in_os; let ii = u % p.inner;\nlet i0 = 2u * ob * p.in_os + 2u * u - ii;\nv = vec2<f32>(src[i0], src[i0 + p.inner]);".to_string()
        }
        LoadKind::Hermitian => {
            "let u = gi - ob * p.in_os; let ii = u % p.inner;\nif (n < p.n_in) { v = src[gi]; } else { let g2 = ob * p.in_os + (p.n_full - n) * p.inner + ii; let z = src[g2]; v = vec2<f32>(z.x, -z.y); }".to_string()
        }
    };
    let _ = writeln!(s, "{guard}{{ {raw}");
    match pass.load_table {
        Table::None => {}
        Table::Line => s.push_str("v = cmul(v, ltab[n]);\n"),
        Table::Slab => s.push_str("v = cmul(v, ltab[gi - ob * p.in_os]);\n"),
    }
    s.push_str("}\n");
    s
}

/// Where the element `(r, k)` of the store phase goes and what it is multiplied by.
fn store_value(pass: &DftPass) -> String {
    let mut s = String::new();
    s.push_str("let lrow = select(l0 + r, 0u, p.low == 1u);\n");
    s.push_str("let ii = lrow % p.inner; let nat = k * p.prod + lrow / p.inner;\n");
    s.push_str("let go = out_base + r * p.out_rs + k * p.out_ds;\n");
    match pass.store_table {
        Table::None => {}
        Table::Line => s.push_str("v = cmul(v, stab[nat]);\n"),
        Table::Slab => s.push_str("v = cmul(v, stab[go - ob * p.out_os]);\n"),
    }
    s.push_str("v = v * p.scale;\n");
    let write = match pass.store {
        StoreKind::Complex => "dst[go] = v;".to_string(),
        StoreKind::Real => "dst[go] = v.x;".to_string(),
        StoreKind::RealPair => {
            "let u = go - ob * p.out_os; let i0 = 2u * ob * p.out_os + 2u * u - ii;\ndst[i0] = v.x; dst[i0 + p.inner] = v.y;".to_string()
        }
    };
    let guard = if pass.store_limit { "if (nat < p.n_out) " } else { "" };
    let _ = writeln!(s, "{guard}{{ {write} }}");
    s
}

/// `body` (which uses `e`) for every element `e` of a tile of `tile` elements that invocation
/// `lid` handles, written out one after the other so that the loads of all of them are issued
/// before any of their results is waited for.
fn unrolled(tile: u32, body: &str) -> String {
    let mut out = String::new();
    for i in 0..tile.div_ceil(WORKGROUP) {
        let guard = if (i + 1) * WORKGROUP <= tile { String::new() } else { format!("if (e < {tile}u) ") };
        let _ = writeln!(out, "{{ let e = lid + {}u; {guard}{{\n{body}\n}} }}", i * WORKGROUP);
    }
    out
}

/// The WGSL source of `pass`. Bindings in order: `src`, `dst`, then `ltab` and `stab` if the
/// pass has those tables, then the parameters.
pub(crate) fn source(pass: &DftPass) -> String {
    let d = pass.points();
    let c = pass.cols;
    let tile = c * d;
    let mut src = String::from(
        "struct Params { tiles: u32, rows: u32, j: u32, low: u32, mid: u32, in_rs: u32, in_ds: u32, in_os: u32, out_rs: u32, out_ds: u32, out_os: u32, den: u32, inner: u32, prod: u32, n_in: u32, n_out: u32, n_full: u32, scale: f32, cj: f32, mrs: u32, n_lim: u32, tpo: u32, prodh: u32, pad: u32 }\n",
    );
    let mut bindings = Vec::new();
    bindings.push(format!("var<storage, read> src: array<{}>", if pass.real_in() { "f32" } else { "vec2<f32>" }));
    bindings.push(format!("var<storage, read_write> dst: array<{}>", if pass.real_out() { "f32" } else { "vec2<f32>" }));
    if pass.load_table != Table::None {
        bindings.push("var<storage, read> ltab: array<vec2<f32>>".to_string());
    }
    if pass.store_table != Table::None {
        bindings.push("var<storage, read> stab: array<vec2<f32>>".to_string());
    }
    bindings.push("var<uniform> p: Params".to_string());
    for (i, binding) in bindings.iter().enumerate() {
        let _ = writeln!(src, "@group(1) @binding({i}) {binding};");
    }
    let _ = write!(
        src,
        "
var<workgroup> sh: array<vec2<f32>, {}>;

// Padding that spreads strided accesses over the memory banks.
fn sx(i: u32) -> u32 {{ return i + (i >> 4u); }}
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}}
// e^(-2 pi i num / den) for num < den < 2^29: the angle is reduced to an octant with integers,
// so the polynomials below see at most pi / 4.
fn tw(num: u32, den: u32) -> vec2<f32> {{
    let a = 8u * num;
    let oct = a / den;
    let rem = a - oct * den;
    let odd = (oct & 1u) == 1u;
    let x = select(rem, den - rem, odd);
    let th = (f32(x) / f32(den)) * 0.7853981633974483;
    let t2 = th * th;
    let sn = th * (1.0 + t2 * (-0.16666667 + t2 * (0.008333334 + t2 * (-0.0001984127 + t2 * 0.0000027557319))));
    let cs = 1.0 + t2 * (-0.5 + t2 * (0.041666668 + t2 * (-0.0013888889 + t2 * (0.0000248015873 + t2 * -0.00000027557319))));
    let cp = select(cs, sn, odd);
    let sp = select(sn, cs, odd);
    let quad = oct >> 1u;
    var c = cp;
    var s = sp;
    if (quad == 1u) {{ c = -sp; s = cp; }}
    if (quad == 2u) {{ c = -cp; s = -sp; }}
    if (quad == 3u) {{ c = sp; s = -cp; }}
    return vec2<f32>(c, -s);
}}
",
        tile + tile / 16 + 2
    );
    let prologue = if pass.post {
        format!("let o = tile / p.tpo; let blk = tile - o * p.tpo; let l0 = blk * {}u;", c / 2)
    } else {
        format!(
            "let rho0 = tile * {c}u;
    let o = rho0 / p.j;
    let j0 = rho0 - o * p.j;
    let m0 = j0 / p.low;
    let l0 = j0 - m0 * p.low;
    let in_base = o * p.in_os + j0;
    let out_base = o * p.out_os + m0 * {d}u * p.low + l0;"
        )
    };
    let _ = write!(
        src,
        "
@compute @workgroup_size({WORKGROUP})
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let tile = wid.x + wid.y * nwg.x;
    if (tile >= p.tiles) {{ return; }}
    {prologue}
"
    );
    // Load.
    let (r_load, d_load) = if pass.in_fast { (format!("e % {c}u"), format!("e / {c}u")) } else { (format!("e / {d}u"), format!("e % {d}u")) };
    let h2 = c / 2;
    let rows_post = format!(
        "let lo = l0 + (r % {h2}u); let hi = r >= {h2}u;
    let l = select(lo, p.low - lo, hi);
    let valid = select(lo < p.prodh, (lo >= 1u) && (lo < p.prodh) && (2u * lo != p.low), hi);"
    );
    let load_body_post = format!(
        "let r = e % {c}u; let d = e / {c}u;
    {rows_post}
    var v = vec2<f32>(0.0, 0.0);
    if (valid) {{
        let gi = o * p.in_os + l + d * p.in_ds;
        let ob = o;
        let mrow = 0u;
        {}
        v.y = v.y * p.cj;
    }}
    sh[{}] = v;",
        load_value(pass),
        at(pass, "d", "r")
    );
    let load_body = format!(
        "let r = {r_load}; let d = {d_load};
    var v = vec2<f32>(0.0, 0.0);
    if ((rho0 + r) < p.rows) {{
        let gi = in_base + r * p.in_rs + d * p.in_ds;
        let ob = o + select(0u, r, p.j == 1u);
        let mrow = m0 + r * p.mrs;
        {}
        v.y = v.y * p.cj;
    }}
    sh[{}] = v;",
        load_value(pass),
        at(pass, "d", "r")
    );
    src.push_str(&unrolled(tile, if pass.post { &load_body_post } else { &load_body }));
    src.push_str("workgroupBarrier();\n");
    for s in 0..pass.stages as usize {
        src.push_str(&stage(pass, s));
    }
    // Store, reading each position through the digit reversal.
    let (r_store, k_store) = if pass.out_fast { (format!("e % {c}u"), format!("e / {c}u")) } else { (format!("e / {d}u"), format!("e % {d}u")) };
    let radices = &pass.radices[..pass.stages as usize];
    let mut reverse = String::from("var kk = k; var pos = 0u;\n");
    let mut done = 1u32;
    for &r in radices {
        done *= u32::from(r);
        let _ = writeln!(reverse, "pos = pos + (kk % {r}u) * {}u; kk = kk / {r}u;", d / done);
    }
    let twiddle = if pass.twiddle { "let mrow = m0 + r * p.mrs; v = cmul(v, tw(k * mrow, p.den));\n" } else { "" };
    let mut reverse_kp = String::from("var kq = kp; var posp = 0u;\n");
    let mut done_p = 1u32;
    for &r in radices {
        done_p *= u32::from(r);
        let _ = writeln!(reverse_kp, "posp = posp + (kq % {r}u) * {}u; kq = kq / {r}u;", d / done_p);
    }
    let store_body_post = format!(
        "let r = e % {c}u; let k = e / {c}u;
    {rows_post}
    if (valid) {{
        {reverse}let zk = sh[{}];
        let self_row = (l == 0u) || (2u * l == p.low);
        let rp = select(select(r + {h2}u, r - {h2}u, hi), r, self_row);
        let kp = select({d}u - 1u - k, ({d}u - k) % {d}u, l == 0u);
        {reverse_kp}let z2 = sh[{}];
        let zc = vec2<f32>(z2.x, -z2.y);
        let ev = (zk + zc) * 0.5;
        let df = zk - zc;
        let od = cmul(tw(k * p.prod + l, p.den), vec2<f32>(df.y, -df.x) * 0.5);
        let nat = k * p.prod + l;
        let a = ev + od;
        let b = ev - od;
        dst[o * p.out_os + nat] = a * p.scale;
        if (nat == 0u) {{ dst[o * p.out_os + p.out_os - 1u] = vec2<f32>(b.x, -b.y) * p.scale; }}
    }}",
        at(pass, "pos", "r"),
        at(pass, "posp", "rp")
    );
    let store_body = format!(
        "let r = {r_store}; let k = {k_store};
    if ((rho0 + r) < p.rows) {{
        let ob = o + select(0u, r, p.j == 1u);
        {reverse}var v = sh[{}];
        {twiddle}v.y = v.y * p.cj;
        {}
    }}",
        at(pass, "pos", "r"),
        store_value(pass)
    );
    src.push_str(&unrolled(tile, if pass.post { &store_body_post } else { &store_body }));
    src.push_str("}\n");
    src
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass(radices: &[u8], cols: u32) -> DftPass {
        let mut r = [0u8; MAX_STAGES];
        r[..radices.len()].copy_from_slice(radices);
        DftPass {
            radices: r,
            stages: radices.len() as u8,
            cols,
            in_fast: true,
            out_fast: true,
            twiddle: true,
            load: LoadKind::Complex,
            load_table: Table::None,
            load_limit: false,
            store: StoreKind::Complex,
            store_table: Table::None,
            store_limit: false,
            post: false,
        }
    }

    /// Every combination of options is valid WGSL that the kernel frontend accepts.
    #[test]
    fn kernels_validate() {
        let variants = [
            DftPass { load: LoadKind::Real, store: StoreKind::Real, ..pass(&[16, 16, 8], 1) },
            DftPass { in_fast: false, out_fast: false, twiddle: false, ..pass(&[5, 5, 5], 4) },
            DftPass { load: LoadKind::RealPair, load_table: Table::Line, load_limit: true, store: StoreKind::RealPair, store_table: Table::Slab, store_limit: true, ..pass(&[7, 3], 8) },
            DftPass { load: LoadKind::Hermitian, load_table: Table::Slab, store_table: Table::Line, ..pass(&[31], 16) },
            pass(&[], 256),
        ];
        for p in variants {
            let wgsl = source(&p);
            luma_core::kir::parse_wgsl(&wgsl, "main", None, luma_core::Features::empty(), [256; 3], 256, 64, true).unwrap_or_else(|e| panic!("{p:?}: {e}\n{wgsl}"));
        }
    }
}
