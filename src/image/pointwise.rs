//! The image kernels that are not windowed filters: colour-space conversions of the planes of an
//! image (one invocation per pixel) and `unwrap` / `wrap` (im2col / col2im, one invocation per
//! element of the result).

use std::fmt::Write as _;

/// Invocations per workgroup of these kernels.
pub(crate) const WORKGROUP: u32 = 256;

/// A colour conversion of `[H, W, C]` planes (the planes of a channel are contiguous).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Color {
    /// Three planes to one: `a r + b g + c b`.
    RgbToGray,
    /// One plane to three: `a x`, `b x`, `c x`.
    GrayToRgb,
    RgbToHsv,
    HsvToRgb,
    /// BT.601 / BT.709 / BT.2020 (see [`ycc_coefficients`]); the coefficients are parameters.
    RgbToYcbcr,
    YcbcrToRgb,
}

impl Color {
    /// The planes of the input and of the output.
    pub(crate) fn planes(self) -> (u32, u32) {
        match self {
            Color::RgbToGray => (3, 1),
            Color::GrayToRgb => (1, 3),
            _ => (3, 3),
        }
    }
}

/// `(kr, kb)` of the luma `Y = kr R + (1 - kr - kb) G + kb B` of a standard: 0 BT.601, 1 BT.709,
/// 2 BT.2020.
pub(crate) fn ycc_coefficients(standard: u32) -> (f32, f32) {
    match standard {
        0 => (0.299, 0.114),
        1 => (0.2126, 0.0722),
        _ => (0.2627, 0.0593),
    }
}

/// The WGSL of a colour conversion. Arguments: out, src, uniform `{total, n, a, b, c}` (`total`
/// pixels over all images, `n` pixels per plane).
pub(crate) fn color_source(op: Color) -> String {
    let (cin, cout) = op.planes();
    let mut s = String::from(
        "struct Params { total: u32, n: u32, a: f32, b: f32, c: f32 }
@group(1) @binding(0) var<storage, read_write> out: array<f32>;
@group(1) @binding(1) var<storage, read> src: array<f32>;
@group(1) @binding(2) var<uniform> p: Params;
",
    );
    let _ = write!(
        s,
        "@compute @workgroup_size({WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {WORKGROUP}u;
    if (i >= p.total) {{ return; }}
    let image = i / p.n;
    let pix = i - image * p.n;
    let ib = image * {cin}u * p.n + pix;
    let ob = image * {cout}u * p.n + pix;
"
    );
    let load3 = "    let r = src[ib];\n    let g = src[ib + p.n];\n    let b = src[ib + 2u * p.n];\n";
    let store3 = |r: &str, g: &str, b: &str| format!("    out[ob] = {r};\n    out[ob + p.n] = {g};\n    out[ob + 2u * p.n] = {b};\n");
    match op {
        Color::RgbToGray => {
            s.push_str(load3);
            s.push_str("    out[ob] = p.a * r + p.b * g + p.c * b;\n");
        }
        Color::GrayToRgb => {
            s.push_str("    let x = src[ib];\n");
            s.push_str(&store3("p.a * x", "p.b * x", "p.c * x"));
        }
        Color::RgbToHsv => {
            s.push_str(load3);
            s.push_str(
                "    let hi = max(r, max(g, b));
    let lo = min(r, min(g, b));
    let d = hi - lo;
    var h = 0.0;
    if (d > 0.0) {
        if (hi == r) { h = (g - b) / d; if (h < 0.0) { h += 6.0; } }
        else if (hi == g) { h = (b - r) / d + 2.0; }
        else { h = (r - g) / d + 4.0; }
    }
    var s = 0.0;
    if (hi > 0.0) { s = d / hi; }
",
            );
            s.push_str(&store3("h / 6.0", "s", "hi"));
        }
        Color::HsvToRgb => {
            s.push_str(
                "    let h = src[ib];
    let s = src[ib + p.n];
    let v = src[ib + 2u * p.n];
    let h6 = (h - floor(h)) * 6.0;
    let sector = floor(h6);
    let f = h6 - sector;
    let pp = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    var r = v; var g = t; var b = pp;
    let k = u32(sector);
    if (k == 1u) { r = q; g = v; b = pp; }
    else if (k == 2u) { r = pp; g = v; b = t; }
    else if (k == 3u) { r = pp; g = q; b = v; }
    else if (k == 4u) { r = t; g = pp; b = v; }
    else if (k >= 5u) { r = v; g = pp; b = q; }
",
            );
            s.push_str(&store3("r", "g", "b"));
        }
        Color::RgbToYcbcr => {
            // p.a = kr, p.b = kb.
            s.push_str(load3);
            s.push_str(
                "    let yn = p.a * r + (1.0 - p.a - p.b) * g + p.b * b;
    let y = 16.0 + 219.0 * yn;
    let cb = 128.0 + 224.0 * (b - yn) / (2.0 * (1.0 - p.b));
    let cr = 128.0 + 224.0 * (r - yn) / (2.0 * (1.0 - p.a));
",
            );
            s.push_str(&store3("y", "cb", "cr"));
        }
        Color::YcbcrToRgb => {
            s.push_str(
                "    let y = src[ib];
    let cb = src[ib + p.n];
    let cr = src[ib + 2u * p.n];
    let yn = (y - 16.0) / 219.0;
    let b = yn + (cb - 128.0) * 2.0 * (1.0 - p.b) / 224.0;
    let r = yn + (cr - 128.0) * 2.0 * (1.0 - p.a) / 224.0;
    let g = (yn - p.a * r - p.b * b) / (1.0 - p.a - p.b);
",
            );
            s.push_str(&store3("r", "g", "b"));
        }
    }
    s.push_str("}\n");
    s
}

/// Words of the uniform block of `unwrap` / `wrap`.
pub(crate) const UNWRAP_WORDS: usize = 16;

/// The WGSL of `unwrap`: the windows of an image as columns (or rows) of a matrix. Arguments:
/// out, src, uniform `{total, n0, n1, w0, w1, s0, s1, p0, p1, nw0, nw1, planes, column}`; element
/// `cell + K * (window + W * plane)` of a column result (`window + W * (cell + K * plane)` of a
/// row result), `K = w0 * w1`, `W = nw0 * nw1`, is the cell `cell` (`cell % w0` along dimension 0)
/// of the window `window` (`window % nw0` along dimension 0) of the image `plane`, 0 where it
/// lies in the padding.
pub(crate) fn unwrap_source() -> String {
    format!(
        "struct Params {{ total: u32, n0: u32, n1: u32, w0: u32, w1: u32, s0: u32, s1: u32, q0: u32, q1: u32, nw0: u32, nw1: u32, planes: u32, column: u32, _a: u32, _b: u32, _c: u32 }}
@group(1) @binding(0) var<storage, read_write> out: array<f32>;
@group(1) @binding(1) var<storage, read> src: array<f32>;
@group(1) @binding(2) var<uniform> p: Params;
@compute @workgroup_size({WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {WORKGROUP}u;
    if (i >= p.total) {{ return; }}
    let k = p.w0 * p.w1;
    let wins = p.nw0 * p.nw1;
    var cell = 0u;
    var win = 0u;
    var plane = 0u;
    if (p.column != 0u) {{
        cell = i % k;
        let rest = i / k;
        win = rest % wins;
        plane = rest / wins;
    }} else {{
        win = i % wins;
        let rest = i / wins;
        cell = rest % k;
        plane = rest / k;
    }}
    let a0 = i32((win % p.nw0) * p.s0 + (cell % p.w0)) - i32(p.q0);
    let a1 = i32((win / p.nw0) * p.s1 + (cell / p.w0)) - i32(p.q1);
    var v = 0.0;
    if (a0 >= 0 && a0 < i32(p.n0) && a1 >= 0 && a1 < i32(p.n1)) {{
        v = src[plane * p.n0 * p.n1 + u32(a0) + p.n0 * u32(a1)];
    }}
    out[i] = v;
}}
"
    )
}

/// The WGSL of `wrap`: the sum of the windows of a matrix of windows back into an image, one
/// invocation per cell of the image, gathering from every window that covers it (no atomics, so
/// the sum is deterministic). Same arguments as [`unwrap_source`]; `n0`, `n1` are the dimensions
/// of the image, `total` its cells over all planes.
pub(crate) fn wrap_source() -> String {
    format!(
        "struct Params {{ total: u32, n0: u32, n1: u32, w0: u32, w1: u32, s0: u32, s1: u32, q0: u32, q1: u32, nw0: u32, nw1: u32, planes: u32, column: u32, _a: u32, _b: u32, _c: u32 }}
@group(1) @binding(0) var<storage, read_write> out: array<f32>;
@group(1) @binding(1) var<storage, read> src: array<f32>;
@group(1) @binding(2) var<uniform> p: Params;
@compute @workgroup_size({WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {WORKGROUP}u;
    if (i >= p.total) {{ return; }}
    let image = p.n0 * p.n1;
    let plane = i / image;
    let rest = i - plane * image;
    let x0 = rest % p.n0 + p.q0;
    let x1 = rest / p.n0 + p.q1;
    let k = p.w0 * p.w1;
    let wins = p.nw0 * p.nw1;
    var acc = 0.0;
    for (var c1 = 0u; c1 < p.w1; c1++) {{
        if (x1 < c1) {{ break; }}
        let t1 = x1 - c1;
        if (t1 % p.s1 != 0u) {{ continue; }}
        let v1 = t1 / p.s1;
        if (v1 >= p.nw1) {{ continue; }}
        for (var c0 = 0u; c0 < p.w0; c0++) {{
            if (x0 < c0) {{ break; }}
            let t0 = x0 - c0;
            if (t0 % p.s0 != 0u) {{ continue; }}
            let v0 = t0 / p.s0;
            if (v0 >= p.nw0) {{ continue; }}
            let cell = c0 + p.w0 * c1;
            let win = v0 + p.nw0 * v1;
            var at = 0u;
            if (p.column != 0u) {{ at = cell + k * (win + wins * plane); }} else {{ at = win + wins * (cell + k * plane); }}
            acc += src[at];
        }}
    }}
    out[i] = acc;
}}
"
    )
}
