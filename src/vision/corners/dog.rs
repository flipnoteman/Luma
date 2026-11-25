//! The difference of Gaussians (`Array::dog`).
//!
//! One kernel per tile of 32 x 8 outputs: it stages the image tile with a halo of the larger
//! radius (zeros outside the image), runs the first separable pass of both Gaussians (along
//! axis 0) into workgroup memory, and the second pass of both (along axis 1) combines them into
//! the output: one read of the image and one write of the result. The taps are literals of the
//! kernel (they only depend on the radius) and, being symmetric, are applied to sums of
//! the two cells they weigh. A radius whose tile does not fit workgroup memory falls back to four
//! `conv2_sep` passes and a subtraction.

use std::fmt::Write as _;
use std::panic::Location;

use super::detect::{self, Tile};
use super::Kernel as Corners;
use crate::jit::structured::opaque::opaque;
use crate::array::{Array, Failure};
use crate::builtin::Builtin;
use crate::context::RawArg;
use crate::error::LumaError;
use crate::image::{gaussian_kernel_1d_on, ConvSpec};
use crate::ops::Outcome;
use crate::vision::{to_u32, Kernel as Any};

/// One kernel: both radii and the tile are literals.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Kernel {
    pub(crate) r1: u8,
    pub(crate) r2: u8,
    pub(crate) tile: Tile,
}

/// The taps of the Gaussian of `2 r + 1` cells with the standard deviation `0.25 * (2 r + 1)`,
/// normalised (ArrayFire's kernel, as [`gaussian_kernel_1d`](crate::gaussian_kernel_1d)).
fn taps(r: usize) -> Vec<f32> {
    let n = 2 * r + 1;
    let sigma = 0.25 * n as f64;
    let line: Vec<f64> = (0..n).map(|i| (-((i as f64 - r as f64).powi(2)) / (2.0 * sigma * sigma)).exp()).collect();
    let total: f64 = line.iter().sum();
    line.iter().map(|v| (v / total) as f32).collect()
}

fn shared_bytes(radius: u32, tile: Tile) -> u64 {
    let (tx, ty) = (u32::from(tile.tx), u32::from(tile.ty));
    u64::from((tx + 2 * radius) * (ty + 2 * radius) + 2 * tx * (ty + 2 * radius)) * 4
}

/// The WGSL of a kernel. Arguments: image, result, uniform `{h, w, tiles_x, c}`.
pub(crate) fn source(k: Kernel) -> String {
    let (r1, r2) = (usize::from(k.r1), usize::from(k.r2));
    let big = r1.max(r2) as u32;
    let tile = k.tile;
    let (tx, ty, nth) = (u32::from(tile.tx), u32::from(tile.ty), tile.threads());
    let (ni0, ni1) = (tx + 2 * big, ty + 2 * big);
    let mut s = String::new();
    s.push_str("struct Params { h: u32, w: u32, tiles_x: u32, c: u32 }\n");
    s.push_str("@group(1) @binding(0) var<storage, read> src: array<f32>;\n");
    s.push_str("@group(1) @binding(1) var<storage, read_write> dst: array<f32>;\n");
    s.push_str("@group(1) @binding(2) var<uniform> p: Params;\n");
    let _ = writeln!(s, "var<workgroup> img: array<f32, {}>;", ni0 * ni1);
    let _ = writeln!(s, "var<workgroup> t1: array<f32, {}>;", tx * ni1);
    let _ = writeln!(s, "var<workgroup> t2: array<f32, {}>;", tx * ni1);
    // `sum` of the symmetric taps of radius `r` applied to `at(offset)` (offset 0 is the centre).
    let sym = |r: usize, at: &dyn Fn(i32) -> String| {
        let g = taps(r);
        let mut e = format!("{:?} * {}", g[r], at(0));
        for j in 1..=r {
            let _ = write!(e, " + {:?} * ({} + {})", g[r + j], at(-(j as i32)), at(j as i32));
        }
        e
    };
    let at_img = |d: i32| format!("img[base + {}u]", (big as i32 + d) as u32);
    let at_t = |name: &'static str| move |d: i32| format!("{name}[t + {}u]", (big as i32 + d) as u32 * tx);
    let _ = write!(
        s,
        "@compute @workgroup_size({nth})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let o0 = i32(wid.x) * {tx};
    let o1 = i32(wid.y) * {ty};
    let h = i32(p.h);
    let w = i32(p.w);
    let plane = wid.z * p.h * p.w;
    for (var i = t; i < {total}u; i += {nth}u) {{
        let a0 = o0 - {big} + i32(i % {ni0}u);
        let a1 = o1 - {big} + i32(i / {ni0}u);
        var v = 0.0;
        if (a0 >= 0 && a0 < h && a1 >= 0 && a1 < w) {{ v = src[plane + u32(a0) + p.h * u32(a1)]; }}
        img[i] = v;
    }}
    workgroupBarrier();
    for (var i = t; i < {total_t}u; i += {nth}u) {{
        let base = (i / {tx}u) * {ni0}u + (i % {tx}u);
        t1[i] = {a1};
        t2[i] = {a2};
    }}
    workgroupBarrier();
    let x0 = o0 + i32(t % {tx}u);
    let x1 = o1 + i32(t / {tx}u);
    if (x0 < h && x1 < w) {{
        dst[plane + u32(x0) + p.h * u32(x1)] = ({b1}) - ({b2});
    }}
}}
",
        total = ni0 * ni1,
        total_t = tx * ni1,
        a1 = sym(r1, &at_img),
        a2 = sym(r2, &at_img),
        b1 = sym(r1, &at_t("t1")),
        b2 = sym(r2, &at_t("t2")),
    );
    s
}

impl Array<f32> {
    /// The difference of Gaussians of every `[H, W]` plane (axes 2 and 3 are a batch;
    /// ArrayFire's `dog`): the image blurred with the Gaussian kernel of `2 * radius1 + 1`
    /// cells a side minus the image blurred with the one of `2 * radius2 + 1` cells. A kernel of
    /// `n` cells a side is [`gaussian_kernel`](crate::gaussian_kernel)'s: the product of two
    /// Gaussians of standard deviation `0.25 * n` centred on the middle, summing to 1. The
    /// cells outside the image are zeros, so the result has the shape of the input.
    ///
    /// A radius of 0 is the identity kernel, equal radii give zeros. Differences from
    /// ArrayFire: the radii are unsigned (ArrayFire takes `int` and returns garbage for a
    /// negative one), any radius is accepted (ArrayFire's CUDA backend fails above 8), and only
    /// `f32` images are taken (ArrayFire's `dog` takes every image type).
    ///
    /// One kernel computes both blurs (two separable passes each) from a single read of the image
    /// and writes the difference; the result is lazy like any other operation and a failure
    /// poisons it. A radius above 255, or one whose tile does not fit workgroup memory (about fifty
    /// on a device with 48 KiB), is computed as four [`Array::conv2_sep`] passes and a subtraction.
    ///
    /// ```
    /// use luma::Array;
    /// # fn main() -> Result<(), luma::LumaError> {
    /// let img = Array::from_slice(&[1.0f32; 25], [5, 5])?;
    /// // A constant image has no detail away from the border (the border is zero padded).
    /// let d = img.dog(1, 2).to_vec()?;
    /// assert!(d[12].abs() < 1e-6);
    /// # Ok(()) }
    /// ```
    #[track_caller]
    pub fn dog(&self, radius1: usize, radius2: usize) -> Array<f32> {
        let location = Location::caller();
        opaque!(self.context(), VISION_DOG, "dog", location);
        let result = self.run_dog(radius1, radius2);
        Array::resolve(self.context(), self.shape(), result, "dog", location)
    }

    fn run_dog(&self, radius1: usize, radius2: usize) -> Outcome<Array<f32>> {
        let ctx = self.context();
        let dims = self.shape().dims();
        let big = radius1.max(radius2);
        let planes = dims[2] * dims[3];
        let fused = u8::try_from(radius1).is_ok()
            && u8::try_from(radius2).is_ok()
            && planes <= ctx.max_workgroups_per_dimension() as usize
            && detect::plan_tile(ctx.tuning().workgroup_storage, |t| shared_bytes(big as u32, t), "").is_ok();
        if !fused {
            let blur = |radius: usize| -> Result<Array<f32>, Failure> {
                let taps = gaussian_kernel_1d_on(ctx, 2 * radius + 1, 0.0)?;
                Ok(self.conv2_sep(&taps, &taps, ConvSpec::new()))
            };
            let (first, second) = (blur(radius1)?, blur(radius2)?);
            return Ok(&first - &second);
        }
        let tile = detect::plan_tile(ctx.tuning().workgroup_storage, |t| shared_bytes(big as u32, t), "a difference of Gaussians").map_err(|_| LumaError::Unsupported("dog".into()))?;
        let (h, w) = (to_u32(dims[0], "height")?, to_u32(dims[1], "width")?);
        let (tiles_x, tiles_y) = (h.div_ceil(u32::from(tile.tx)), w.div_ceil(u32::from(tile.ty)));
        let max = ctx.max_workgroups_per_dimension();
        if tiles_x > max || tiles_y > max {
            return Err(LumaError::KernelGridTooLarge { workgroups: [tiles_x, tiles_y, planes as u32], max }.into());
        }
        let out = Array::<f32>::alloc(ctx, self.shape())?;
        let (src, src_bytes) = self.ready()?;
        let (dst, dst_bytes) = out.ready()?;
        let kernel = ctx.builtin(Builtin::Vision(Any::Corners(Corners::Dog(Kernel { r1: radius1 as u8, r2: radius2 as u8, tile }))), "f32")?;
        let params = [h, w, tiles_x, 0];
        let args = [RawArg::read(src.buffer(), src_bytes), RawArg::write(dst.buffer(), dst_bytes), RawArg::Uniform(bytemuck::bytes_of(&params))];
        ctx.launch(&kernel, [tiles_x, tiles_y, planes as u32], [h, w, planes as u32], &args)?;
        Ok(out)
    }
}
