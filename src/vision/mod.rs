//! Feature detection, matching and image analysis: the corner detectors ([`Array::fast`],
//! [`Array::harris`], ...), descriptors and matchers, template matching and the edge and region
//! operations of ArrayFire's `vision.h` and `image.h`.
//!
//! Like the filters of [`crate::images`] they take `[H, W]` images (`H` along axis 0, the fastest
//! one) and are built-in kernels that run on every backend. The image coordinates of a feature
//! follow ArrayFire: `x` is the index along axis 1 (the column) and `y` the one along axis 0 (the
//! row).
//!
//! Detectors return their points as [`Keypoints`]. The number of points depends on the data, so
//! a detector reads the count back and **synchronizes** once, like [`Array::filter`]; an image
//! without a feature is `None`, since luma has no arrays without elements.

mod fast;

// The implementation groups; each owns its file (and a directory of submodules).
mod corners;
mod descriptors;
mod edges;
mod matching;
mod regions;

// Public items of the groups.
pub use corners::*;
pub use descriptors::*;
pub use edges::*;
pub use matching::*;
pub use regions::*;

use std::panic::Location;
use std::sync::Arc;

use crate::array::{Array, Failure};
use crate::builtin::Builtin;
use crate::context::{Context, RawArg};
use crate::error::LumaError;
use crate::gpu::pager::Pin;
use crate::ops::Outcome;

/// The points a detector found, as five one-dimensional arrays of the same length (ArrayFire's
/// `features`).
///
/// `x` is the index along axis 1 of the image and `y` the index along axis 0, as in ArrayFire;
/// both are exact integers in `f32` for the detectors that work at pixel resolution. A detector
/// that has no use for `orientation` (an angle in radians) or `size` (the diameter of the
/// neighbourhood the point describes, in pixels) fills them with 0 and 1, like ArrayFire.
#[derive(Clone, Debug)]
pub struct Keypoints {
    /// Position along axis 1 (the column).
    pub x: Array<f32>,
    /// Position along axis 0 (the row).
    pub y: Array<f32>,
    /// The detector's response; larger is stronger.
    pub score: Array<f32>,
    /// The orientation in radians, 0 where the detector computes none.
    pub orientation: Array<f32>,
    /// The size of the point, 1 where the detector computes none.
    pub size: Array<f32>,
}

impl Keypoints {
    /// The number of points.
    pub fn count(&self) -> usize {
        self.x.len()
    }

    /// The first `n` points of `x`, `y` and `score` (views of larger arrays) with the constant
    /// `orientation` and `size` of a detector that computes neither. `n` is at least 1.
    pub(crate) fn plain(ctx: &Context, x: &Array<f32>, y: &Array<f32>, score: &Array<f32>, n: usize) -> Result<Keypoints, LumaError> {
        Ok(Keypoints {
            x: x.slice(0, 0..n),
            y: y.slice(0, 0..n),
            score: score.slice(0, 0..n),
            orientation: Array::<f32>::zeros_on(ctx, n)?,
            size: Array::<f32>::full_on(ctx, n, 1.0)?,
        })
    }
}

/// The error of a failed operation as the error every result carries.
#[allow(dead_code)]
pub(crate) fn poison(failure: Failure, op: &'static str, location: &'static Location<'static>) -> Arc<LumaError> {
    match failure {
        Failure::Inherited(error) => error,
        Failure::New(error) => Arc::new(LumaError::Op { op, location, source: Arc::new(error) }),
    }
}

pub(crate) fn invalid(what: impl Into<String>) -> Failure {
    LumaError::InvalidShape(what.into()).into()
}

pub(crate) fn to_u32(value: usize, what: &str) -> Result<u32, Failure> {
    u32::try_from(value).map_err(|_| invalid(format!("{what} {value} exceeds u32::MAX")))
}

/// The image of a detector: a plain `[H, W]` array.
pub(crate) fn plane(image: &Array<f32>, op: &str) -> Result<[usize; 2], Failure> {
    let d = image.shape().dims();
    if d[2] != 1 || d[3] != 1 {
        return Err(invalid(format!("{op} takes an [H, W] image, got shape {:?}", d)));
    }
    Ok([d[0], d[1]])
}

/// One kernel of this module (part of the key of its cache entry in the context).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Kernel {
    /// [`scan_counts_source`].
    ScanCounts,
    /// The FAST detector, see [`fast`].
    Fast(fast::Kernel),
    /// Harris, SUSAN, difference of Gaussians.
    Corners(corners::Kernel),
    /// Template matching, nearest neighbours, Hamming matcher.
    Matching(matching::Kernel),
    /// Canny, anisotropic diffusion, mean shift.
    Edges(edges::Kernel),
    /// Connected components, confidence connected, deconvolution.
    Regions(regions::Kernel),
    /// ORB, SIFT, GLOH.
    Descriptors(descriptors::Kernel),
}

/// The WGSL of a kernel of this module.
pub(crate) fn source(kernel: Kernel) -> String {
    match kernel {
        Kernel::ScanCounts => scan_counts_source(),
        Kernel::Fast(k) => fast::source(k),
        Kernel::Corners(k) => corners::source(k),
        Kernel::Matching(k) => matching::source(k),
        Kernel::Edges(k) => edges::source(k),
        Kernel::Regions(k) => regions::source(k),
        Kernel::Descriptors(k) => descriptors::source(k),
    }
}

/// Invocations of the workgroup of [`scan_counts_source`].
const SCAN_GROUP: u32 = 256;
/// Counts each invocation of [`scan_counts_source`] scans.
const SCAN_ITEMS: u32 = 4;

/// The WGSL of the exclusive scan of `n` counts followed by their total, in one workgroup:
/// `counts[0..n]` become the exclusive prefix sums and `counts[n]` the sum. Arguments: counts
/// (`n + 1` words), uniform `{n}`. The counts of the tiles of a detector are turned into the
/// offsets at which the tiles write their points.
fn scan_counts_source() -> String {
    let span = SCAN_GROUP * SCAN_ITEMS;
    format!(
        "struct Params {{ n: u32, a: u32, b: u32, c: u32 }}
@group(1) @binding(0) var<storage, read_write> counts: array<u32>;
@group(1) @binding(1) var<uniform> p: Params;
var<workgroup> sums: array<u32, {SCAN_GROUP}>;
@compute @workgroup_size({SCAN_GROUP})
fn main(@builtin(local_invocation_index) t: u32) {{
    var carry = 0u;
    for (var s0 = 0u; s0 < p.n; s0 += {span}u) {{
        let k0 = s0 + t * {SCAN_ITEMS}u;
        var v: array<u32, {SCAN_ITEMS}>;
        var run = 0u;
        for (var r = 0u; r < {SCAN_ITEMS}u; r++) {{
            var x = 0u;
            if (k0 + r < p.n) {{ x = counts[k0 + r]; }}
            v[r] = x;
            run += x;
        }}
        sums[t] = run;
        workgroupBarrier();
        for (var d = 1u; d < {SCAN_GROUP}u; d = d << 1u) {{
            var add = 0u;
            if (t >= d) {{ add = sums[t - d]; }}
            workgroupBarrier();
            sums[t] = sums[t] + add;
            workgroupBarrier();
        }}
        var before = carry;
        if (t > 0u) {{ before += sums[t - 1u]; }}
        for (var r = 0u; r < {SCAN_ITEMS}u; r++) {{
            if (k0 + r < p.n) {{ counts[k0 + r] = before; }}
            before += v[r];
        }}
        carry += sums[{last}u];
        workgroupBarrier();
    }}
    if (t == 0u) {{ counts[p.n] = carry; }}
}}
",
        last = SCAN_GROUP - 1
    )
}

/// Records the exclusive scan of the first `n` words of `counts` (which has `n + 1`): the total
/// ends up in the last one.
pub(crate) fn scan_counts(ctx: &Context, counts: &(Pin, u64), n: u32) -> Outcome<()> {
    let kernel = ctx.builtin(Builtin::Vision(Kernel::ScanCounts), "u32")?;
    let params = [n, 0, 0, 0];
    let args = [RawArg::write(counts.0.buffer(), counts.1), RawArg::Uniform(bytemuck::bytes_of(&params))];
    ctx.launch(&kernel, [1, 1, 1], [SCAN_GROUP, 1, 1], &args)?;
    Ok(())
}

/// The workgroups of a launch of `x` by `y` tiles; each dimension must fit the device.
pub(crate) fn tile_grid(ctx: &Context, x: u32, y: u32) -> Outcome<[u32; 3]> {
    let max = ctx.max_workgroups_per_dimension();
    if x > max || y > max {
        // Fold the tiles over x and y; kernels then number their tile as `x + y * num_workgroups.x`.
        return Err(LumaError::KernelGridTooLarge { workgroups: [x, y, 1], max }.into());
    }
    Ok([x, y, 1])
}
