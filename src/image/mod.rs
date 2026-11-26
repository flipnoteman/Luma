//! Image and filter operations on arrays of two or three dimensions.
//!
//! An image is an `[H, W]` array (`H` along axis 0, the fastest one in memory, as in
//! ArrayFire) or an `[H, W, C]` one with the channels in planes; the axes past the window of an
//! operation are a batch: a 2-D filter (`conv2`, `medfilt2`, `sobel`, ...) filters every
//! `[H, W]` plane of an `[H, W, N, M]` array independently, a 3-D one (`conv3`, `dilate3`, ...)
//! every `[D0, D1, D2]` volume of an `[D0, D1, D2, N]` one. Colour conversions take `[H, W, 3]`
//! images (and batches `[H, W, 3, N]`).
//!
//! The windowed filters are built-in kernels: a workgroup stages a tile of the input with its
//! halo in workgroup memory (borders applied once, there) and computes every output of the tile
//! from it, see [`window`]. They run on every backend, need arrays that fit one buffer
//! ([`LumaError::Chunked`](crate::LumaError)), and (like all operations) poison their result on
//! an error.

mod pointwise;
mod sat;
mod warp;
pub(crate) mod window;

use std::panic::Location;
use std::sync::Arc;

use crate::array::{Array, Failure};
use crate::builtin::Builtin;
use crate::context::{Context, RawArg, Scratch};
use crate::error::LumaError;
use crate::gpu::pager::Pin;
use crate::gpu_type::{Numeric, ScalarType};
use crate::jit::structured::opaque::opaque;
use crate::jit::structured::boundary::{dim, WindowAttrs, WindowParams, WindowStage};
use crate::jit::structured::families::window as window_family;
use crate::jit::structured::strategy::{first_candidate_in, note, Boundary, BoundaryAttrs, FamilyTag, Params};
use crate::ops::{fold_workgroups, Outcome};
use crate::shape::Shape;
use crate::view::PadMode;

use pointwise::Color;
pub(crate) use pointwise::{color_source, unwrap_source, wrap_source, Color as ColorKernel};
pub(crate) use sat::{source as sat_source, Pass as SatPass};
pub(crate) use window::{source as window_source, Window};
pub use warp::{transform_coordinates, transform_coordinates_on, TransformMatrix};
use window::{Edge, Op, Pixel};

/// What the cells outside an array are, for a filter whose window reaches past its border.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Border {
    /// Zeros.
    #[default]
    Zero,
    /// The nearest cell of the array.
    Edge,
    /// The array mirrored about its boundary face (`-1` reads `0`, `n` reads `n - 1`).
    Mirror,
    /// The array repeated periodically.
    Wrap,
    /// The cells are not there: a minimum, a maximum, a dilation or an erosion takes the
    /// cells that exist, nothing else. Not available for filters that sum or rank.
    Ignore,
}

impl From<PadMode> for Border {
    fn from(mode: PadMode) -> Border {
        match mode {
            PadMode::Zero => Border::Zero,
            PadMode::Edge => Border::Edge,
            PadMode::Mirror => Border::Mirror,
            PadMode::Wrap => Border::Wrap,
        }
    }
}

/// How much of the border a convolution pads with.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Padding {
    /// None: only windows that lie inside the array; the output shrinks by `k - 1`.
    Valid,
    /// `(k - 1) / 2` cells before and the rest after on every axis, so a stride 1 output has the
    /// shape of the input.
    Same,
    /// `k - 1` cells on both sides: every window that touches the array (the full convolution).
    Full,
    /// Cells before and after on axes 0, 1, 2.
    Explicit {
        /// Cells before the array.
        before: [usize; 3],
        /// Cells after the array.
        after: [usize; 3],
    },
}

/// The padding, stride and border of a convolution ([`Array::conv2`], [`Array::conv3`],
/// [`Array::conv2_sep`]). The default is [`Padding::Same`], stride 1 and a [`Border::Zero`]
/// border: the output has the shape of the input.
///
/// The output has `floor((n + before + after - k) / stride) + 1` cells along an axis of `n`
/// cells and `k` taps. Entries of `stride` and padding past the dimensions of the convolution
/// are ignored.
///
/// ```
/// use luma::{Border, ConvSpec, Padding};
/// let spec = ConvSpec::new().padding(Padding::Valid).stride([2, 2]).border(Border::Edge);
/// # let _ = spec;
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ConvSpec {
    padding: Padding,
    stride: [usize; 3],
    border: Border,
    /// A builder call was given more than three axes.
    too_many_axes: bool,
}

impl Default for ConvSpec {
    fn default() -> Self {
        ConvSpec { padding: Padding::Same, stride: [1; 3], border: Border::Zero, too_many_axes: false }
    }
}

impl ConvSpec {
    /// The default: same padding, stride 1, zero border.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the padding.
    pub fn padding(mut self, padding: Padding) -> Self {
        self.padding = padding;
        self
    }

    /// Explicit padding: `before` and `after` cells on axes 0, 1, ...
    pub fn pad<const N: usize>(mut self, before: [usize; N], after: [usize; N]) -> Self {
        let (mut b, mut a) = ([0; 3], [0; 3]);
        let n = N.min(3);
        b[..n].copy_from_slice(&before[..n]);
        a[..n].copy_from_slice(&after[..n]);
        self.too_many_axes |= N > 3;
        self.padding = Padding::Explicit { before: b, after: a };
        self
    }

    /// Sets the stride along axes 0, 1, ... (1 for the axes not given). Zero poisons the result.
    pub fn stride<const N: usize>(mut self, stride: [usize; N]) -> Self {
        self.stride = [1; 3];
        let n = N.min(3);
        self.stride[..n].copy_from_slice(&stride[..n]);
        self.too_many_axes |= N > 3;
        self
    }

    /// Sets what the cells outside the array are. [`Border::Ignore`] poisons the result.
    pub fn border(mut self, border: impl Into<Border>) -> Self {
        self.border = border.into();
        self
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for i32 {}
    impl Sealed for u32 {}
}

/// The element types of the rank and morphological filters: `f32`, `i32` and `u32`.
pub trait ImagePixel: ScalarType + sealed::Sealed {
    #[doc(hidden)]
    fn pixel() -> Pixel;
    /// The bits of the smallest value (`-inf` for floats).
    #[doc(hidden)]
    fn lowest_bits() -> u32;
    /// The bits of the largest value (`inf` for floats).
    #[doc(hidden)]
    fn highest_bits() -> u32;
    /// The elements as unsigned bin indices, for [`Array::hist_equal`].
    #[doc(hidden)]
    fn to_bins(array: &Array<Self>) -> Array<u32>;
    /// `f32` values as elements, truncating toward zero for integers.
    #[doc(hidden)]
    fn from_f32(array: &Array<f32>) -> Array<Self>;
}

impl ImagePixel for f32 {
    fn pixel() -> Pixel {
        Pixel::F32
    }
    fn lowest_bits() -> u32 {
        f32::NEG_INFINITY.to_bits()
    }
    fn highest_bits() -> u32 {
        f32::INFINITY.to_bits()
    }
    fn to_bins(array: &Array<f32>) -> Array<u32> {
        array.to_u32()
    }
    fn from_f32(array: &Array<f32>) -> Array<f32> {
        array.clone()
    }
}

impl ImagePixel for i32 {
    fn pixel() -> Pixel {
        Pixel::I32
    }
    fn lowest_bits() -> u32 {
        i32::MIN as u32
    }
    fn highest_bits() -> u32 {
        i32::MAX as u32
    }
    fn to_bins(array: &Array<i32>) -> Array<u32> {
        array.to_u32()
    }
    fn from_f32(array: &Array<f32>) -> Array<i32> {
        array.to_i32()
    }
}

impl ImagePixel for u32 {
    fn pixel() -> Pixel {
        Pixel::U32
    }
    fn lowest_bits() -> u32 {
        0
    }
    fn highest_bits() -> u32 {
        u32::MAX
    }
    fn to_bins(array: &Array<u32>) -> Array<u32> {
        array.clone()
    }
    fn from_f32(array: &Array<f32>) -> Array<u32> {
        array.to_u32()
    }
}

/// The error of a failed operation as the error every result carries.
fn poison(failure: Failure, op: &'static str, location: &'static Location<'static>) -> Arc<LumaError> {
    match failure {
        Failure::Inherited(error) => error,
        Failure::New(error) => Arc::new(LumaError::Op { op, location, source: Arc::new(error) }),
    }
}

fn invalid(what: impl Into<String>) -> Failure {
    LumaError::InvalidShape(what.into()).into()
}

/// What a windowed launch computes: the kernel, the windows' geometry and the constants of the
/// operation.
struct Spec {
    op: Op,
    /// Window extent, stride and padding before / after, along axes 0, 1, 2.
    k: [usize; 3],
    stride: [usize; 3],
    before: [usize; 3],
    after: [usize; 3],
    border: Border,
    /// Whether axis 2 is part of the window (else it is a batch axis with axis 3).
    three_d: bool,
    /// The extra constants of the kernel (see `window::Op`).
    c1: u32,
    /// The window cells of the weights / mask binding, 0 without one.
    aux_cells: usize,
}

/// The geometry of one launch.
struct Geometry {
    n: [u32; 3],
    m: [u32; 3],
    pad: [u32; 3],
    batch: u32,
}

fn to_u32(value: usize, what: &str) -> Result<u32, Failure> {
    u32::try_from(value).map_err(|_| invalid(format!("{what} {value} exceeds u32::MAX")))
}

fn geometry(shape: Shape, spec: &Spec) -> Result<Geometry, Failure> {
    let dims = shape.dims();
    let axes = if spec.three_d { 3 } else { 2 };
    let mut n = [1u32; 3];
    let mut m = [1u32; 3];
    let mut pad = [0u32; 3];
    for d in 0..axes {
        let (size, k, s) = (dims[d], spec.k[d], spec.stride[d]);
        if k == 0 {
            return Err(invalid(format!("window of 0 cells along axis {d}")));
        }
        if s == 0 {
            return Err(invalid("stride must be non-zero"));
        }
        let padded = size
            .checked_add(spec.before[d])
            .and_then(|v| v.checked_add(spec.after[d]))
            .ok_or_else(|| invalid("padding is too large"))?;
        if padded < k {
            return Err(invalid(format!("window of {k} cells is larger than the padded axis {d} of {padded}")));
        }
        n[d] = to_u32(size, "axis")?;
        m[d] = to_u32((padded - k) / s + 1, "output axis")?;
        pad[d] = to_u32(spec.before[d], "padding")?;
    }
    let batch = if spec.three_d { dims[3] } else { dims[2] * dims[3] };
    Ok(Geometry { n, m, pad, batch: to_u32(batch, "batch")? })
}

/// The kernel for `spec` on `pixel`, with the largest tile whose staged window fits the
/// device's workgroup memory: the `Window` family's selection (`window::tiled` and the median selections).
fn plan(ctx: &Context, spec: &Spec, pixel: Pixel, edge: Edge, name: &'static str, location: &'static Location<'static>) -> Result<Window, Failure> {
    let to_k = |v: usize| u16::try_from(v).map_err(|_| invalid(format!("window of {v} cells is too large")));
    let k = [to_k(spec.k[0])?, to_k(spec.k[1])?, to_k(spec.k[2])?];
    let to_s = |v: usize| u8::try_from(v).map_err(|_| invalid(format!("stride {v} is too large")));
    let stride = [to_s(spec.stride[0])?, to_s(spec.stride[1])?, to_s(spec.stride[2])?];
    let boundary = Boundary::new(
        FamilyTag::Window,
        name,
        location,
        [u32::from(k[0]), u32::from(k[1]), u32::from(k[2]), u32::from(stride[0])],
        BoundaryAttrs::Window(WindowAttrs { stage: WindowStage::Plan, op: spec.op, k, stride, three_d: spec.three_d, aux_cells: spec.aux_cells as u64, pixel, edge }),
    );
    let Some((chosen, params)) = first_candidate_in(window_family::PLAN_STRATEGIES, ctx, &boundary) else {
        return Err(LumaError::Unsupported(format!("a window of {:?} cells does not fit workgroup memory", spec.k)).into());
    };
    note(ctx, &boundary, chosen, &params, &[]);
    let Params::Window(WindowParams { window: Some(w) }) = params else { unreachable!("a window plan carries its kernel") };
    Ok(w)
}

/// The first of the tile candidates of `op` whose window fits workgroup memory (`limit` bytes) and a workgroup of 256 invocations:
/// today's `plan`, a pure function of its arguments (the `fits` of the `Window` strategies call it).
#[allow(clippy::too_many_arguments)]
pub(crate) fn first_fit(limit: u64, op: Op, k: [u16; 3], stride: [u8; 3], three_d: bool, aux_cells: u64, pixel: Pixel, edge: Edge) -> Option<Window> {
    let blocked = op == Op::Correlate && stride[1] == 1 && !three_d;
    let cells01 = usize::from(k[0]) * usize::from(k[1]);
    let candidates: &[([u8; 3], u8)] = if three_d && op != Op::BilateralColor {
        &[([32, 4, 2], 1), ([16, 4, 2], 1), ([8, 4, 2], 1), ([8, 4, 1], 1), ([4, 4, 1], 1)]
    } else if blocked {
        &[([32, 32, 1], 4), ([32, 16, 1], 2), ([32, 8, 1], 1), ([16, 8, 1], 1), ([8, 8, 1], 1)]
    } else if op == Op::Correlate || op == Op::Bilateral || (op == Op::Median && cells01 > window::MEDIAN_BLOCKED_CELLS) {
        &[([32, 8, 1], 1), ([16, 8, 1], 1), ([8, 8, 1], 1), ([8, 4, 1], 1)]
    } else {
        // Larger tiles (four outputs per invocation) cost less halo per output and keep more
        // loads in flight: 3x3 filters at 4K ran 1.5x faster than with 32 x 8 tiles.
        &[([32, 32, 1], 4), ([32, 16, 1], 2), ([32, 8, 1], 1), ([16, 8, 1], 1), ([8, 8, 1], 1), ([8, 4, 1], 1)]
    };
    for &(tile, ry) in candidates {
        let stage_aux = aux_cells * 4 <= 8192;
        let w = Window { op, pixel, k, stride, edge, tile, ry, stage_aux };
        if window::shared_bytes(&w) <= limit && window::invocations(&w) <= 256 {
            return Some(w);
        }
        if stage_aux {
            let w = Window { stage_aux: false, ..w };
            if window::shared_bytes(&w) <= limit && window::invocations(&w) <= 256 {
                return Some(w);
            }
        }
    }
    None
}

/// Records the launch of `w` and returns nothing: the results are in `outs`.
#[allow(clippy::too_many_arguments)]
fn launch_window(
    ctx: &Context,
    w: &Window,
    geo: &Geometry,
    fill: u32,
    c0: u32,
    c1: u32,
    outs: &[(&Pin, u64)],
    src: (&Pin, u64),
    aux: Option<(&Pin, u64)>,
) -> Result<(), Failure> {
    let tiles: Vec<u32> = (0..3).map(|d| geo.m[d].div_ceil(u32::from(w.tile[d]))).collect();
    let groups = tiles[0]
        .checked_mul(tiles[1])
        .and_then(|v| v.checked_mul(tiles[2]))
        .and_then(|v| v.checked_mul(geo.batch))
        .ok_or_else(|| LumaError::DimensionTooLarge { shape: [geo.m[0] as usize, geo.m[1] as usize, geo.m[2] as usize, geo.batch as usize] })?;
    let params: [u32; window::PARAM_WORDS] = [
        geo.n[0], geo.n[1], geo.n[2], geo.m[0], geo.m[1], geo.m[2], geo.pad[0], geo.pad[1], geo.pad[2], tiles[0], tiles[1], tiles[2], geo.batch, fill, c0, c1,
    ];
    let kernel = ctx.builtin(Builtin::Window(*w), w.pixel.name())?;
    let mut args: Vec<RawArg<'_>> = outs.iter().map(|(pin, bytes)| RawArg::write(pin.buffer(), *bytes)).collect();
    args.push(RawArg::read(src.0.buffer(), src.1));
    if let Some((pin, bytes)) = aux {
        args.push(RawArg::read(pin.buffer(), bytes));
    }
    args.push(RawArg::Uniform(bytemuck::bytes_of(&params)));
    ctx.launch(&kernel, fold_workgroups(ctx, groups)?, [geo.m[0], geo.m[1], geo.m[2]], &args)?;
    Ok(())
}

/// The windowed filter `spec` over `input`: one output array (two for the derivatives), each
/// of the shape of the filtered axes and the batch of `input`.
fn run_window<T: ImagePixel>(input: &Array<T>, spec: &Spec, aux: Option<&Array<impl ScalarType>>, name: &'static str, location: &'static Location<'static>) -> Outcome<Vec<Array<T>>> {
    let ctx = input.context();
    input.storage()?;
    let geo = geometry(input.shape(), spec)?;
    let identity = |ignore: bool| match spec.op {
        Op::Min | Op::MaskedMin if ignore => T::highest_bits(),
        Op::Max | Op::MaskedMax if ignore => T::lowest_bits(),
        _ => 0,
    };
    let (edge, fill) = match spec.border {
        Border::Zero => (Edge::Zero, 0),
        Border::Edge => (Edge::Clamp, 0),
        Border::Mirror => (Edge::Mirror, 0),
        Border::Wrap => (Edge::Wrap, 0),
        Border::Ignore => match spec.op {
            Op::Min | Op::Max | Op::MaskedMin | Op::MaskedMax => (Edge::Ignore, identity(true)),
            _ => return Err(invalid("Border::Ignore needs a minimum, maximum, dilation or erosion")),
        },
    };
    let c0 = match spec.op {
        Op::Min | Op::MaskedMin => T::highest_bits(),
        Op::Max | Op::MaskedMax => T::lowest_bits(),
        _ => 0,
    };
    let w = plan(ctx, spec, T::pixel(), edge, name, location)?;
    let mut shape = input.shape().dims();
    for d in 0..if spec.three_d { 3 } else { 2 } {
        shape[d] = geo.m[d] as usize;
    }
    let shape = Shape::new(shape);
    let outs = (0..spec.op.outputs()).map(|_| Array::<T>::alloc(ctx, shape)).collect::<Result<Vec<_>, _>>()?;
    let (src_pin, src_bytes) = input.ready()?;
    let aux = aux
        .map(|a| {
            if !ctx.same_as(a.context()) {
                return Err(Failure::from(LumaError::ContextMismatch));
            }
            a.ready()
        })
        .transpose()?;
    let out_pins = outs.iter().map(|o| o.ready()).collect::<Result<Vec<_>, _>>()?;
    let out_refs: Vec<(&Pin, u64)> = out_pins.iter().map(|(pin, bytes)| (pin, *bytes)).collect();
    launch_window(ctx, &w, &geo, fill, c0, spec.c1, &out_refs, (&src_pin, src_bytes), aux.as_ref().map(|(pin, bytes)| (pin, *bytes)))?;
    Ok(outs)
}

fn single<T: ImagePixel>(outs: Vec<Array<T>>) -> Array<T> {
    outs.into_iter().next().expect("one output")
}

/// `result` as an array, or a poisoned array of `shape`.
fn finish<T: ImagePixel>(input: &Array<T>, result: Outcome<Array<T>>, op: &'static str, location: &'static Location<'static>) -> Array<T> {
    Array::resolve(input.context(), input.shape(), result, op, location)
}

fn finish_pair<T: ImagePixel>(input: &Array<T>, result: Outcome<Vec<Array<T>>>, op: &'static str, location: &'static Location<'static>) -> (Array<T>, Array<T>) {
    match result {
        Ok(mut outs) => {
            let b = outs.pop().expect("two outputs");
            let a = outs.pop().expect("two outputs");
            (a, b)
        }
        Err(failure) => {
            let error = poison(failure, op, location);
            (Array::poisoned(input.context(), input.shape(), error.clone()), Array::poisoned(input.context(), input.shape(), error))
        }
    }
}

/// `value` of a window axis (zero is rejected by [`geometry`]).
fn window3(window: &[usize]) -> [usize; 3] {
    let mut k = [1; 3];
    k[..window.len()].copy_from_slice(window);
    k
}

/// The padding of an anchored window (rank filters): the cell `k / 2` is the output's own.
fn anchored(k: [usize; 3]) -> ([usize; 3], [usize; 3]) {
    let before = k.map(|k| k / 2);
    let after = [0, 1, 2].map(|d| k[d].saturating_sub(1 + before[d]));
    (before, after)
}

impl<T: ImagePixel> Array<T> {
    #[allow(clippy::too_many_arguments)]
    fn rank_filter(&self, op: Op, k: [usize; 3], three_d: bool, border: Border, mask: Option<&Array<T>>, name: &'static str, location: &'static Location<'static>) -> Array<T> {
        let (before, after) = anchored(k);
        let spec = Spec { op, k, stride: [1; 3], before, after, border, three_d, c1: 0, aux_cells: mask.map_or(0, |m| m.len()) };
        let result = (|| {
            if let Some(mask) = mask {
                let d = mask.shape().dims();
                let want = if three_d { [k[0], k[1], k[2], 1] } else { [k[0], k[1], 1, 1] };
                if d != want {
                    return Err(invalid(format!("mask of shape {d:?} does not have the window's shape {want:?}")));
                }
            }
            Ok(single(run_window(self, &spec, mask, name, location)?))
        })();
        finish(self, result, name, location)
    }

    /// The median filter along axis 0 with a window of `len` cells (the cell `len / 2` is the
    /// output's own; axes 1 to 3 are a batch). Cells outside the array are given by `border`
    /// ([`Border::Ignore`] poisons the result). For an even window the median is the cell of
    /// rank `len / 2` of the sorted window. NaN sorts above every number.
    #[track_caller]
    pub fn medfilt1(&self, len: usize, border: impl Into<Border>) -> Array<T> {
        self.rank_filter(Op::Median, [len, 1, 1], false, border.into(), None, "medfilt1", Location::caller())
    }

    /// The median filter over a window of `[axis 0, axis 1]` cells (axes 2 and 3 are a batch),
    /// see [`Array::medfilt1`]. Windows of up to 49 cells run in registers (a 19-exchange network
    /// for 3 x 3), larger ones select bit by bit.
    #[track_caller]
    pub fn medfilt2(&self, window: [usize; 2], border: impl Into<Border>) -> Array<T> {
        self.rank_filter(Op::Median, window3(&window), false, border.into(), None, "medfilt2", Location::caller())
    }

    /// The minimum over a window of `[axis 0, axis 1]` cells (a rectangular erosion); axes 2 and
    /// 3 are a batch. The cell `window / 2` is the output's own. Cells outside the array are
    /// given by `border`; with [`Border::Ignore`] the minimum is over the cells that exist (what
    /// ArrayFire's `minfilt` does).
    ///
    /// A rectangle is separable: windows of 25 cells and more run as two passes over axis 0 and
    /// axis 1.
    #[track_caller]
    pub fn minfilt(&self, window: [usize; 2], border: impl Into<Border>) -> Array<T> {
        self.extremum(Op::Min, window, border.into(), Location::caller(), "minfilt")
    }

    /// The maximum over a window, see [`Array::minfilt`].
    #[track_caller]
    pub fn maxfilt(&self, window: [usize; 2], border: impl Into<Border>) -> Array<T> {
        self.extremum(Op::Max, window, border.into(), Location::caller(), "maxfilt")
    }

    fn extremum(&self, op: Op, window: [usize; 2], border: Border, location: &'static Location<'static>, name: &'static str) -> Array<T> {
        let k = window3(&window);
        let saturate = |v: usize| u16::try_from(v).unwrap_or(u16::MAX);
        let boundary = Boundary::new(
            FamilyTag::Window,
            name,
            location,
            [dim(k[0]), dim(k[1]), dim(k[2]), 1],
            BoundaryAttrs::Window(WindowAttrs {
                stage: WindowStage::Entry,
                op,
                k: [saturate(k[0]), saturate(k[1]), saturate(k[2])],
                stride: [1; 3],
                three_d: false,
                aux_cells: 0,
                pixel: T::pixel(),
                edge: Edge::Clamp,
            }),
        );
        if let Some((chosen, params)) = first_candidate_in(window_family::ENTRY_STRATEGIES, self.context(), &boundary) {
            note(self.context(), &boundary, chosen, &params, &[]);
            let pass = |input: &Array<T>, k: [usize; 3]| -> Outcome<Array<T>> {
                let (before, after) = anchored(k);
                let spec = Spec { op, k, stride: [1; 3], before, after, border, three_d: false, c1: 0, aux_cells: 0 };
                Ok(single(run_window(input, &spec, None::<&Array<f32>>, name, location)?))
            };
            let result = pass(self, [k[0], 1, 1]).and_then(|rows| pass(&rows, [1, k[1], 1]));
            return finish(self, result, name, location);
        }
        self.rank_filter(op, k, false, border, None, name, location)
    }

    /// The minimum over a window of `[axis 0, axis 1, axis 2]` cells; axis 3 is a batch. See
    /// [`Array::minfilt`].
    #[track_caller]
    pub fn minfilt3(&self, window: [usize; 3], border: impl Into<Border>) -> Array<T> {
        self.rank_filter(Op::Min, window, true, border.into(), None, "minfilt3", Location::caller())
    }

    /// The maximum over a window of `[axis 0, axis 1, axis 2]` cells; axis 3 is a batch. See
    /// [`Array::minfilt`].
    #[track_caller]
    pub fn maxfilt3(&self, window: [usize; 3], border: impl Into<Border>) -> Array<T> {
        self.rank_filter(Op::Max, window, true, border.into(), None, "maxfilt3", Location::caller())
    }

    /// The grey-scale dilation by a structuring element: the maximum of the cells of the window
    /// where `mask` is not 0 (`mask` is 2-D, its axes are the window's axes 0 and 1, the cell
    /// `mask_shape / 2` is the output's own; axes 2 and 3 of this array are a batch). Cells
    /// outside the array do not count (see [`Border::Ignore`]); an all-zero mask gives the
    /// lowest value of the type.
    ///
    /// `mask` is not flipped: the output at `x` is `max { in[x + j - mask_shape / 2] | mask[j] != 0 }`.
    #[track_caller]
    pub fn dilate(&self, mask: &Array<T>) -> Array<T> {
        self.dilate_with(mask, Border::Ignore)
    }

    /// [`Array::dilate`] with the cells outside the array given by `border`.
    #[track_caller]
    pub fn dilate_with(&self, mask: &Array<T>, border: impl Into<Border>) -> Array<T> {
        let k = window3(&mask.shape().dims()[..2]);
        self.rank_filter(Op::MaskedMax, k, false, border.into(), Some(mask), "dilate", Location::caller())
    }

    /// The grey-scale erosion by a structuring element: the minimum of the cells of the window
    /// where `mask` is not 0; see [`Array::dilate`].
    #[track_caller]
    pub fn erode(&self, mask: &Array<T>) -> Array<T> {
        self.erode_with(mask, Border::Ignore)
    }

    /// [`Array::erode`] with the cells outside the array given by `border`.
    #[track_caller]
    pub fn erode_with(&self, mask: &Array<T>, border: impl Into<Border>) -> Array<T> {
        let k = window3(&mask.shape().dims()[..2]);
        self.rank_filter(Op::MaskedMin, k, false, border.into(), Some(mask), "erode", Location::caller())
    }

    /// The 3-D dilation: `mask` is a 3-D structuring element and axis 3 is a batch; see
    /// [`Array::dilate`].
    #[track_caller]
    pub fn dilate3(&self, mask: &Array<T>) -> Array<T> {
        let k = window3(&mask.shape().dims()[..3]);
        self.rank_filter(Op::MaskedMax, k, true, Border::Ignore, Some(mask), "dilate3", Location::caller())
    }

    /// The 3-D erosion; see [`Array::dilate3`].
    #[track_caller]
    pub fn erode3(&self, mask: &Array<T>) -> Array<T> {
        let k = window3(&mask.shape().dims()[..3]);
        self.rank_filter(Op::MaskedMin, k, true, Border::Ignore, Some(mask), "erode3", Location::caller())
    }

    /// Histogram equalisation: every element `x` is replaced by the normalised cumulative
    /// distribution of `hist` at the bin `x` (truncated to an integer, clamped to the bins),
    /// scaled to `0 ..= bins - 1`: `(cdf[x] - cdf.min) * (bins - 1) / (cdf.max - cdf.min)`, `bins`
    /// being the length of `hist` (ArrayFire's `histEqual`). `hist` is the histogram of the values over
    /// the bins `0 .. bins`, for example `img.histogram(256, 0.0, 256.0)` of an image of values in
    /// `0 .. 256`. Integer elements are truncated back from the `f32` result.
    ///
    /// Every reduction stays on the device; the whole thing is lazy.
    #[track_caller]
    pub fn hist_equal(&self, hist: &Array<u32>) -> Array<T> {
        let location = Location::caller();
        let ctx = self.context();
        opaque!(ctx, IMAGE_HIST_EQUAL, "hist_equal", location);
        if hist.is_empty() || !ctx.same_as(hist.context()) {
            let error = if hist.is_empty() { LumaError::InvalidShape("empty histogram".into()) } else { LumaError::ContextMismatch };
            return Array::poisoned(ctx, self.shape(), Arc::new(LumaError::Op { op: "hist_equal", location, source: Arc::new(error) }));
        }
        let cdf = hist.to_f32().reshape(hist.len()).cumsum(0);
        let (lo, hi) = (cdf.min(), cdf.max());
        let factor = (hist.len() as f32 - 1.0) / (&hi - &lo);
        let norm = (&cdf - &lo) * &factor;
        T::from_f32(&norm.take(&T::to_bins(self)).reshape(self.shape()))
    }
}

impl<T: Numeric + ImagePixel> Array<T> {
    /// The summed-area table: the inclusive running sum along axis 0 and then along axis 1 (axes
    /// 2 and 3 are a batch), so that element `(i, j)` is the sum of the rectangle
    /// `[0, i] x [0, j]`. The sum of any rectangle is then four lookups.
    ///
    /// Three launches read and write the image twice: axis 1 is scanned in chunks of 64 cells,
    /// one invocation each (neighbouring invocations on neighbouring lines, coalesced), their
    /// sums are turned into carries, and a workgroup per column scans axis 0 in workgroup memory
    /// adding the carry. `f32` sums are accumulated in a different order than a scan along each
    /// axis would; integers wrap on overflow.
    #[track_caller]
    pub fn sat(&self) -> Array<T> {
        let location = Location::caller();
        opaque!(self.context(), IMAGE_SAT, "sat", location);
        match run_sat(self) {
            Ok(table) => table,
            // An array over several buffers: two scans.
            Err(Failure::New(LumaError::Chunked(_))) => self.cumsum(0).cumsum(1),
            Err(failure) => Array::resolve(self.context(), self.shape(), Err(failure), "sat", location),
        }
    }
}

/// The summed-area table of `input` in one buffer.
fn run_sat<T: ImagePixel>(input: &Array<T>) -> Outcome<Array<T>> {
    let ctx = input.context();
    let dims = input.shape().dims();
    let (n0, n1, planes) = (to_u32(dims[0], "axis")?, to_u32(dims[1], "axis")?, to_u32(dims[2] * dims[3], "batch")?);
    let chunks = n1.div_ceil(sat::CHUNK);
    let too_large = || LumaError::DimensionTooLarge { shape: dims };
    let sums_len = n0.checked_mul(chunks).and_then(|v| v.checked_mul(planes)).ok_or_else(too_large)?;
    let lines = n1.checked_mul(planes).ok_or_else(too_large)?;
    let out = Array::<T>::alloc(ctx, input.shape())?;
    let (src_pin, src_bytes) = input.ready()?;
    let (out_pin, out_bytes) = out.ready()?;
    let mut scratch = Scratch::new(ctx);
    let sums_bytes = u64::from(sums_len) * 4;
    let sums = scratch.take(sums_bytes)?;
    let ty = T::pixel().name();
    let params = |total: u32| -> [u32; sat::PARAM_WORDS] { [n0, n1, planes, chunks, total, 0, 0, 0] };
    let groups = |n: u32| fold_workgroups(ctx, n.div_ceil(sat::GROUP));
    // 1: scan the chunks.
    let (uniform, kernel) = (params(sums_len), ctx.builtin(Builtin::Sat(SatPass::Chunks), ty)?);
    let args = [
        RawArg::write(out_pin.buffer(), out_bytes),
        RawArg::read(src_pin.buffer(), src_bytes),
        RawArg::write(&sums, sums_bytes),
        RawArg::Uniform(bytemuck::bytes_of(&uniform)),
    ];
    ctx.launch(&kernel, groups(sums_len)?, [sums_len, 1, 1], &args)?;
    // 2: the carries.
    let (uniform, kernel) = (params(n0 * planes), ctx.builtin(Builtin::Sat(SatPass::Carry), ty)?);
    let args = [RawArg::write(&sums, sums_bytes), RawArg::Uniform(bytemuck::bytes_of(&uniform))];
    ctx.launch(&kernel, groups(n0 * planes)?, [n0 * planes, 1, 1], &args)?;
    // 3: the lines along axis 0.
    let (uniform, kernel) = (params(lines), ctx.builtin(Builtin::Sat(SatPass::Lines), ty)?);
    let args = [RawArg::write(out_pin.buffer(), out_bytes), RawArg::read(&sums, sums_bytes), RawArg::Uniform(bytemuck::bytes_of(&uniform))];
    ctx.launch(&kernel, fold_workgroups(ctx, lines)?, [lines, 1, 1], &args)?;
    Ok(out)
}

impl Array<f32> {
    fn convolve(&self, kernel: &Array<f32>, spec: &ConvSpec, three_d: bool, name: &'static str, location: &'static Location<'static>) -> Array<f32> {
        let result = (|| {
            kernel.storage()?;
            if spec.too_many_axes {
                return Err(invalid("a convolution has at most 3 axes"));
            }
            if spec.border == Border::Ignore {
                return Err(invalid("Border::Ignore needs a minimum, maximum, dilation or erosion"));
            }
            let kd = kernel.shape().dims();
            let ok = if three_d { kd[3] == 1 } else { kd[2] == 1 && kd[3] == 1 };
            if !ok {
                return Err(invalid(format!("a {}-D kernel needs the shape of its window, got {kd:?}", if three_d { 3 } else { 2 })));
            }
            let axes = if three_d { 3 } else { 2 };
            let k = [kd[0], kd[1], if three_d { kd[2] } else { 1 }];
            let (before, after) = padding_of(spec.padding, k, axes);
            let mut stride = spec.stride;
            if !three_d {
                stride[2] = 1;
            }
            let run = Spec { op: Op::Correlate, k, stride, before, after, border: spec.border, three_d, c1: 0, aux_cells: kernel.len() };
            Ok(single(run_window(self, &run, Some(kernel), name, location)?))
        })();
        finish(self, result, name, location)
    }

    /// The 2-D cross-correlation of every `[H, W]` plane (axes 0 and 1; axes 2 and 3 are a
    /// batch) with a 2-D `kernel`: `out[x] = sum_j in[x * stride + j - before] * kernel[j]`, with
    /// the padding, stride and border of `spec` (the default: the output has the shape of the
    /// input, zero border). Like [`Array::conv1d`] this does not flip the kernel; flip it
    /// (`kernel.flip(0).flip(1)`) for a true convolution, as ArrayFire's `convolve2`
    /// computes.
    ///
    /// The window of the output cell is staged in workgroup memory together with its tile's
    /// halo, and every invocation computes four cells one above the other from it.
    ///
    /// ```
    /// use luma::{Array, ConvSpec};
    /// # fn main() -> Result<(), luma::LumaError> {
    /// let img = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3])?;
    /// let k = Array::from_slice(&[0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [3, 3])?;
    /// assert_eq!(img.conv2(&k, ConvSpec::new()).to_vec()?, img.to_vec()?);
    /// # Ok(()) }
    /// ```
    #[track_caller]
    pub fn conv2(&self, kernel: &Array<f32>, spec: ConvSpec) -> Array<f32> {
        self.convolve(kernel, &spec, false, "conv2", Location::caller())
    }

    /// The 3-D cross-correlation of every `[D0, D1, D2]` volume (axis 3 is a batch) with a 3-D
    /// kernel; see [`Array::conv2`].
    #[track_caller]
    pub fn conv3(&self, kernel: &Array<f32>, spec: ConvSpec) -> Array<f32> {
        self.convolve(kernel, &spec, true, "conv3", Location::caller())
    }

    /// The separable 2-D cross-correlation: `along0` (a 1-D kernel along axis 0) then `along1`
    /// (along axis 1), the same as `conv2` with the outer product of the two but `k0 + k1` taps
    /// per output instead of `k0 * k1`. The padding, stride and border of `spec` apply per axis.
    #[track_caller]
    pub fn conv2_sep(&self, along0: &Array<f32>, along1: &Array<f32>, spec: ConvSpec) -> Array<f32> {
        let location = Location::caller();
        let result = (|| {
            along0.storage()?;
            along1.storage()?;
            if spec.too_many_axes {
                return Err(invalid("a convolution has at most 3 axes"));
            }
            if spec.border == Border::Ignore {
                return Err(invalid("Border::Ignore needs a minimum, maximum, dilation or erosion"));
            }
            for (name, k) in [("along0", along0), ("along1", along1)] {
                if k.shape().dims()[1..] != [1, 1, 1] {
                    return Err(invalid(format!("{name} must be a 1-D kernel, got {:?}", k.shape().dims())));
                }
            }
            let k = [along0.len(), along1.len(), 1];
            let (before, after) = padding_of(spec.padding, k, 2);
            let pass = |input: &Array<f32>, taps: &Array<f32>, axis: usize| {
                let mut kk = [1; 3];
                kk[axis] = k[axis];
                let (mut b, mut a, mut s) = ([0; 3], [0; 3], [1; 3]);
                b[axis] = before[axis];
                a[axis] = after[axis];
                s[axis] = spec.stride[axis];
                let run = Spec { op: Op::Correlate, k: kk, stride: s, before: b, after: a, border: spec.border, three_d: false, c1: 0, aux_cells: taps.len() };
                run_window(input, &run, Some(taps), "conv2_sep", location).map(single)
            };
            let rows = pass(self, along0, 0)?;
            pass(&rows, along1, 1)
        })();
        finish(self, result, "conv2_sep", location)
    }

    /// The Sobel derivatives `(d0, d1)` of every `[H, W]` plane (axes 2 and 3 are a batch):
    /// the 3 x 3 Sobel operator along axis 0 and axis 1 (`d0 = (in[i + 1] - in[i - 1])` smoothed
    /// by `[1, 2, 1]` along axis 1, and the same transposed). In ArrayFire's order `(dx, dy)`
    /// with `dx` the derivative along axis 0. The cells outside the array are the nearest ones
    /// ([`Border::Edge`]); see [`Array::sobel_with`].
    ///
    /// One kernel computes both from a single read of the image.
    #[track_caller]
    pub fn sobel(&self) -> (Array<f32>, Array<f32>) {
        self.sobel_run(Op::Sobel, Border::Edge, "sobel", Location::caller())
    }

    /// [`Array::sobel`] with the cells outside the array given by `border`
    /// ([`Border::Ignore`] poisons the result).
    #[track_caller]
    pub fn sobel_with(&self, border: impl Into<Border>) -> (Array<f32>, Array<f32>) {
        self.sobel_run(Op::Sobel, border.into(), "sobel_with", Location::caller())
    }

    /// The magnitude `sqrt(d0^2 + d1^2)` of the Sobel gradient ([`Array::sobel`]) in one kernel.
    #[track_caller]
    pub fn sobel_magnitude(&self) -> Array<f32> {
        let location = Location::caller();
        let spec = self.sobel_spec(Op::SobelMagnitude, Border::Edge);
        finish(self, run_window(self, &spec, None::<&Array<f32>>, "sobel_magnitude", location).map(single), "sobel_magnitude", location)
    }

    /// The cheap magnitude `|d0| + |d1|` of the Sobel gradient.
    #[track_caller]
    pub fn sobel_magnitude_fast(&self) -> Array<f32> {
        let location = Location::caller();
        let spec = self.sobel_spec(Op::SobelFast, Border::Edge);
        finish(self, run_window(self, &spec, None::<&Array<f32>>, "sobel_magnitude_fast", location).map(single), "sobel_magnitude_fast", location)
    }

    fn sobel_spec(&self, op: Op, border: Border) -> Spec {
        Spec { op, k: [3, 3, 1], stride: [1; 3], before: [1, 1, 0], after: [1, 1, 0], border, three_d: false, c1: 0, aux_cells: 0 }
    }

    fn sobel_run(&self, op: Op, border: Border, name: &'static str, location: &'static Location<'static>) -> (Array<f32>, Array<f32>) {
        let spec = self.sobel_spec(op, border);
        finish_pair(self, run_window(self, &spec, None::<&Array<f32>>, name, location), name, location)
    }

    /// The gradient `(d0, d1)` along axis 0 and axis 1 of every `[H, W]` plane (axes 2 and 3 are
    /// a batch): central differences `(in[i + 1] - in[i - 1]) / 2` inside, one-sided
    /// differences `in[1] - in[0]` at the borders (as ArrayFire's `gradient` and NumPy's
    /// `gradient`). An axis of one cell has gradient 0.
    #[track_caller]
    pub fn grad(&self) -> (Array<f32>, Array<f32>) {
        let spec = Spec { op: Op::Gradient, k: [3, 3, 1], stride: [1; 3], before: [1, 1, 0], after: [1, 1, 0], border: Border::Edge, three_d: false, c1: 0, aux_cells: 0 };
        let location = Location::caller();
        finish_pair(self, run_window(self, &spec, None::<&Array<f32>>, "grad", location), "grad", location)
    }

    /// The bilateral filter of every `[H, W]` plane (axes 2 and 3 are a batch): every cell
    /// becomes the mean of the window of `2 r + 1` cells a side, `r = max(floor(1.5 *
    /// spatial_sigma), 1)`, weighted by `exp(-(dx^2 + dy^2) / (2 spatial_sigma^2) - (v - c)^2 /
    /// (2 range_sigma^2))` for a cell of value `v` at offset `(dx, dy)` of the centre `c`. The
    /// cells outside the image are the nearest ones. Poisons the result for a non-positive
    /// sigma.
    #[track_caller]
    pub fn bilateral(&self, spatial_sigma: f32, range_sigma: f32) -> Array<f32> {
        let location = Location::caller();
        let result = (|| {
            check_sigma(spatial_sigma, range_sigma)?;
            let r = ((spatial_sigma * 1.5) as usize).max(1);
            let spec = Spec {
                op: Op::Bilateral,
                k: [2 * r + 1, 2 * r + 1, 1],
                stride: [1; 3],
                before: [r, r, 0],
                after: [r, r, 0],
                border: Border::Edge,
                three_d: false,
                c1: (-1.0 / (2.0 * range_sigma * range_sigma)).to_bits(),
                aux_cells: 0,
            };
            bilateral_run(self, &spec, spatial_sigma, "bilateral", location)
        })();
        finish(self, result, "bilateral", location)
    }

    /// The bilateral filter of `[H, W, 3]` colour images (axis 3 is a batch): as
    /// [`Array::bilateral`] with the range distance `(v - c)^2` taken as the squared distance of
    /// the three channels. Poisons the result for another channel count.
    #[track_caller]
    pub fn bilateral_color(&self, spatial_sigma: f32, range_sigma: f32) -> Array<f32> {
        let location = Location::caller();
        let result = (|| {
            check_sigma(spatial_sigma, range_sigma)?;
            if self.shape().dims()[2] != 3 {
                return Err(invalid(format!("a colour image has 3 channels along axis 2, got {:?}", self.shape().dims())));
            }
            let r = ((spatial_sigma * 1.5) as usize).max(1);
            let spec = Spec {
                op: Op::BilateralColor,
                k: [2 * r + 1, 2 * r + 1, 3],
                stride: [1; 3],
                before: [r, r, 0],
                after: [r, r, 0],
                border: Border::Edge,
                three_d: true,
                c1: (-1.0 / (2.0 * range_sigma * range_sigma)).to_bits(),
                aux_cells: 0,
            };
            bilateral_run(self, &spec, spatial_sigma, "bilateral_color", location)
        })();
        finish(self, result, "bilateral_color", location)
    }

    /// The raw image moments of every `[H, W]` plane (axes 2 and 3 are a batch), as a
    /// `[4, 1, N, M]` array: `M00 = sum f`, `M01 = sum f*i`, `M10 = sum f*j`, `M11 = sum f*i*j`
    /// with `i` the index along axis 0 and `j` the one along axis 1, counted from 0
    /// (ArrayFire's `moments` with `FIRST_ORDER`, in this order). Lazy; the four sums are
    /// reductions over the image.
    #[track_caller]
    pub fn moments(&self) -> Array<f32> {
        let location = Location::caller();
        let ctx = self.context();
        opaque!(ctx, IMAGE_MOMENTS, "moments", location);
        let result = (|| {
            self.storage()?;
            let shape = self.shape();
            let i = Array::<u32>::range_on(ctx, shape, 0)?.to_f32();
            let j = Array::<u32>::range_on(ctx, shape, 1)?.to_f32();
            let sum = |a: &Array<f32>| a.sum_axis(0).sum_axis(1);
            let (m00, m01, m10, m11) = (sum(self), sum(&(self * &i)), sum(&(self * &j)), sum(&(&(self * &i) * &j)));
            Ok(crate::concat(0, [&m00, &m01, &m10, &m11]))
        })();
        Array::resolve(ctx, Shape::new([4, 1, self.shape().dims()[2], self.shape().dims()[3]]), result, "moments", location)
    }

    /// The windows of every `[H, W]` plane (axes 2 and 3 are a batch) as the columns (`column`)
    /// or rows of a matrix, for convolutions as matrix products (im2col): windows of
    /// `window = [axis 0, axis 1]` cells every `stride` cells, the image padded with `pad` zero
    /// cells on every side. The result is `[window0 * window1, windows, N, M]` for columns and
    /// `[windows, window0 * window1, N, M]` for rows, `windows = windows0 * windows1`,
    /// `windows_d = (n_d + 2 pad_d - window_d) / stride_d + 1`; window `a + windows0 * b` starts
    /// at `(a * stride_0 - pad_0, b * stride_1 - pad_1)` and holds its cell `(c, d)` at `c +
    /// window0 * d` (ArrayFire's `unwrap`).
    #[track_caller]
    pub fn unwrap(&self, window: [usize; 2], stride: [usize; 2], pad: [usize; 2], column: bool) -> Array<f32> {
        let location = Location::caller();
        let ctx = self.context();
        opaque!(ctx, IMAGE_UNWRAP, "unwrap", location);
        let mut dims = self.shape().dims();
        let result = (|| {
            self.storage()?;
            let g = unwrap_geometry([dims[0], dims[1]], window, stride, pad)?;
            let (k, wins) = (window[0] * window[1], g[0] * g[1]);
            let planes = dims[2] * dims[3];
            let out_dims = if column { [k, wins, dims[2], dims[3]] } else { [wins, k, dims[2], dims[3]] };
            let shape = Shape::new(out_dims);
            let out = Array::<f32>::alloc(ctx, shape)?;
            let params = unwrap_params(shape.len(), [dims[0], dims[1]], window, stride, pad, g, planes, column)?;
            launch_pointwise(ctx, Builtin::Unwrap, shape.len(), &params, &out, self)?;
            Ok(out)
        })();
        dims[0] = 1;
        Array::resolve(ctx, Shape::new(dims), result, "unwrap", location)
    }

    /// The inverse of [`Array::unwrap`] that sums overlapping windows (col2im, ArrayFire's
    /// `wrap`): `self` holds the windows of an `out_dims = [axis 0, axis 1]` image as columns
    /// (`column`) or rows (see [`Array::unwrap`]) and the result is the `[out0, out1, N, M]`
    /// image in which every cell is the sum of the window cells that cover it (the padding is
    /// dropped). Every output cell gathers its window cells, so there are no atomics and the sum
    /// is the same on every run.
    #[track_caller]
    pub fn wrap(&self, out_dims: [usize; 2], window: [usize; 2], stride: [usize; 2], pad: [usize; 2], column: bool) -> Array<f32> {
        let location = Location::caller();
        let ctx = self.context();
        opaque!(ctx, IMAGE_WRAP, "wrap", location);
        let dims = self.shape().dims();
        let result = (|| {
            self.storage()?;
            let g = unwrap_geometry(out_dims, window, stride, pad)?;
            let (k, wins) = (window[0] * window[1], g[0] * g[1]);
            let want = if column { [k, wins] } else { [wins, k] };
            if dims[0] != want[0] || dims[1] != want[1] {
                return Err(invalid(format!("windows of shape {:?} do not match the {:?} that {window:?} windows of a {out_dims:?} image give", [dims[0], dims[1]], want)));
            }
            let planes = dims[2] * dims[3];
            let shape = Shape::new([out_dims[0], out_dims[1], dims[2], dims[3]]);
            let out = Array::<f32>::alloc(ctx, shape)?;
            let params = unwrap_params(shape.len(), out_dims, window, stride, pad, g, planes, column)?;
            launch_pointwise(ctx, Builtin::Wrap, shape.len(), &params, &out, self)?;
            Ok(out)
        })();
        Array::resolve(ctx, Shape::new([out_dims[0], out_dims[1], dims[2], dims[3]]), result, "wrap", location)
    }

    fn color(&self, op: Color, params: [f32; 3], standard: u32, name: &'static str, location: &'static Location<'static>) -> Array<f32> {
        let ctx = self.context();
        opaque!(ctx, IMAGE_POINTWISE, name, location);
        let (cin, cout) = op.planes();
        let mut dims = self.shape().dims();
        let result = (|| {
            self.storage()?;
            if dims[2] != cin as usize {
                return Err(invalid(format!("{name} needs {cin} channels along axis 2, got an array of shape {dims:?}")));
            }
            let pixels = dims[0] * dims[1];
            let images = dims[3];
            let mut out_dims = dims;
            out_dims[2] = cout as usize;
            let shape = Shape::new(out_dims);
            let out = Array::<f32>::alloc(ctx, shape)?;
            let total = to_u32(pixels * images, "pixel count")?;
            let (a, b) = pointwise::ycc_coefficients(standard);
            let [pa, pb, pc] = if matches!(op, Color::RgbToYcbcr | Color::YcbcrToRgb) { [a, b, 0.0] } else { params };
            let words = [total, to_u32(pixels, "plane")?, pa.to_bits(), pb.to_bits(), pc.to_bits()];
            launch_pointwise(ctx, Builtin::Color(op), total as usize, &words, &out, self)?;
            Ok(out)
        })();
        dims[2] = cout as usize;
        Array::resolve(ctx, Shape::new(dims), result, name, location)
    }

    /// The grey value `0.2126 r + 0.7152 g + 0.0722 b` of `[H, W, 3]` images (batch axis 3), as
    /// `[H, W, 1, N]`; see [`Array::rgb2gray_with`].
    #[track_caller]
    pub fn rgb2gray(&self) -> Array<f32> {
        self.color(Color::RgbToGray, [0.2126, 0.7152, 0.0722], 0, "rgb2gray", Location::caller())
    }

    /// The grey value `r_weight * r + g_weight * g + b_weight * b` of `[H, W, 3]` images.
    #[track_caller]
    pub fn rgb2gray_with(&self, r_weight: f32, g_weight: f32, b_weight: f32) -> Array<f32> {
        self.color(Color::RgbToGray, [r_weight, g_weight, b_weight], 0, "rgb2gray", Location::caller())
    }

    /// Three planes `r_factor * x`, `g_factor * x`, `b_factor * x` of `[H, W, 1, N]` grey images
    /// (`[H, W, 3, N]`).
    #[track_caller]
    pub fn gray2rgb(&self, r_factor: f32, g_factor: f32, b_factor: f32) -> Array<f32> {
        self.color(Color::GrayToRgb, [r_factor, g_factor, b_factor], 0, "gray2rgb", Location::caller())
    }

    /// RGB (planes `r, g, b` in `0 ..= 1`) to HSV: hue in `0 .. 1` (a full turn is 1), saturation
    /// and value in `0 ..= 1`.
    #[track_caller]
    pub fn rgb2hsv(&self) -> Array<f32> {
        self.color(Color::RgbToHsv, [0.0; 3], 0, "rgb2hsv", Location::caller())
    }

    /// HSV (hue `0 .. 1`, saturation, value) to RGB.
    #[track_caller]
    pub fn hsv2rgb(&self) -> Array<f32> {
        self.color(Color::HsvToRgb, [0.0; 3], 0, "hsv2rgb", Location::caller())
    }

    /// RGB (`0 ..= 1`) to Y'CbCr of the standard `standard`, in the 8-bit studio range (Y 16 to
    /// 235, Cb and Cr 16 to 240) as `f32`.
    #[track_caller]
    pub fn rgb2ycbcr(&self, standard: YccStd) -> Array<f32> {
        self.color(Color::RgbToYcbcr, [0.0; 3], standard as u32, "rgb2ycbcr", Location::caller())
    }

    /// Y'CbCr in the 8-bit studio range to RGB (`0 ..= 1`); the inverse of [`Array::rgb2ycbcr`].
    #[track_caller]
    pub fn ycbcr2rgb(&self, standard: YccStd) -> Array<f32> {
        self.color(Color::YcbcrToRgb, [0.0; 3], standard as u32, "ycbcr2rgb", Location::caller())
    }

    /// Converts `[H, W, C]` images between colour spaces; see [`ColorSpace`].
    #[track_caller]
    pub fn color_space(&self, from: ColorSpace, to: ColorSpace) -> Array<f32> {
        let location = Location::caller();
        let rgb = |x: &Array<f32>| -> Array<f32> {
            match from {
                ColorSpace::Rgb => x.clone(),
                ColorSpace::Gray => x.color(Color::GrayToRgb, [1.0; 3], 0, "color_space", location),
                ColorSpace::Hsv => x.color(Color::HsvToRgb, [0.0; 3], 0, "color_space", location),
                ColorSpace::YCbCr(std) => x.color(Color::YcbcrToRgb, [0.0; 3], std as u32, "color_space", location),
            }
        };
        let rgb = rgb(self);
        match to {
            ColorSpace::Rgb => rgb,
            ColorSpace::Gray => rgb.color(Color::RgbToGray, [0.2126, 0.7152, 0.0722], 0, "color_space", location),
            ColorSpace::Hsv => rgb.color(Color::RgbToHsv, [0.0; 3], 0, "color_space", location),
            ColorSpace::YCbCr(std) => rgb.color(Color::RgbToYcbcr, [0.0; 3], std as u32, "color_space", location),
        }
    }
}

/// The standard of a Y'CbCr conversion.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum YccStd {
    /// ITU-R BT.601 (`kr = 0.299`, `kb = 0.114`).
    Bt601 = 0,
    /// ITU-R BT.709 (`kr = 0.2126`, `kb = 0.0722`).
    Bt709 = 1,
    /// ITU-R BT.2020 (`kr = 0.2627`, `kb = 0.0593`).
    Bt2020 = 2,
}

/// A colour space of [`Array::color_space`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColorSpace {
    /// One grey plane.
    Gray,
    /// Three planes `r, g, b` in `0 ..= 1`.
    Rgb,
    /// Three planes hue (`0 .. 1`), saturation, value.
    Hsv,
    /// Three planes Y', Cb, Cr in the studio range.
    YCbCr(YccStd),
}

fn check_sigma(spatial: f32, range: f32) -> Result<(), Failure> {
    if spatial.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || range.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || spatial > 64.0 {
        return Err(invalid(format!("bilateral sigmas must be positive (spatial up to 64), got {spatial} and {range}")));
    }
    Ok(())
}

/// Runs a bilateral kernel: the spatial exponent factor `-1 / (2 s^2)` goes in `p.c0`.
fn bilateral_run(input: &Array<f32>, spec: &Spec, spatial_sigma: f32, name: &'static str, location: &'static Location<'static>) -> Outcome<Array<f32>> {
    let ctx = input.context();
    input.storage()?;
    let geo = geometry(input.shape(), spec)?;
    let w = plan(ctx, spec, Pixel::F32, Edge::Clamp, name, location)?;
    let out = Array::<f32>::alloc(ctx, input.shape())?;
    let (src_pin, src_bytes) = input.ready()?;
    let (out_pin, out_bytes) = out.ready()?;
    let geo = if spec.op == Op::BilateralColor { Geometry { m: [geo.m[0], geo.m[1], 1], ..geo } } else { geo };
    launch_window(ctx, &w, &geo, 0, (-1.0 / (2.0 * spatial_sigma * spatial_sigma)).to_bits(), spec.c1, &[(&out_pin, out_bytes)], (&src_pin, src_bytes), None)?;
    Ok(out)
}

/// The cells of `padding` before and after an axis of `k` taps, for the first `axes` axes.
fn padding_of(padding: Padding, k: [usize; 3], axes: usize) -> ([usize; 3], [usize; 3]) {
    let mut before = [0; 3];
    let mut after = [0; 3];
    for d in 0..axes {
        match padding {
            Padding::Valid => {}
            Padding::Same => {
                before[d] = (k[d] - 1) / 2;
                after[d] = k[d] - 1 - before[d];
            }
            Padding::Full => {
                before[d] = k[d] - 1;
                after[d] = k[d] - 1;
            }
            Padding::Explicit { before: b, after: a } => {
                before[d] = b[d];
                after[d] = a[d];
            }
        }
    }
    (before, after)
}

/// The windows of every axis of an image of `n` cells with `window`, `stride` and `pad`.
fn unwrap_geometry(n: [usize; 2], window: [usize; 2], stride: [usize; 2], pad: [usize; 2]) -> Result<[usize; 2], Failure> {
    let mut count = [0; 2];
    for d in 0..2 {
        if window[d] == 0 || stride[d] == 0 {
            return Err(invalid("windows and strides must be non-zero"));
        }
        let padded = n[d] + 2 * pad[d];
        if padded < window[d] {
            return Err(invalid(format!("window of {} cells is larger than the padded axis {d} of {padded}", window[d])));
        }
        count[d] = (padded - window[d]) / stride[d] + 1;
    }
    Ok(count)
}

#[allow(clippy::too_many_arguments)]
fn unwrap_params(total: usize, n: [usize; 2], window: [usize; 2], stride: [usize; 2], pad: [usize; 2], g: [usize; 2], planes: usize, column: bool) -> Result<[u32; pointwise::UNWRAP_WORDS], Failure> {
    let u = |v: usize| to_u32(v, "dimension");
    Ok([
        u(total)?, u(n[0])?, u(n[1])?, u(window[0])?, u(window[1])?, u(stride[0])?, u(stride[1])?, u(pad[0])?, u(pad[1])?, u(g[0])?, u(g[1])?, u(planes)?, u32::from(column), 0, 0, 0,
    ])
}

/// Records a one-invocation-per-element kernel `op` writing `out` from `input`.
fn launch_pointwise(ctx: &Context, op: Builtin, elements: usize, params: &[u32], out: &Array<f32>, input: &Array<f32>) -> Result<(), Failure> {
    let (out_pin, out_bytes) = out.ready()?;
    let (in_pin, in_bytes) = input.ready()?;
    let kernel = ctx.builtin(op, "f32")?;
    let args = [
        RawArg::write(out_pin.buffer(), out_bytes),
        RawArg::read(in_pin.buffer(), in_bytes),
        RawArg::Uniform(bytemuck::cast_slice(params)),
    ];
    let n = to_u32(elements, "element count")?;
    ctx.launch(&kernel, fold_workgroups(ctx, n.div_ceil(pointwise::WORKGROUP))?, [n, 1, 1], &args)?;
    Ok(())
}

/// The Gaussian kernel of `rows x cols` cells (axis 0 by axis 1), the product of two Gaussians
/// centred on the middle of the kernel with standard deviations `sigma_r` along axis 0 and
/// `sigma_c` along axis 1, normalised to sum 1 (ArrayFire's `gaussianKernel`). A sigma that is
/// not positive is `0.25 * extent`.
pub fn gaussian_kernel(rows: usize, cols: usize, sigma_r: f32, sigma_c: f32) -> Result<Array<f32>, LumaError> {
    gaussian_kernel_on(&crate::default_context()?, rows, cols, sigma_r, sigma_c)
}

/// [`gaussian_kernel`] on an explicit context.
pub fn gaussian_kernel_on(ctx: &Context, rows: usize, cols: usize, sigma_r: f32, sigma_c: f32) -> Result<Array<f32>, LumaError> {
    let sigma = |sigma: f32, n: usize| if sigma > 0.0 { f64::from(sigma) } else { 0.25 * n as f64 };
    let (sr, sc) = (sigma(sigma_r, rows), sigma(sigma_c, cols));
    let line = |n: usize, sigma: f64| -> Vec<f64> {
        let centre = (n as f64 - 1.0) / 2.0;
        (0..n).map(|i| (-(i as f64 - centre).powi(2) / (2.0 * sigma * sigma)).exp()).collect()
    };
    let (gr, gc) = (line(rows, sr), line(cols, sc));
    let total: f64 = gr.iter().sum::<f64>() * gc.iter().sum::<f64>();
    let mut data = Vec::with_capacity(rows * cols);
    for c in &gc {
        for r in &gr {
            data.push((r * c / total) as f32);
        }
    }
    Array::from_slice_on(ctx, &data, [rows, cols])
}

/// The Gaussian of `len` cells centred on the middle, standard deviation `sigma` (`0.25 * len`
/// if not positive), normalised to sum 1: the taps of a separable blur
/// ([`Array::conv2_sep`]).
pub fn gaussian_kernel_1d(len: usize, sigma: f32) -> Result<Array<f32>, LumaError> {
    gaussian_kernel_1d_on(&crate::default_context()?, len, sigma)
}

/// [`gaussian_kernel_1d`] on an explicit context.
pub fn gaussian_kernel_1d_on(ctx: &Context, len: usize, sigma: f32) -> Result<Array<f32>, LumaError> {
    let sigma = if sigma > 0.0 { f64::from(sigma) } else { 0.25 * len as f64 };
    let centre = (len as f64 - 1.0) / 2.0;
    let line: Vec<f64> = (0..len).map(|i| (-(i as f64 - centre).powi(2) / (2.0 * sigma * sigma)).exp()).collect();
    let total: f64 = line.iter().sum();
    let data: Vec<f32> = line.iter().map(|v| (v / total) as f32).collect();
    Array::from_slice_on(ctx, &data, len)
}
