//! `filter` and `nonzero` (stream compaction by a mask), their padded no-sync forms, and
//! `topk` / `bottomk`.
//!
//! Compaction runs in three steps, like a radix sort pass:
//!
//! 1. *Count*: a workgroup of [`WORKGROUP`] invocations counts the set bits of the [`TILE`]
//!    elements of a tile: each invocation takes one word of the bit-packed mask (32 elements) and
//!    counts it with `countOneBits`. The counts of the tiles are written to a buffer.
//! 2. *Scan*: the inclusive scan of the tile counts (`cumsum`); its last element is the number of
//!    selected elements, and the exclusive prefix of a tile is where its first selected element
//!    goes.
//! 3. *Scatter*: a workgroup scans the counts of the 256 words of its tile, then walks the tile
//!    32 elements at a time (the 32 invocations of a subgroup-sized slice read 32 consecutive
//!    elements, so loads and stores are coalesced) and writes every selected element to the tile
//!    offset plus the number of selected elements before it in the tile.
//!
//! The elements are copied as `u32` words (`stride / 4` per element), so the kernels do not know
//! the element type; `bool` arrays are compacted as `u32` and compared with 0 afterwards. A
//! mask that is an `Array<u32>` is compared with 0 first; an operand that is pending is
//! evaluated first.

use std::any::Any;
use std::fmt::Write;
use std::panic::Location;
use std::sync::Arc;

use crate::array::{Array, Failure, Storage};
use crate::builtin::Builtin;
use crate::context::{Context, RawArg};
use crate::error::{LumaError, Result};
use crate::gpu_type::{ElemLayout, ScalarType};
use crate::ops::{fold_workgroups, Cond, Condition, Outcome};
use crate::shape::Shape;
use crate::sort::SortKey;

/// Invocations of a workgroup of the filter kernels (and mask words of a tile).
const WORKGROUP: u32 = 256;
/// Elements of a tile: one 32-bit word of the mask per invocation.
const TILE: u32 = WORKGROUP * 32;

/// The WGSL of the kernel counting the set bits of every tile. Arguments: counts, mask, uniform
/// `{len, nwords, tiles}`.
pub(crate) fn count_source() -> String {
    format!(
        "struct Params {{ len: u32, nwords: u32, tiles: u32 }}
@group(1) @binding(0) var<storage, read_write> counts: array<u32>;
@group(1) @binding(1) var<storage, read> mask: array<u32>;
@group(1) @binding(2) var<uniform> p: Params;
var<workgroup> red: array<u32, {WORKGROUP}>;
@compute @workgroup_size({WORKGROUP})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let tile = wg.x + wg.y * nwg.x;
    if (tile >= p.tiles) {{ return; }}
    let w = tile * {WORKGROUP}u + t;
    var c = 0u;
    if (w < p.nwords) {{ c = countOneBits(mask[w]); }}
    red[t] = c;
    workgroupBarrier();
    for (var s = {half}u; s > 0u; s = s >> 1u) {{
        if (t < s) {{ red[t] = red[t] + red[t + s]; }}
        workgroupBarrier();
    }}
    if (t == 0u) {{ counts[tile] = red[0]; }}
}}
",
        half = WORKGROUP / 2
    )
}

/// The WGSL of the kernel writing the selected elements. Arguments: out, `src` (unless `index`),
/// mask, the inclusive scan of the tile counts, uniform `{len, nwords, tiles}`. An element is
/// `words` consecutive `u32`; `index` kernels write the element's index instead of reading it.
pub(crate) fn scatter_source(words: u8, index: bool) -> String {
    let mut src = String::from("struct Params { len: u32, nwords: u32, tiles: u32 }\n");
    let mut binding = 0;
    let mut declare = |access: &str, name: &str| {
        let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, {access}> {name}: array<u32>;");
        binding += 1;
    };
    declare("read_write", "out");
    if !index {
        declare("read", "src");
    }
    declare("read", "mask");
    declare("read", "ends");
    let _ = writeln!(src, "@group(1) @binding({binding}) var<uniform> p: Params;");
    let copy = if index {
        "out[dest] = e;".to_string()
    } else {
        (0..u32::from(words)).map(|j| format!("out[dest * {words}u + {j}u] = src[e * {words}u + {j}u];")).collect::<Vec<_>>().join(" ")
    };
    let _ = write!(
        src,
        "var<workgroup> wd: array<u32, {WORKGROUP}>;
var<workgroup> pre: array<u32, {WORKGROUP}>;
var<workgroup> ex: array<u32, {WORKGROUP}>;
@compute @workgroup_size({WORKGROUP})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let tile = wg.x + wg.y * nwg.x;
    if (tile >= p.tiles) {{ return; }}
    let gw = tile * {WORKGROUP}u + t;
    var w = 0u;
    if (gw < p.nwords) {{ w = mask[gw]; }}
    let c = countOneBits(w);
    wd[t] = w;
    pre[t] = c;
    workgroupBarrier();
    for (var s = 1u; s < {WORKGROUP}u; s = s << 1u) {{
        var v = pre[t];
        if (t >= s) {{ v = v + pre[t - s]; }}
        workgroupBarrier();
        pre[t] = v;
        workgroupBarrier();
    }}
    ex[t] = pre[t] - c;
    workgroupBarrier();
    var first = 0u;
    if (tile > 0u) {{ first = ends[tile - 1u]; }}
    let bit = t & 31u;
    let below = (1u << bit) - 1u;
    for (var m = 0u; m < 32u; m++) {{
        let q = m * {WORKGROUP}u + t;
        let wi = m * 8u + (t >> 5u);
        let v = wd[wi];
        if (((v >> bit) & 1u) != 0u) {{
            let dest = first + ex[wi] + countOneBits(v & below);
            let e = tile * {TILE}u + q;
            {copy}
        }}
    }}
}}
"
    );
    src
}

/// The tiles of a mask: its per-tile counts, scanned.
struct Tiles {
    /// The mask as bits (evaluated).
    mask: Array<bool>,
    /// The inclusive scan of the number of selected elements of every tile.
    ends: Array<u32>,
    len: u32,
    tiles: u32,
}

/// `cond` as an array of bits.
fn bits_of(cond: &impl Condition) -> Array<bool> {
    match cond.condition() {
        Cond::Bool(bits) => bits.clone(),
        Cond::Word(words) => words.ne(0u32),
    }
}

/// `value` as a `B`, if it is one.
fn downcast<A: 'static, B: 'static>(value: A) -> Option<B> {
    (Box::new(value) as Box<dyn Any>).downcast::<B>().ok().map(|boxed| *boxed)
}

/// The poison of a failed operation, as the error every result carries.
fn poison(failure: Failure, op: &'static str, location: &'static Location<'static>) -> Arc<LumaError> {
    match failure {
        Failure::Inherited(error) => error,
        Failure::New(error) => Arc::new(LumaError::Op { op, location, source: Arc::new(error) }),
    }
}

/// Counts the selected elements of every tile of `mask` and scans the counts. `shape` is the
/// shape of the array the mask selects from and `ctx` its context.
fn count_tiles(ctx: &Context, shape: Shape, mask: &Array<bool>) -> Outcome<Tiles> {
    let storage = Arc::clone(mask.storage()?);
    if !ctx.same_as(mask.context()) {
        return Err(LumaError::ContextMismatch.into());
    }
    if mask.shape() != shape {
        return Err(LumaError::ShapeMismatch { lhs: shape.dims(), rhs: mask.shape().dims() }.into());
    }
    let too_large = || LumaError::DimensionTooLarge { shape: shape.dims() };
    let len = u32::try_from(shape.len()).map_err(|_| too_large())?;
    let tiles = len.div_ceil(TILE);
    let params = [len, len.div_ceil(32), tiles];
    let counts = Arc::new(Storage::new(ctx, tiles as usize, ElemLayout::plain(4)));
    {
        let mask_pin = storage.pin_single()?;
        let counts_pin = counts.pin_single()?;
        let kernel = ctx.builtin(Builtin::FilterCount, "u32")?;
        let args = [
            RawArg::write(counts_pin.buffer(), counts.byte_len()),
            RawArg::read(mask_pin.buffer(), storage.byte_len()),
            RawArg::Uniform(bytemuck::bytes_of(&params)),
        ];
        ctx.launch(&kernel, fold_workgroups(ctx, tiles)?, [len, 1, 1], &args)?;
    }
    let ends = Array::<u32>::from_storage(ctx, Shape::from(tiles as usize), counts).cumsum(0);
    Ok(Tiles { mask: mask.clone(), ends, len, tiles })
}

impl Tiles {
    /// The number of selected elements, as a one-element array.
    fn count(&self) -> Array<u32> {
        let last = self.tiles as usize - 1;
        self.ends.slice(0, last..last + 1)
    }

    /// Records the scatter of the selected elements of `src` (all of them as `u32` words if
    /// `None`: their indices) into the start of `out`, which has an element of `words` words.
    fn scatter(&self, ctx: &Context, src: Option<&Storage>, words: u8, out: &Storage) -> Outcome<()> {
        let params = [self.len, self.len.div_ceil(32), self.tiles];
        let mask = self.mask.storage()?;
        let (mask_pin, ends) = (mask.pin_single()?, self.ends.storage()?);
        let ends_pin = ends.pin_single()?;
        let out_pin = out.pin_single()?;
        let src_pin = src.map(Storage::pin_single).transpose()?;
        let kernel = ctx.builtin(Builtin::FilterScatter { words, index: src.is_none() }, "u32")?;
        let mut args = vec![RawArg::write(out_pin.buffer(), out.byte_len())];
        if let (Some(pin), Some(storage)) = (&src_pin, src) {
            args.push(RawArg::read(pin.buffer(), storage.byte_len()));
        }
        args.push(RawArg::read(mask_pin.buffer(), mask.byte_len()));
        args.push(RawArg::read(ends_pin.buffer(), ends.byte_len()));
        args.push(RawArg::Uniform(bytemuck::bytes_of(&params)));
        ctx.launch(&kernel, fold_workgroups(ctx, self.tiles)?, [self.len, 1, 1], &args)?;
        Ok(())
    }

    /// The number of selected elements on the host. Synchronizes.
    fn selected(&self) -> Outcome<usize> {
        Ok(self.count().item()? as usize)
    }
}

/// The number of `u32` words of an element of `T`; fails if it has more than a `u32` can index.
fn words_of<T: ScalarType>(len: usize) -> Outcome<u8> {
    let words = T::LAYOUT.stride() / 4;
    if (len as u64) * (words as u64) > u64::from(u32::MAX) {
        return Err(LumaError::DimensionTooLarge { shape: [len, 1, 1, 1] }.into());
    }
    Ok(words as u8)
}

impl<T: ScalarType> Array<T> {
    /// The elements where `mask` holds, in flat order, as a one-dimensional array; `None` if
    /// there are none (luma has no arrays without elements). `mask` is an `Array<bool>` or an
    /// `Array<u32>` (non-zero means selected) of this array's shape, else
    /// [`LumaError::ShapeMismatch`].
    ///
    /// The result's length depends on the data, so this reads the count back and
    /// **synchronizes** once (about 100 µs on an RTX 3090: submit, wait, map); use
    /// [`Array::filter_padded`] to stay on the device. This array and the mask are evaluated
    /// first. The work is a count and a scan over tiles of 8192 elements and one scatter that
    /// reads this array once and writes the selected elements: about as fast as a copy.
    ///
    /// `None` rather than a poisoned array because selecting nothing is a result, not an error;
    /// the `Result` is for failures of the device, a shape mismatch or an array that is poisoned.
    pub fn filter(&self, mask: &impl Condition) -> Result<Option<Array<T>>> {
        self.run_filter(&bits_of(mask)).map_err(Failure::into_error)
    }

    /// Like [`Array::filter`], without reading anything back: the first array has the shape of
    /// this one and holds the selected elements in flat order followed by `fill`; the second is
    /// the number of selected elements, as a one-element `u32` array that can be read with
    /// [`Array::item`] or used lazily. Errors poison both results.
    #[track_caller]
    pub fn filter_padded(&self, mask: &impl Condition, fill: T) -> (Array<T>, Array<u32>) {
        let location = Location::caller();
        match self.run_filter_padded(&bits_of(mask), fill) {
            Ok(pair) => pair,
            Err(failure) => {
                let error = poison(failure, "filter_padded", location);
                (Array::poisoned(self.context(), self.shape(), error.clone()), Array::poisoned(self.context(), Shape::from(1usize), error))
            }
        }
    }

    fn run_filter(&self, mask: &Array<bool>) -> Outcome<Option<Array<T>>> {
        if T::LAYOUT.is_packed() && T::wgsl_type() == "f32" {
            // Half-precision floats are compacted as `f32` (exact both ways).
            let location = Location::caller();
            let kept = self.cast_at::<f32>(location).run_filter(mask)?;
            return Ok(kept.map(|kept| kept.cast_at::<T>(location)));
        }
        if T::LAYOUT.is_packed() && !T::LAYOUT.is_bits() {
            // Small integers are compacted as the words they widen to and narrowed again.
            let location = Location::caller();
            let kept = self.cast_at::<u32>(location).run_filter(mask)?;
            return Ok(kept.map(|kept| kept.cast_at::<T>(location)));
        }
        if T::LAYOUT.is_packed() {
            // Bits are compacted as words and compared with 0 afterwards.
            let words = self.cast_at::<u32>(Location::caller());
            let kept = words.run_filter(mask)?;
            return Ok(kept.map(|kept| downcast(kept.ne(0u32)).expect("only bool arrays are bit-packed")));
        }
        let ctx = self.context();
        let words = words_of::<T>(self.len())?;
        let src = Arc::clone(self.storage()?);
        let tiles = count_tiles(ctx, self.shape(), mask)?;
        let selected = tiles.selected()?;
        if selected == 0 {
            return Ok(None);
        }
        let out = Array::<T>::alloc(ctx, Shape::from(selected))?;
        tiles.scatter(ctx, Some(&src), words, out.storage()?)?;
        Ok(Some(out))
    }

    fn run_filter_padded(&self, mask: &Array<bool>, fill: T) -> Outcome<(Array<T>, Array<u32>)> {
        if T::LAYOUT.is_packed() && T::wgsl_type() == "f32" {
            let location = Location::caller();
            let fill = downcast::<T, half::f16>(fill).map(f32::from).or_else(|| downcast::<T, half::bf16>(fill).map(f32::from)).expect("only f16 and bf16 are packed floats");
            let (kept, count) = self.cast_at::<f32>(location).run_filter_padded(mask, fill)?;
            return Ok((kept.cast_at::<T>(location), count));
        }
        if T::LAYOUT.is_packed() && !T::LAYOUT.is_bits() {
            let location = Location::caller();
            let (kept, count) = self.cast_at::<u32>(location).run_filter_padded(mask, bytemuck::cast::<_, u32>(fill.into_repr()))?;
            return Ok((kept.cast_at::<T>(location), count));
        }
        if T::LAYOUT.is_packed() {
            let fill = downcast::<T, bool>(fill).expect("only bool arrays are bit-packed");
            let words = self.cast_at::<u32>(Location::caller());
            let (kept, count) = words.run_filter_padded(mask, u32::from(fill))?;
            return Ok((downcast(kept.ne(0u32)).expect("only bool arrays are bit-packed"), count));
        }
        let ctx = self.context();
        let words = words_of::<T>(self.len())?;
        let src = Arc::clone(self.storage()?);
        let tiles = count_tiles(ctx, self.shape(), mask)?;
        let out = Array::<T>::full_on(ctx, self.shape(), fill)?;
        tiles.scatter(ctx, Some(&src), words, out.storage()?)?;
        Ok((out, tiles.count()))
    }
}

/// `nonzero` and `nonzero_padded` of a mask type (see [`Array<bool>::nonzero`](Array::nonzero)).
macro_rules! nonzero {
    ($ty:ty, $set:literal) => {
        impl Array<$ty> {
            /// The flat indices of the
            #[doc = $set]
            /// elements, ascending, as a one-dimensional `Array<u32>`; `None` if there are none
            /// (luma has no arrays without elements). Like [`Array::filter`] this reads the count
            /// back and **synchronizes** once (about 100 µs on an RTX 3090).
            ///
            /// The indices are flat: axis 0 varies fastest, so for a shape `[d0, d1, d2, d3]` the
            /// element `[i0, i1, i2, i3]` has index `i0 + d0 * (i1 + d1 * (i2 + d2 * i3))`, and
            /// the index `f` is element `[f % d0, f / d0 % d1, f / (d0 * d1) % d2, f / (d0 * d1 *
            /// d2)]`; compute those lazily from the result with `%` and `/`.
            pub fn nonzero(&self) -> Result<Option<Array<u32>>> {
                nonzero_of(self).map_err(Failure::into_error)
            }

            /// [`Array::nonzero`] without a readback: a `u32` array of this array's length holding
            /// the indices followed by `fill`, and their number as a one-element `u32` array.
            /// Errors poison both results.
            #[track_caller]
            pub fn nonzero_padded(&self, fill: u32) -> (Array<u32>, Array<u32>) {
                let location = Location::caller();
                match nonzero_padded_of(self, fill) {
                    Ok(pair) => pair,
                    Err(failure) => {
                        let error = poison(failure, "nonzero_padded", location);
                        (Array::poisoned(self.context(), self.shape(), error.clone()), Array::poisoned(self.context(), Shape::from(1usize), error))
                    }
                }
            }
        }
    };
}

nonzero!(bool, "true");
nonzero!(u32, "non-zero");

fn nonzero_of(cond: &impl Condition) -> Outcome<Option<Array<u32>>> {
    let mask = bits_of(cond);
    let ctx = mask.context();
    let tiles = count_tiles(ctx, mask.shape(), &mask)?;
    let selected = tiles.selected()?;
    if selected == 0 {
        return Ok(None);
    }
    let out = Array::<u32>::alloc(ctx, Shape::from(selected))?;
    tiles.scatter(ctx, None, 1, out.storage()?)?;
    Ok(Some(out))
}

fn nonzero_padded_of(cond: &impl Condition, fill: u32) -> Outcome<(Array<u32>, Array<u32>)> {
    let mask = bits_of(cond);
    let ctx = mask.context();
    let tiles = count_tiles(ctx, mask.shape(), &mask)?;
    let out = Array::<u32>::full_on(ctx, mask.shape(), fill)?;
    tiles.scatter(ctx, None, 1, out.storage()?)?;
    Ok((out, tiles.count()))
}

impl<T: SortKey> Array<T> {
    /// The `k` largest elements, largest first, and their flat indices (ties in input order;
    /// floats order as in [`Array::sort`], so NaNs count as the largest). Both are
    /// one-dimensional with `k` elements; `k` of 0 or more than the length poisons both with
    /// [`LumaError::InvalidShape`].
    ///
    /// Implemented as [`Array::argsort_descending`] of the flat array and a gather of the first
    /// `k`, so the cost is that of a full radix sort (about 1.7 ms for 2^24 keys on an RTX 3090)
    /// whatever `k` is.
    #[track_caller]
    pub fn topk(&self, k: usize) -> (Array<T>, Array<u32>) {
        self.top(k, true, "topk", Location::caller())
    }

    /// The `k` smallest elements, smallest first, and their flat indices (ties in input order;
    /// NaNs come last). Otherwise as [`Array::topk`], with [`Array::argsort`].
    #[track_caller]
    pub fn bottomk(&self, k: usize) -> (Array<T>, Array<u32>) {
        self.top(k, false, "bottomk", Location::caller())
    }

    fn top(&self, k: usize, descending: bool, op: &'static str, location: &'static Location<'static>) -> (Array<T>, Array<u32>) {
        let ctx = self.context();
        let n = self.len();
        if k == 0 || k > n {
            let error = LumaError::InvalidShape(format!("{op}({k}) of {n} elements"));
            let error = Arc::new(LumaError::Op { op, location, source: Arc::new(error) });
            return (Array::poisoned(ctx, Shape::from(k.max(1)), error.clone()), Array::poisoned(ctx, Shape::from(k.max(1)), error));
        }
        let flat = self.reshape(n);
        let order = if descending { flat.argsort_descending() } else { flat.argsort() };
        let indices = order.slice(0, 0..k);
        (flat.take(&indices), indices)
    }
}
