//! Reductions: whole-array sums, extrema and norms, fused with the expression behind them. They
//! are lazy: the result is a one-element array, and the host reads it with [`Array::item`].

use std::panic::Location;
use std::sync::{Arc, OnceLock};

use crate::array::{Array, Failure, Storage};
use crate::builtin::{Builtin, ReduceList, ReduceOp, MAX_REDUCTIONS};
use crate::context::{Context, RawArg, Scratch};
use crate::error::{LumaError, Result};
use crate::gpu_type::{GpuType, ScalarType};
use crate::jit::{self, Node};
use crate::ops::{fold_workgroups, len_u32, run_select, Rhs};
use crate::shape::Shape;

/// A reduction of a pending expression that has not been recorded yet: the operand is kept as an
/// expression until the result is used on the GPU (then the passes are recorded into a buffer,
/// without reading anything back) or on the host (then it is read like a synchronous reduction,
/// which lets the speculation of [`crate::jit`] answer loops of readbacks).
pub(crate) struct Reduced {
    operand: Arc<Node>,
    ops: ReduceList,
    wgsl: &'static str,
    /// Bytes of one element.
    elem: u64,
    result: OnceLock<std::result::Result<Arc<Storage>, Arc<LumaError>>>,
}

impl std::fmt::Debug for Reduced {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reduced").field("ops", &self.ops).field("forced", &self.result.get().is_some()).finish()
    }
}

impl Reduced {
    /// The error of the operand's evaluation or of the recorded passes, if any.
    pub(crate) fn failure(&self) -> Option<&Arc<LumaError>> {
        self.operand.failure().or_else(|| self.result.get().and_then(|result| result.as_ref().err()))
    }

    /// Whether the passes have not been recorded into a buffer yet.
    pub(crate) fn is_unforced(&self) -> bool {
        self.result.get().is_none()
    }

    /// Records the passes into a new buffer (once) and returns it. Nothing is read back.
    pub(crate) fn force(&self, ctx: &Context) -> &std::result::Result<Arc<Storage>, Arc<LumaError>> {
        self.result.get_or_init(|| self.record(ctx).map_err(Arc::new))
    }

    fn record(&self, ctx: &Context) -> Result<Arc<Storage>> {
        let count = self.ops.as_slice().len();
        let storage = Arc::new(Storage::new(ctx, count, self.elem as usize));
        let pin = storage.pin(0)?;
        let array = Array::<u32>::from_node(ctx, Shape::from(self.operand.len()), Arc::clone(&self.operand));
        let mut scratch = Scratch::new(ctx);
        reduce_into(&array, self.elem, self.wgsl, self.ops, &mut scratch, Some(pin.buffer()))?;
        Ok(storage)
    }

    /// The values on the host, without keeping them on the GPU.
    pub(crate) fn read<T: GpuType>(&self, ctx: &Context) -> Result<Vec<T>> {
        let array = Array::<T>::from_node(ctx, Shape::from(self.operand.len()), Arc::clone(&self.operand));
        reduce_array(&array, self.wgsl, self.ops)
    }
}

impl<T: ScalarType> Array<T> {
    /// The sum of all elements, as a lazy one-element array: [`Array::item`] reads the value on
    /// the host, and using the array in an expression with `self` (`&x / &x.sum()`) keeps it on
    /// the GPU. Float sums combine in trees and over at most a few hundred sequential steps per
    /// invocation, so the error stays far below that of a sequential sum. `u32` sums wrap around.
    #[track_caller]
    pub fn sum(&self) -> Array<T> {
        self.reduce_at([Reduction::Sum], "sum", Location::caller())
    }

    /// The smallest element, as a lazy one-element array (see [`Array::sum`]).
    #[track_caller]
    pub fn min(&self) -> Array<T> {
        self.reduce_at([Reduction::Min], "min", Location::caller())
    }

    /// The largest element, as a lazy one-element array (see [`Array::sum`]).
    #[track_caller]
    pub fn max(&self) -> Array<T> {
        self.reduce_at([Reduction::Max], "max", Location::caller())
    }

    /// Several reductions of the array in one pass over it (up to four), as an array of `N`
    /// elements: `let [lo, hi] = x.reduce([Reduction::Min, Reduction::Max]).to_array()?;`.
    /// Reading the data once instead of once per reduction is what makes it faster than
    /// separate calls on large arrays. Lazy like [`Array::sum`].
    #[track_caller]
    pub fn reduce<const N: usize>(&self, ops: [Reduction; N]) -> Array<T> {
        self.reduce_at(ops, "reduce", Location::caller())
    }

    /// The sum of the elements where `mask` is non-zero: `select(mask, self, 0).sum()` in one
    /// pass. Lazy like [`Array::sum`].
    #[track_caller]
    pub fn sum_where(&self, mask: &Array<u32>) -> Array<T> {
        let location = Location::caller();
        let selected = run_select(mask, Rhs::Array(self), Rhs::Scalar(T::zeroed()), location);
        let selected = Array::resolve(mask.context(), mask.shape(), selected, "sum_where", location);
        selected.reduce_at([Reduction::Sum], "sum_where", location)
    }

    fn reduce_at<const N: usize>(&self, ops: [Reduction; N], name: &'static str, location: &'static Location<'static>) -> Array<T> {
        const { assert!(N >= 1 && N <= MAX_REDUCTIONS, "Array::reduce takes one to four reductions") };
        let result = self.run_reduce(ReduceList::new(&ops.map(Reduction::op)));
        Array::resolve(self.context(), Shape::from(N), result, name, location)
    }

    fn run_reduce(&self, ops: ReduceList) -> std::result::Result<Array<T>, Failure> {
        let ctx = self.context();
        let shape = Shape::from(ops.as_slice().len());
        let elem = std::mem::size_of::<T>() as u64;
        let wgsl = T::reduce_type();
        // Poisoned operands pass their error on.
        self.src()?;
        if let Some(node) = self.pending() {
            let reduced = Reduced { operand: Arc::clone(node), ops, wgsl, elem, result: OnceLock::new() };
            return Ok(Array::from_reduced(ctx, shape, Arc::new(reduced)));
        }
        // Nothing is left to fuse: record the passes now.
        let storage = Arc::new(Storage::new(ctx, shape.len(), elem as usize));
        let pin = storage.pin(0)?;
        let mut scratch = Scratch::new(ctx);
        reduce_into(self, elem, wgsl, ops, &mut scratch, Some(pin.buffer()))?;
        Ok(Array::from_storage(ctx, shape, storage))
    }
}

impl<T: GpuType> Array<T> {
    /// The value of a one-element array on the host (for example of a reduction). Synchronizes.
    pub fn item(&self) -> Result<T> {
        if self.len() != 1 {
            return Err(LumaError::InvalidShape(format!("item() of an array of shape {}", self.shape())));
        }
        Ok(self.to_vec()?[0])
    }

    /// The `N` values of an `N`-element array on the host, for example of
    /// [`Array::reduce`]. Synchronizes.
    pub fn to_array<const N: usize>(&self) -> Result<[T; N]> {
        if self.len() != N {
            return Err(LumaError::LengthMismatch { expected: N, actual: self.len() });
        }
        let values = self.to_vec()?;
        let mut out = [T::zeroed(); N];
        out.copy_from_slice(&values);
        Ok(out)
    }
}

impl Array<f32> {
    /// The mean of all elements, as a lazy one-element array: the sum divided by the length.
    #[track_caller]
    pub fn mean(&self) -> Array<f32> {
        &self.sum() / (self.len() as f32)
    }

    /// The Euclidean (L2) norm: the square root of the sum of squares, as a lazy one-element
    /// array.
    #[track_caller]
    pub fn norm(&self) -> Array<f32> {
        self.reduce([Reduction::SumOfSquares]).sqrt()
    }
}

/// A reduction for [`Array::reduce`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Reduction {
    /// The sum of the elements.
    Sum,
    /// The smallest element.
    Min,
    /// The largest element.
    Max,
    /// The sum of the squared elements.
    SumOfSquares,
}

impl Reduction {
    fn op(self) -> ReduceOp {
        match self {
            Reduction::Sum => ReduceOp::Sum,
            Reduction::Min => ReduceOp::Min,
            Reduction::Max => ReduceOp::Max,
            Reduction::SumOfSquares => ReduceOp::SumSquares,
        }
    }
}

fn reduce_n<T: GpuType, const N: usize>(array: &Array<T>, wgsl: &'static str, ops: [Reduction; N]) -> Result<[T; N]> {
    const { assert!(N >= 1 && N <= MAX_REDUCTIONS, "Array::reduce takes one to four reductions") };
    let values = reduce_array(array, wgsl, ReduceList::new(&ops.map(Reduction::op)))?;
    let mut out = [T::zeroed(); N];
    for (slot, value) in out.iter_mut().zip(values) {
        *slot = value;
    }
    Ok(out)
}

/// Reduces the whole array on the GPU, once per op of `ops`: the first pass leaves at most
/// `Tuning::reduce_max_groups` partial results per op, the next one workgroup's worth of them,
/// until one element per op is left; those are read back together.
/// The first pass reads the data once for all ops; if the array is a pending lazy expression, it
/// computes it on the fly instead of reading a materialized buffer. `wgsl` is the WGSL type of
/// the elements; for vectors (which are never lazy) the operations act on every component
/// separately. Synchronizes.
fn reduce_array<T: GpuType>(array: &Array<T>, wgsl: &'static str, ops: ReduceList) -> Result<Vec<T>> {
    let ctx = array.context();
    let elem = std::mem::size_of::<T>() as u64;
    let count = ops.as_slice().len() as u64;
    if let Some(node) = array.pending()
        && matches!(wgsl, "f32" | "u32")
        && let Some(words) =
            jit::speculate::readback(node, ops, |scratch| reduce_into(array, elem, wgsl, ops, scratch, None))
    {
        return Ok(bytemuck::cast_slice(&words?).to_vec());
    }
    let mut scratch = Scratch::new(ctx);
    let current = reduce_into(array, elem, wgsl, ops, &mut scratch, None)?;
    let values = ctx.read::<T>(&current, elem * count)?;
    if values.len() as u64 != count {
        return Err(LumaError::Gpu("reduction results are missing".into()));
    }
    Ok(values)
}

/// Runs passes over `current` (`n` elements, or `n` per op once `per_op`) until one element per
/// op is left, and returns the buffer holding them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_passes(
    ctx: &Context,
    scratch: &mut Scratch<'_>,
    ops: ReduceList,
    wgsl: &'static str,
    elem: u64,
    mut current: wgpu::Buffer,
    mut n: u32,
    mut per_op: bool,
    target: Option<&wgpu::Buffer>,
) -> Result<wgpu::Buffer> {
    let count = ops.as_slice().len() as u64;
    while !per_op || n > 1 {
        let groups = ctx.tuning().reduce_groups(n);
        let list = if per_op { ops.rest() } else { ops };
        let in_bytes = u64::from(n) * elem * if per_op { count } else { 1 };
        // The last pass leaves one element per op: into `target` if there is one.
        let out = match target {
            Some(target) if groups == 1 => target.clone(),
            _ => scratch.take(u64::from(groups) * elem * count)?,
        };
        let core = ctx.builtin(Builtin::Reduce(list, per_op), wgsl)?;
        // The double-float pass has a second uniform field: the guard of its arithmetic (always 0).
        let uniform = [n, 0u32];
        let uniform = if wgsl == "df32" { &uniform[..] } else { &uniform[..1] };
        ctx.launch(
            &core,
            fold_workgroups(ctx, groups)?,
            [n, 1, 1],
            &[
                RawArg::write(&out, u64::from(groups) * elem * count),
                RawArg::read(&current, in_bytes),
                RawArg::Uniform(bytemuck::cast_slice(uniform)),
            ],
        )?;
        current = out;
        n = groups;
        per_op = true;
    }
    Ok(current)
}

/// The reduction of `array` as a buffer holding one element per op; see [`reduce_array`]. `elem`
/// is the size of an element in bytes. The values end up in `target` when it is given (a buffer
/// of at least `elem` bytes per op), without any copy.
fn reduce_into<T: GpuType>(
    array: &Array<T>,
    elem: u64,
    wgsl: &'static str,
    ops: ReduceList,
    scratch: &mut Scratch<'_>,
    target: Option<&wgpu::Buffer>,
) -> Result<wgpu::Buffer> {
    let ctx = array.context();
    let count = ops.as_slice().len() as u64;
    let finish = |scratch: &mut Scratch<'_>, current: wgpu::Buffer, n: u32, per_op: bool, target: Option<&wgpu::Buffer>| {
        finish_passes(ctx, scratch, ops, wgsl, elem, current, n, per_op, target)
    };

    let fused = match array.pending() {
        Some(node) => jit::reduce_first_pass(node, ops, scratch, target).map_err(Failure::into_error)?,
        None => None,
    };
    let mut results = Vec::new();
    match fused {
        Some(partials) => {
            // With several segments only the final collecting pass writes `target`.
            let target = target.filter(|_| partials.len() == 1);
            for (partial, groups) in partials {
                results.push(finish(scratch, partial, groups, true, target)?);
            }
        }
        None => {
            let storage = array.storage().map_err(Failure::into_error)?;
            let target = target.filter(|_| storage.segments() == 1);
            for j in 0..storage.segments() {
                let n = len_u32(storage.range(j).len())?;
                // Wgpu buffers are reference counted; the pin keeps it resident while it is bound.
                let pin = storage.pin(j)?;
                results.push(finish(scratch, pin.buffer().clone(), n, false, target)?);
            }
        }
    }

    if results.len() == 1 {
        return Ok(results.swap_remove(0));
    }
    let segments = results.len() as u64;
    let collect = scratch.take(segments * count * elem)?;
    for i in 0..count {
        for (j, partial) in results.iter().enumerate() {
            ctx.copy(partial, i * elem, &collect, (i * segments + j as u64) * elem, elem)?;
        }
    }
    finish(scratch, collect, len_u32(results.len())?, true, target)
}

/// Per-component reductions of `f32` vector arrays: the same tree reduction as the scalar
/// reductions, run on all components at once, so the accuracy is that of the scalar sum.
macro_rules! vector_reductions {
    ($n:literal, $wgsl:literal) => {
        impl Array<[f32; $n]> {
            /// The sum of all elements, per component, with the accuracy of [`Array::sum`].
            /// Synchronizes.
            pub fn sum(&self) -> Result<[f32; $n]> {
                Ok(self.reduce([Reduction::Sum])?[0])
            }

            /// The smallest value of every component. Synchronizes.
            pub fn min(&self) -> Result<[f32; $n]> {
                Ok(self.reduce([Reduction::Min])?[0])
            }

            /// The largest value of every component. Synchronizes.
            pub fn max(&self) -> Result<[f32; $n]> {
                Ok(self.reduce([Reduction::Max])?[0])
            }

            /// The Euclidean (L2) norm of every component taken over the whole array: the square
            /// root of the sum of squares, per component. Synchronizes.
            pub fn norm(&self) -> Result<[f32; $n]> {
                Ok(self.reduce([Reduction::SumOfSquares])?[0].map(f32::sqrt))
            }

            /// Several per-component reductions in one pass over the array (up to four), like
            /// [`Array::reduce`]. Synchronizes.
            pub fn reduce<const N: usize>(&self, ops: [Reduction; N]) -> Result<[[f32; $n]; N]> {
                reduce_n(self, $wgsl, ops)
            }
        }
    };
}

vector_reductions!(2, "vec2<f32>");
vector_reductions!(4, "vec4<f32>");
