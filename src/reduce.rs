//! Reductions: whole-array sums, extrema and norms, fused with the expression behind them.

use crate::array::{Array, Failure};
use crate::builtin::{Builtin, ReduceList, ReduceOp, MAX_REDUCTIONS};
use crate::context::{Context, RawArg, Scratch};
use crate::error::{LumaError, Result};
use crate::gpu_type::{GpuType, ScalarType};
use crate::jit;
use crate::ops::{fold_workgroups, len_u32};

impl<T: ScalarType> Array<T> {
    /// The sum of all elements. Float sums combine in trees and over at most a few hundred
    /// sequential steps per invocation, so the error stays far below that of a sequential sum.
    /// `u32` sums wrap around. Synchronizes.
    pub fn sum(&self) -> Result<T> {
        Ok(self.reduce([Reduction::Sum])?[0])
    }

    /// The smallest element. Synchronizes.
    pub fn min(&self) -> Result<T> {
        Ok(self.reduce([Reduction::Min])?[0])
    }

    /// The largest element. Synchronizes.
    pub fn max(&self) -> Result<T> {
        Ok(self.reduce([Reduction::Max])?[0])
    }

    /// Several reductions of the array in one pass over it (up to four), for example
    /// `let [lo, hi] = x.reduce([Reduction::Min, Reduction::Max])?;`. Reading the data once
    /// instead of once per reduction is what makes it faster than separate calls on large arrays.
    /// Synchronizes.
    pub fn reduce<const N: usize>(&self, ops: [Reduction; N]) -> Result<[T; N]> {
        reduce_n(self, T::wgsl_type(), ops)
    }
}

impl Array<f32> {
    /// The mean of all elements (the sum computed on the GPU, divided on the host).
    /// Synchronizes.
    pub fn mean(&self) -> Result<f32> {
        let sum = self.sum()?;
        Ok((f64::from(sum) / self.len() as f64) as f32)
    }

    /// The Euclidean (L2) norm: the square root of the sum of squares. Synchronizes.
    pub fn norm(&self) -> Result<f32> {
        Ok(self.reduce([Reduction::SumOfSquares])?[0].sqrt())
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
        && let Some(words) = jit::speculate::readback(node, ops, |scratch| reduce_into(array, wgsl, ops, scratch))
    {
        return Ok(bytemuck::cast_slice(&words?).to_vec());
    }
    let mut scratch = Scratch::new(ctx);
    let current = reduce_into(array, wgsl, ops, &mut scratch)?;
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
) -> Result<wgpu::Buffer> {
    let count = ops.as_slice().len() as u64;
    while !per_op || n > 1 {
        let groups = ctx.tuning().reduce_groups(n);
        let list = if per_op { ops.rest() } else { ops };
        let in_bytes = u64::from(n) * elem * if per_op { count } else { 1 };
        let out = scratch.take(u64::from(groups) * elem * count)?;
        let core = ctx.builtin(Builtin::Reduce(list, per_op), wgsl)?;
        ctx.launch(
            &core,
            fold_workgroups(ctx, groups)?,
            [n, 1, 1],
            &[
                RawArg::write(&out, u64::from(groups) * elem * count),
                RawArg::read(&current, in_bytes),
                RawArg::Uniform(bytemuck::bytes_of(&[n])),
            ],
        )?;
        current = out;
        n = groups;
        per_op = true;
    }
    Ok(current)
}

/// The reduction of `array` as a buffer holding one element per op; see [`reduce_array`].
fn reduce_into<T: GpuType>(array: &Array<T>, wgsl: &'static str, ops: ReduceList, scratch: &mut Scratch<'_>) -> Result<wgpu::Buffer> {
    let ctx = array.context();
    let elem = std::mem::size_of::<T>() as u64;
    let count = ops.as_slice().len() as u64;
    let finish = |scratch: &mut Scratch<'_>, current: wgpu::Buffer, n: u32, per_op: bool| {
        finish_passes(ctx, scratch, ops, wgsl, elem, current, n, per_op)
    };

    let fused = match array.pending() {
        Some(node) => jit::reduce_first_pass(node, ops, scratch).map_err(Failure::into_error)?,
        None => None,
    };
    let mut results = Vec::new();
    match fused {
        Some(partials) => {
            for (partial, groups) in partials {
                results.push(finish(scratch, partial, groups, true)?);
            }
        }
        None => {
            let storage = array.storage().map_err(Failure::into_error)?;
            for j in 0..storage.segments() {
                let n = len_u32(storage.range(j).len())?;
                // Wgpu buffers are reference counted; the pin keeps it resident while it is bound.
                let pin = storage.pin(j)?;
                results.push(finish(scratch, pin.buffer().clone(), n, false)?);
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
    finish(scratch, collect, len_u32(results.len())?, true)
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
