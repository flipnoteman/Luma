//! Incremental accumulation: statistics of data that arrives in batches over time.
//!
//! An accumulator folds batches into a small mergeable *state* on the device. `push` records the
//! work and reads nothing back; the state can be turned into result arrays at any time
//! (`mean`, `var`, `hist`, ...), merged with the state of another accumulator (another run,
//! thread or device), cleared and written to bytes. The batches are never kept, and their number
//! and size are unbounded: `count` of a state is exact to 2^48 samples (`Df32` states) or 2^53
//! (`f64` states), and the merge arithmetic is Chan's pairwise update, so `1e9` samples do not
//! lose the low digits of the mean or of the variance.
//!
//! * [`Accumulator`]: count, sum, mean, variance, skewness, kurtosis, minimum, maximum and their
//!   global sample indices of one stream, over the whole batch or per position along chosen
//!   axes ([`Fold`]), also of the elements a mask selects or per group label.
//! * [`CoMoments`]: the two streams' means, variances, covariance and correlation per
//!   position; [`Covariance`]: the covariance and correlation matrix between the features of
//!   a stream.
//! * [`Histogram`] and [`Histogram2d`]: counts in fixed bins of one stream or of two streams
//!   jointly, with the marginals and mutual information derived.
//! * [`TopK`]: the `k` largest or smallest elements of a stream with their global indices.
//!
//! # Precision
//!
//! A pushed batch is reduced in `f32` (one pass over the batch, fused with the expression that
//! produces it) *relative to a pivot*: the running mean of the state, which makes the sums of
//! `x - pivot` small and their rounding errors small too, however far from zero the data is
//! (`1e9 + 1e-2 * noise` in an `f64` batch keeps full accuracy). The reduced sums are added
//! into a state kept in [`Df32`] (default) or `f64` ([`Wide`]) arithmetic. Integer batches are
//! exact in counts, sums and extrema while `|x - pivot|` and the batch's partial sums stay below
//! 2^24, and rounded to `f32` precision beyond that.
//!
//! # NaN
//!
//! With [`NanPolicy::Propagate`] (the default) a NaN or infinity in a position's stream makes
//! its mean, variance and sums NaN for good (and its minimum and maximum unspecified: backends
//! differ in what `min` of a NaN gives). [`NanPolicy::Skip`] leaves NaN elements out: they
//! count in no statistic, as if masked out by `push_where`.

mod blob;
mod comoments;
mod hist;
mod moments;
mod topk;
mod wide;

pub use comoments::{CoMoments, CoMomentsBuilder, Covariance};
pub use hist::{Histogram, Histogram2d};
pub use moments::{Accumulator, AccumulatorBuilder};
pub use topk::TopK;
pub use wide::{AccumInput, Wide};

use crate::error::{LumaError, Result};

/// The axes of a pushed batch that are folded into the state; the other axes are *positions*,
/// each with a state of its own. Axis 0 is the fastest varying one, so a `[features, batch]`
/// array (row-major `batch` rows of `features`) is folded along axis 1 to give per-feature
/// statistics; the results have extent 1 along the folded axes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fold {
    /// Every element of the batch is one sample of a single stream (one position).
    All,
    /// The listed axes (distinct, below 4) are folded.
    Axes(Vec<usize>),
}

impl Fold {
    /// Fold the whole batch.
    pub fn all() -> Fold {
        Fold::All
    }

    /// Fold one axis.
    pub fn axis(axis: usize) -> Fold {
        Fold::Axes(vec![axis])
    }

    /// Fold several axes.
    pub fn axes(axes: &[usize]) -> Fold {
        Fold::Axes(axes.to_vec())
    }

    /// The folded axes of a batch of shape `dims`, sorted.
    pub(crate) fn resolve(&self, dims: [usize; 4]) -> Result<Vec<usize>> {
        match self {
            Fold::All => Ok(vec![0, 1, 2, 3]),
            Fold::Axes(axes) => {
                let mut sorted = axes.clone();
                sorted.sort_unstable();
                sorted.dedup();
                if sorted.is_empty() || sorted.len() != axes.len() || sorted.last().is_some_and(|&a| a >= 4) {
                    return Err(LumaError::InvalidShape(format!("fold axes {axes:?} must be distinct and below 4")));
                }
                let _ = dims;
                Ok(sorted)
            }
        }
    }

    pub(crate) fn code(&self) -> u8 {
        match self {
            Fold::All => 0xFF,
            Fold::Axes(axes) => axes.iter().fold(0u8, |bits, &a| bits | 1 << a),
        }
    }

    pub(crate) fn from_code(code: u8) -> Fold {
        if code == 0xFF { Fold::All } else { Fold::Axes((0..4).filter(|a| code >> a & 1 != 0).collect()) }
    }
}

/// What a push does with NaN elements, see the module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NanPolicy {
    /// NaN poisons the positions it falls in.
    #[default]
    Propagate,
    /// NaN elements are left out of every statistic.
    Skip,
}

/// The shape of a batch with the folded axes set to 1: the shape of the state of one stream.
pub(crate) fn kept_dims(dims: [usize; 4], axes: &[usize]) -> [usize; 4] {
    std::array::from_fn(|k| if axes.contains(&k) { 1 } else { dims[k] })
}

/// The number of samples a batch adds along the folded axes.
pub(crate) fn folded_extent(dims: [usize; 4], axes: &[usize]) -> u64 {
    axes.iter().map(|&a| dims[a] as u64).product()
}

/// Whether reducing `axes` of a batch of shape `dims` is a reduction of the whole array.
pub(crate) fn is_flat(dims: [usize; 4], axes: &[usize]) -> bool {
    (0..4).all(|k| dims[k] == 1 || axes.contains(&k))
}

pub(crate) fn invalid(msg: impl Into<String>) -> LumaError {
    LumaError::InvalidShape(msg.into())
}
