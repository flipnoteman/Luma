//! Template matching, nearest-neighbour and Hamming matchers.
//!
//! [`Array::match_template`] slides a template over an image and scores every window with one of
//! the metrics of [`MatchType`]; [`nearest_neighbour`] and [`hamming_matcher`] find, for every
//! query feature, the `k` closest of a set of train features.

mod neighbours;
mod template;

use std::panic::Location;

use crate::array::Array;
use crate::error::Result;
use crate::gpu_type::ScalarType;

/// The metric of [`Array::match_template`] (ArrayFire's `af_match_type`, which has the three
/// families of the variants below and `AF_SHD`; the Hamming distance is [`DistType::Shd`]).
///
/// `t` is the template and `w` the window of the image the template is held against, both of
/// `n` pixels; `mean` is their average, the mean of the window taken over the zero-padded window
/// (as the mean of the template over the template).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MatchType {
    /// Sum of absolute differences: `sum |w - t|`.
    Sad,
    /// Zero-mean SAD: `sum |(w - mean(w)) - (t - mean(t))|`.
    Zsad,
    /// Locally scaled SAD: `sum |w - mean(w) / mean(t) * t|`.
    Lsad,
    /// Sum of squared differences: `sum (w - t)^2`.
    Ssd,
    /// Zero-mean SSD.
    Zssd,
    /// Locally scaled SSD.
    Lssd,
    /// Normalized cross-correlation: `sum (w t) / sqrt(sum w^2 * sum t^2)`. ArrayFire rejects it.
    Ncc,
    /// Zero-mean NCC: the correlation of `w - mean(w)` and `t - mean(t)`. ArrayFire rejects it.
    Zncc,
}

impl Array<f32> {
    /// Template matching (ArrayFire's `matchTemplate`): for every pixel of the `[H, W]` image
    /// (more axes are a batch), the disparity between the template (`[h, w]`) and the window of
    /// the image *starting* at that pixel and extending by the template's size (axis 0 first); the
    /// pixels past the image count as 0. The result has the shape of the image. For the metrics
    /// see [`MatchType`].
    ///
    /// Differences from ArrayFire: the normalized metrics [`MatchType::Ncc`] and
    /// [`MatchType::Zncc`] work (ArrayFire fails with an invalid argument; a window whose
    /// denominator is 0 scores 0), a template with a single column or row works (ArrayFire wants
    /// at least a 2-D shape), and images with a fourth axis are matched plane by plane (ArrayFire
    /// computes the first plane only). A template larger than the image is fine, as in
    /// ArrayFire.
    ///
    /// Float sums are accumulated in a different order than ArrayFire does, so results agree to
    /// a few ulps per term, not bitwise.
    ///
    /// ```
    /// use luma::{Array, MatchType};
    /// # fn main() -> Result<(), luma::LumaError> {
    /// let image = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])?;
    /// let template = Array::from_slice(&[2.0f32, 3.0], [2, 1])?;
    /// // The window at the last row of a column reaches one pixel past the image (0).
    /// let sad = image.match_template(&template, MatchType::Sad);
    /// assert_eq!(sad.to_vec()?, [2.0, 0.0, 4.0, 4.0, 6.0, 7.0]);
    /// # Ok(()) }
    /// ```
    #[track_caller]
    pub fn match_template(&self, template: &Array<f32>, metric: MatchType) -> Array<f32> {
        let location = Location::caller();
        let result = template::run(self, template, metric);
        Array::resolve(self.context(), self.shape(), result, "match_template", location)
    }
}

/// The distance of [`nearest_neighbour`] (ArrayFire's `af_match_type` restricted to the three
/// it supports there).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DistType {
    /// Sum of absolute differences.
    Sad,
    /// Sum of squared differences.
    Ssd,
    /// Hamming distance: the number of differing bits, summed over the elements (`u32` only).
    Shd,
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
}

/// The element types of feature matrices: `f32` (distances `f32`), `u32` (distances `u32`) and
/// `i32` (distances `i32`).
pub trait Neighbour: ScalarType + sealed::Sealed {
    /// The type of the distances.
    type Dist: ScalarType;
    #[doc(hidden)]
    const NAME: &'static str;
    #[doc(hidden)]
    const ELEM: neighbours::Elem;
    #[doc(hidden)]
    fn supports(metric: DistType) -> bool;
}

impl Neighbour for f32 {
    type Dist = f32;
    const NAME: &'static str = "f32";
    const ELEM: neighbours::Elem = neighbours::Elem::F32;
    fn supports(metric: DistType) -> bool {
        metric != DistType::Shd
    }
}

impl Neighbour for u32 {
    type Dist = u32;
    const NAME: &'static str = "u32";
    const ELEM: neighbours::Elem = neighbours::Elem::U32;
    fn supports(_: DistType) -> bool {
        true
    }
}

impl Neighbour for i32 {
    type Dist = i32;
    const NAME: &'static str = "i32";
    const ELEM: neighbours::Elem = neighbours::Elem::I32;
    fn supports(metric: DistType) -> bool {
        metric != DistType::Shd
    }
}

/// The `n_dist` nearest train features of every query feature (ArrayFire's `nearestNeighbour`).
///
/// `query` and `train` are matrices of features: with `dist_dim` 0 a feature is a column
/// (`[len, n]`), with 1 a row (`[n, len]`); `len` is the same for both. The result is
/// `(index, distance)`, both `[n_dist, nq]`: column `j` lists the train features nearest to query
/// `j`, closest first, as their index along the feature axis and their distance (`f32` for `f32`
/// features, else of the element type). `n_dist` is at most 256 and at most the number of train
/// features, as in ArrayFire.
///
/// Ties are broken by the smaller train index (ArrayFire's order among equal distances is
/// arbitrary).
///
/// # Errors
///
/// A mismatch of the feature length, an out-of-range `dist_dim` or `n_dist`, and
/// [`DistType::Shd`] for `f32` or `i32` features (ArrayFire rejects these too). ArrayFire also
/// takes `u8`, `u16`, `i16` and `f64` features; luma has no such element types (`f64` emulated
/// by luma is not supported here).
///
/// ```
/// use luma::{nearest_neighbour, Array, DistType};
/// # fn main() -> Result<(), luma::LumaError> {
/// // Two-element features as columns; the queries are (0, 0) and (5, 5).
/// let train = Array::from_slice(&[0.0f32, 1.0, 4.0, 4.0, 9.0, 9.0], [2, 3])?;
/// let query = Array::from_slice(&[0.0f32, 0.0, 5.0, 5.0], [2, 2])?;
/// let (index, dist) = nearest_neighbour(&query, &train, 0, 2, DistType::Ssd)?;
/// assert_eq!(index.to_vec()?, [0, 1, 1, 2]);
/// assert_eq!(dist.to_vec()?, [1.0, 32.0, 2.0, 32.0]);
/// # Ok(()) }
/// ```
pub fn nearest_neighbour<T: Neighbour>(query: &Array<T>, train: &Array<T>, dist_dim: usize, n_dist: usize, dist: DistType) -> Result<(Array<u32>, Array<T::Dist>)> {
    neighbours::run(query, train, dist_dim, n_dist, dist)
}

/// The Hamming matcher (ArrayFire's `hammingMatcher`): [`nearest_neighbour`] with
/// [`DistType::Shd`] on `u32` descriptors (256-bit ORB descriptors are 8 words). The distance of
/// two features is the number of differing bits over all their words; the result is
/// `(index, distance)`, both `[n_dist, nq]` and `u32`.
///
/// ArrayFire also accepts `u8`, `u16` and `u64` descriptors, which luma has no arrays of.
///
/// ```
/// use luma::{hamming_matcher, Array};
/// # fn main() -> Result<(), luma::LumaError> {
/// // Descriptors of two words each, as columns.
/// let train = Array::from_slice(&[0u32, 0, 0xFF, 0, 0xF, 1], [2, 3])?;
/// let query = Array::from_slice(&[0x7u32, 0], [2, 1])?;
/// let (index, dist) = hamming_matcher(&query, &train, 0, 2)?;
/// assert_eq!(index.to_vec()?, [2, 0]);
/// assert_eq!(dist.to_vec()?, [2, 3]);
/// # Ok(()) }
/// ```
pub fn hamming_matcher(query: &Array<u32>, train: &Array<u32>, dist_dim: usize, n_dist: usize) -> Result<(Array<u32>, Array<u32>)> {
    neighbours::run(query, train, dist_dim, n_dist, DistType::Shd)
}

/// One kernel of this module (part of the key of its cache entry in the context).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Kernel {
    /// [`template`].
    Template(template::Kernel),
    /// [`neighbours`].
    Neighbours(neighbours::Kernel),
}

pub(crate) fn source(kernel: Kernel) -> String {
    match kernel {
        Kernel::Template(k) => template::source(k),
        Kernel::Neighbours(k) => neighbours::source(k),
    }
}
