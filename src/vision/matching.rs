//! Template matching, nearest-neighbour and Hamming matchers.
//!
//! [`Array::match_template`] slides a template over an image and scores every window with one of
//! the metrics of [`MatchType`]; [`nearest_neighbour`] and [`hamming_matcher`] find, for every
//! query feature, the `k` closest of a set of train features.

mod template;

use std::panic::Location;

use crate::array::Array;

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

/// One kernel of this module (part of the key of its cache entry in the context).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Kernel {
    /// [`template`].
    Template(template::Kernel),
}

pub(crate) fn source(kernel: Kernel) -> String {
    match kernel {
        Kernel::Template(k) => template::source(k),
    }
}
