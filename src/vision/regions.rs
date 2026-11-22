//! Connected-component labelling, confidence connected components, deconvolution.

mod confidence;
mod deconv;
mod label;

pub use deconv::{InverseDeconvAlgo, IterativeDeconvAlgo};

use std::panic::Location;

use crate::array::Array;

/// The neighbours of a pixel that connect it to a component (ArrayFire's `af_connectivity`).
///
/// ```
/// use luma::Connectivity;
/// assert_ne!(Connectivity::Four, Connectivity::Eight);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Connectivity {
    /// The 4 neighbours that share an edge.
    Four,
    /// The 8 neighbours that share an edge or a corner.
    Eight,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Kernel {
    Label(label::Kernel),
    Confidence(confidence::Kernel),
}

pub(crate) fn source(kernel: Kernel) -> String {
    match kernel {
        Kernel::Label(k) => label::source(k),
        Kernel::Confidence(k) => confidence::source(k),
    }
}

impl Array<bool> {
    /// The connected components of a binary `[H, W]` image (ArrayFire's `regions`): an array of the
    /// same shape holding 0 for the background and, for a foreground pixel, the number of its
    /// component, `1..=N`.
    ///
    /// Components are numbered in the order of their first pixel in memory order (axis 0 fastest,
    /// then axis 1), which is the numbering of both ArrayFire backends, so the labels are equal to
    /// ArrayFire's, not just the partition.
    ///
    /// Differences from ArrayFire: the labels are `u32` (ArrayFire converts to `u32`, `i32`, `u16`,
    /// `f32`, `f64`: use [`Array::to_f32`](crate::Array) and the like on the result), and the image must have
    /// fewer than 2^32 - 1 pixels. Like ArrayFire the image must be 2-dimensional.
    ///
    /// The operation is lazy-shaped (no synchronization): six launches, union-find in workgroup
    /// memory per tile of 32 x 8 pixels, then across the tile borders in global memory.
    ///
    /// ```
    /// use luma::{Array, Connectivity};
    /// # fn main() -> Result<(), luma::LumaError> {
    /// // Axis 0 fastest: the columns of the 3 x 3 image are [1, 0, 1], [0, 0, 0], [0, 1, 1].
    /// let img = Array::from_slice(&[true, false, true, false, false, false, false, true, true], [3, 3])?;
    /// let four = img.regions(Connectivity::Four);
    /// assert_eq!(four.to_vec()?, [1, 0, 2, 0, 0, 0, 0, 3, 3]);
    /// # Ok(()) }
    /// ```
    #[track_caller]
    pub fn regions(&self, connectivity: Connectivity) -> Array<u32> {
        let location = Location::caller();
        let result = label::run(self, connectivity);
        Array::resolve(self.context(), self.shape(), result, "regions", location)
    }
}

impl Array<f32> {
    /// Confidence connected components of an `[H, W]` image grown from seed pixels (ArrayFire's
    /// `confidenceCC`): an `f32` image of the same shape holding `segmented_value` where a pixel
    /// belongs to the segmentation and 0 elsewhere.
    ///
    /// `seeds_x[i]` and `seeds_y[i]` are the indices of seed `i` along **axis 0 and axis 1**
    /// (rows and columns; like ArrayFire's `confidenceCC` and unlike the `x` of
    /// [`Keypoints`], so pass `keypoints.y` as `seeds_x`). The image is scaled linearly to
    /// `[1, 2]`; mean and standard deviation of the pixels in the `(2 radius + 1)^2` windows around the seeds
    /// give the interval `[mean - multiplier sd, mean + multiplier sd]`, which is widened to contain
    /// the seed values; the segmentation is the set of pixels in the interval that are
    /// 8-connected to a seed through such pixels. Each of the `iterations` further rounds
    /// re-estimates mean and standard deviation from the pixels of the current segmentation and
    /// segments again from the seeds. The rounds stop early if the segmentation is empty or a
    /// variance is below 1e-6, as in ArrayFire.
    ///
    /// Differences from ArrayFire (whose behaviour here is partly undefined): a seed window that
    /// reaches over the border is clipped to the image (ArrayFire reads outside its summed-area
    /// table and gets garbage: a segmentation of the seeds only, or of the whole image), the
    /// statistics of several seeds are those of the union of their windows (ArrayFire sums over all
    /// pairs of seed rows and columns, which makes its first segmentation just the seeds), the
    /// widening uses the values of the seeds (ArrayFire's the values at all pairs of rows and
    /// columns) and seeds outside the image are ignored. Sums are accumulated from shifted values
    /// per workgroup, so the intervals differ from ArrayFire's in the last digits. The input is
    /// `f32` (convert integer images with the `to_f32` methods; the result has ArrayFire's type for
    /// `f32` input). An image without any pixel of different value is segmented whole.
    ///
    /// No synchronization: the rounds are launches on the device, and rounds after the final one
    /// are no-ops there. It takes six launches per round, a labelling of the pixels in the
    /// interval ([`Array::regions`]'s union-find).
    ///
    /// ```
    /// use luma::Array;
    /// # fn main() -> Result<(), luma::LumaError> {
    /// // A 4 x 4 image: a bright 2 x 2 block (value 10) on a dark background (value 1).
    /// let mut data = vec![1.0f32; 16];
    /// for (i, j) in [(1, 1), (2, 1), (1, 2), (2, 2)] { data[i + 4 * j] = 10.0; }
    /// let img = Array::from_slice(&data, [4, 4])?;
    /// let sx = Array::from_slice(&[1u32], [1])?;
    /// let sy = Array::from_slice(&[1u32], [1])?;
    /// let seg = img.confidence_cc(&sx, &sy, 0, 1, 1, 255.0);
    /// let mut want = vec![0.0f32; 16];
    /// for (i, j) in [(1, 1), (2, 1), (1, 2), (2, 2)] { want[i + 4 * j] = 255.0; }
    /// assert_eq!(seg.to_vec()?, want);
    /// # Ok(()) }
    /// ```
    #[track_caller]
    pub fn confidence_cc(&self, seeds_x: &Array<u32>, seeds_y: &Array<u32>, radius: u32, multiplier: u32, iterations: u32, segmented_value: f32) -> Array<f32> {
        let location = Location::caller();
        let result = confidence::run(self, seeds_x, seeds_y, radius, multiplier, iterations).map(|seg| seg.to_f32() * segmented_value);
        Array::resolve(self.context(), self.shape(), result, "confidence_cc", location)
    }

    /// [`Array::confidence_cc`] with the seeds as the columns of an `[N, 2]` array (`[:, 0]` along
    /// axis 0, `[:, 1]` along axis 1), ArrayFire's `confidenceCC(in, seeds, ...)`.
    ///
    /// ```
    /// use luma::Array;
    /// # fn main() -> Result<(), luma::LumaError> {
    /// let img = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], [2, 2])?;
    /// let seeds = Array::from_slice(&[0u32, 1], [1, 2])?;
    /// assert_eq!(img.confidence_cc_seeds(&seeds, 0, 1, 0, 1.0).to_vec()?, [1.0; 4]);
    /// # Ok(()) }
    /// ```
    #[track_caller]
    pub fn confidence_cc_seeds(&self, seeds: &Array<u32>, radius: u32, multiplier: u32, iterations: u32, segmented_value: f32) -> Array<f32> {
        let location = Location::caller();
        let d = seeds.shape().dims();
        if d[1] != 2 || d[2] != 1 || d[3] != 1 {
            let error = std::sync::Arc::new(crate::LumaError::InvalidShape(format!("confidence_cc_seeds takes an [N, 2] array of seeds, got {d:?}")));
            return Array::poisoned(self.context(), self.shape(), std::sync::Arc::new(crate::LumaError::Op { op: "confidence_cc_seeds", location, source: error }));
        }
        let n = d[0];
        self.confidence_cc(&seeds.slice(1, 0..1).reshape([n]), &seeds.slice(1, 1..2).reshape([n]), radius, multiplier, iterations, segmented_value)
    }

    /// [`Array::confidence_cc`] with the seeds given on the host (ArrayFire's overload with
    /// `num_seeds`, `seedx`, `seedy`; a single point is a slice of one).
    ///
    /// ```
    /// use luma::Array;
    /// # fn main() -> Result<(), luma::LumaError> {
    /// let img = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], [2, 2])?;
    /// assert_eq!(img.confidence_cc_at(&[0], &[1], 0, 1, 0, 2.0)?.to_vec()?, [2.0; 4]);
    /// # Ok(()) }
    /// ```
    #[track_caller]
    pub fn confidence_cc_at(&self, seeds_x: &[u32], seeds_y: &[u32], radius: u32, multiplier: u32, iterations: u32, segmented_value: f32) -> crate::Result<Array<f32>> {
        let ctx = self.context();
        if seeds_x.is_empty() || seeds_x.len() != seeds_y.len() {
            return Err(crate::LumaError::InvalidShape(format!("confidence_cc_at needs as many seed rows as columns and at least one, got {} and {}", seeds_x.len(), seeds_y.len())));
        }
        let sx = Array::from_slice_on(ctx, seeds_x, [seeds_x.len()])?;
        let sy = Array::from_slice_on(ctx, seeds_y, [seeds_y.len()])?;
        Ok(self.confidence_cc(&sx, &sy, radius, multiplier, iterations, segmented_value))
    }
}

impl Array<f32> {
    /// Iterative deconvolution of an `[H, W]` image with the point spread function `psf`
    /// (`[PH, PW]`, centre `(PH / 2, PW / 2)`; ArrayFire's `iterativeDeconv`): `iterations` steps of
    /// [`IterativeDeconvAlgo::Landweber`] (`x += relaxation * P^T (y - P x)`) or
    /// [`IterativeDeconvAlgo::RichardsonLucy`] (`x *= P^T (y / (P x))`, `relaxation` unused), from
    /// the blurred image as the first estimate. `iterations` and `relaxation` must be positive.
    ///
    /// The image is padded to `H + PH` x `W + PW`, each rounded up to a size whose prime factors
    /// are at most 13, with its edge pixels; the convolutions are circular over the padded
    /// image (see the guide).
    ///
    /// Differences from ArrayFire: the Landweber result is the true estimate, while ArrayFire's is
    /// `N0 * N1` times larger (the padded pixel count: it skips the scaling of its last inverse
    /// transform); Richardson-Lucy treats a non-positive blurred estimate as ratio 0 (ArrayFire's
    /// division gives inf or NaN that spreads over the whole image); `f32` input only (convert
    /// integer images with `to_f32`).
    ///
    /// ```
    /// use luma::{Array, IterativeDeconvAlgo};
    /// # fn main() -> Result<(), luma::LumaError> {
    /// // A point spread function that is a single 1 at its centre does not blur: nothing changes.
    /// let img = Array::from_slice(&(0..20).map(|i| 1.0 + i as f32).collect::<Vec<_>>(), [4, 5])?;
    /// let psf = Array::from_slice(&[0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [3, 3])?;
    /// let sharp = img.iterative_deconv(&psf, 3, 0.5, IterativeDeconvAlgo::Landweber);
    /// let got = sharp.to_vec()?;
    /// let want = img.to_vec()?;
    /// assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-4));
    /// # Ok(()) }
    /// ```
    #[track_caller]
    pub fn iterative_deconv(&self, psf: &Array<f32>, iterations: u32, relaxation: f32, algo: IterativeDeconvAlgo) -> Array<f32> {
        let location = Location::caller();
        let result = deconv::iterative(self, psf, iterations, relaxation, algo);
        Array::resolve(self.context(), self.shape(), result, "iterative_deconv", location)
    }

    /// Inverse deconvolution (ArrayFire's `inverseDeconv`): with [`InverseDeconvAlgo::Tikhonov`]
    /// `X = Y conj(P) / (|P|^2 + gamma)` per frequency of the padded image (padding as in
    /// [`Array::iterative_deconv`]). `gamma` must be positive.
    ///
    /// ArrayFire's result is `N0 * N1` times larger (the padded pixel count, an inverse transform
    /// left unscaled); luma returns the properly scaled estimate.
    ///
    /// ```
    /// use luma::{Array, InverseDeconvAlgo};
    /// # fn main() -> Result<(), luma::LumaError> {
    /// let img = Array::from_slice(&(0..20).map(|i| 1.0 + i as f32).collect::<Vec<_>>(), [4, 5])?;
    /// let psf = Array::from_slice(&[0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [3, 3])?;
    /// // With a unit psf the estimate is the image divided by 1 + gamma.
    /// let got = img.inverse_deconv(&psf, 0.5, InverseDeconvAlgo::Tikhonov).to_vec()?;
    /// let want: Vec<f32> = img.to_vec()?.iter().map(|v| v / 1.5).collect();
    /// assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-4));
    /// # Ok(()) }
    /// ```
    #[track_caller]
    pub fn inverse_deconv(&self, psf: &Array<f32>, gamma: f32, algo: InverseDeconvAlgo) -> Array<f32> {
        let location = Location::caller();
        let result = deconv::inverse(self, psf, gamma, algo);
        Array::resolve(self.context(), self.shape(), result, "inverse_deconv", location)
    }
}
