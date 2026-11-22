//! Connected-component labelling, confidence connected components, deconvolution.

mod label;

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
}

pub(crate) fn source(kernel: Kernel) -> String {
    match kernel {
        Kernel::Label(k) => label::source(k),
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
