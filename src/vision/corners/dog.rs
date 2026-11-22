//! The difference of Gaussians (`Array::dog`).

use std::panic::Location;

use crate::array::{Array, Failure};
use crate::image::{gaussian_kernel_1d_on, ConvSpec};

impl Array<f32> {
    /// The difference of Gaussians of every `[H, W]` plane (axes 2 and 3 are a batch;
    /// ArrayFire's `dog`): the image blurred with the Gaussian kernel of `2 * radius1 + 1`
    /// cells a side minus the image blurred with the one of `2 * radius2 + 1` cells. A kernel of
    /// `n` cells a side is [`gaussian_kernel`](crate::gaussian_kernel)'s: the product of two
    /// Gaussians of standard deviation `0.25 * n` centred on the middle, summing to 1. The
    /// cells outside the image are zeros, so the result has the shape of the input.
    ///
    /// A radius of 0 is the identity kernel, equal radii give zeros. Differences from
    /// ArrayFire: the radii are unsigned (ArrayFire takes `int` and returns garbage for a
    /// negative one), any radius is accepted (ArrayFire's CUDA backend fails above 8), and only
    /// `f32` images are taken (ArrayFire's `dog` takes every image type).
    ///
    /// Each blur is two separable passes ([`Array::conv2_sep`]); the result is lazy like any
    /// other operation and a failure poisons it.
    ///
    /// ```
    /// use luma::Array;
    /// # fn main() -> Result<(), luma::LumaError> {
    /// let img = Array::from_slice(&[1.0f32; 25], [5, 5])?;
    /// // A constant image has no detail away from the border (the border is zero padded).
    /// let d = img.dog(1, 2).to_vec()?;
    /// assert!(d[12].abs() < 1e-6);
    /// # Ok(()) }
    /// ```
    #[track_caller]
    pub fn dog(&self, radius1: usize, radius2: usize) -> Array<f32> {
        let location = Location::caller();
        let result = (|| {
            let ctx = self.context();
            let blur = |radius: usize| -> Result<Array<f32>, Failure> {
                let taps = gaussian_kernel_1d_on(ctx, 2 * radius + 1, 0.0)?;
                Ok(self.conv2_sep(&taps, &taps, ConvSpec::new()))
            };
            let (first, second) = (blur(radius1)?, blur(radius2)?);
            Ok(&first - &second)
        })();
        Array::resolve(self.context(), self.shape(), result, "dog", location)
    }
}
