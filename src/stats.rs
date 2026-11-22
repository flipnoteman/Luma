//! Statistics of `f32` and `f64` arrays (variance, weighted means, covariance, norms) and the
//! axis forms of the boolean reductions.
//!
//! Every statistic is built from reductions of expressions of the data, so the elementwise work
//! (the shift, the weights, the products of two arrays) is computed inside the reduction's first
//! pass and the data is read once: a variance or a mean and variance together is one pass
//! (`reduce([Sum, SumOfSquares])` of the shifted data), a covariance is the pending sums of
//! three expressions that the kernels of `jit::multireduce` record in one pass. The finishing
//! arithmetic is a few elementwise operations on one-element arrays (or on one value per line),
//! lazy like everything else.
//!
//! Variances are computed from sums of the data shifted by its first element (per line for the
//! axis forms), see [`Array::var_axis`]: the formula `(S2 - S1^2 / n) / (n - ddof)` is shift
//! invariant, and the shift removes the cancellation that an offset far from zero causes.

use std::panic::Location;

use crate::array::{Array, Failure};
use crate::error::LumaError;
use crate::gpu_type::{Numeric, ScalarType};
use crate::ops::{run_select, Cond, Rhs};
use crate::reduce::Reduction;
use crate::shape::Shape;

/// Which divisor a variance (or covariance) takes, as ArrayFire's `af_var_bias`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Bias {
    /// Divide by `n`, the number of elements (NumPy's `ddof = 0`; ArrayFire's default and
    /// `AF_VARIANCE_POPULATION`). With weights `n` is the sum of the weights.
    #[default]
    Population,
    /// Divide by `n - 1` (NumPy's `ddof = 1`, `AF_VARIANCE_SAMPLE`: the unbiased estimate of the
    /// variance of the distribution the elements were drawn from). With weights, `n` is the sum
    /// of the weights.
    Sample,
}

impl Bias {
    /// What is subtracted from the element count to get the divisor.
    fn ddof(self) -> f64 {
        match self {
            Bias::Population => 0.0,
            Bias::Sample => 1.0,
        }
    }
}

/// A norm for [`Array::norm_p`].
///
/// The vector norms take the array as one flat vector of all its elements; the matrix norms take
/// a two-dimensional array whose axis 0 indexes the rows and axis 1 the columns (as in
/// ArrayFire: the array of shape `[rows, cols]`).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Norm {
    /// The sum of the absolute values (`p = 1`).
    One,
    /// The Euclidean norm, the square root of the sum of squares (`p = 2`).
    Two,
    /// The largest absolute value (`p = infinity`). A NaN gives NaN (unlike [`Array::max`],
    /// which leaves NaNs out).
    Inf,
    /// `(sum |x|^p)^(1/p)`; `f32` arrays only (others poison the result with
    /// [`LumaError::Unsupported`]). `P(1.0)`, `P(2.0)` and `P(INFINITY)` are the norms above.
    P(f32),
    /// The Frobenius norm of a matrix: the square root of the sum of the squares of all its
    /// elements (the same value as [`Norm::Two`] of the flattened array).
    Frobenius,
    /// The induced 1-norm of a matrix: the largest sum of absolute values over its columns
    /// (lines along axis 0).
    MatrixOne,
    /// The induced infinity-norm of a matrix: the largest sum of absolute values over its rows
    /// (lines along axis 1).
    MatrixInf,
}

/// The first element of `x` (in memory order) as an array of shape `[1]`: a lazy slice that
/// fuses into whatever subtracts it.
fn first_cell<T: ScalarType>(x: &Array<T>) -> Array<T> {
    let mut cell = x.clone();
    for axis in 0..4 {
        cell = cell.slice(axis, ..1);
    }
    cell
}

/// `cond ? a : b` (a lazy select of the shape of `cond`).
fn pick<T: ScalarType>(cond: &Array<bool>, a: Rhs<'_, T>, b: Rhs<'_, T>, shape: Shape, location: &'static Location<'static>) -> Array<T> {
    Array::resolve(cond.context(), shape, run_select(Cond::Bool(cond), a, b, location), "select", location)
}

fn mismatch<T: ScalarType>(x: &Array<T>, other: &Array<T>, shape: Shape, name: &'static str, location: &'static Location<'static>) -> Array<T> {
    Array::resolve(x.context(), shape, Err(Failure::New(LumaError::ShapeMismatch { lhs: x.shape().dims(), rhs: other.shape().dims() })), name, location)
}

fn unsupported<T: ScalarType>(x: &Array<T>, what: &str, name: &'static str, location: &'static Location<'static>) -> Array<T> {
    Array::resolve(x.context(), Shape::from(1usize), Err(Failure::New(LumaError::Unsupported(what.to_owned()))), name, location)
}

/// The shape of a reduction of `shape` along `axis` (or of all of it).
fn reduced_shape(shape: Shape, axis: Option<usize>) -> Shape {
    match axis {
        Some(axis) => {
            let mut dims = shape.dims();
            if axis < 4 {
                dims[axis] = 1;
            }
            Shape::from(dims)
        }
        None => Shape::from(1usize),
    }
}

/// `v` clamped at 0 as `(v + |v|) / 2`: exact, and keeps NaN and infinity (a compare or `max`
/// would let a driver's NaN-dropping `fmax` turn a NaN into 0). Rounding can leave a variance
/// slightly negative.
macro_rules! clamp_at_zero {
    ($v:expr) => {{
        let v = $v;
        &(&v * 0.5) + &(&v.abs() * 0.5)
    }};
}

macro_rules! float_stats {
    ($t:ty, $pow:expr, $root:expr) => {
        impl Array<$t> {
            /// The largest element, NaN if any element is NaN (a plain [`Array::max`] leaves the
            /// NaNs out). The NaN test is a second reduction of the same expression, which the
            /// kernels of `jit::multireduce` record together with the maximum.
            #[track_caller]
            fn max_with_nan(&self) -> Array<$t> {
                let location = Location::caller();
                let flag = pick(&self.is_nan(), Rhs::Scalar(1.0 as $t), Rhs::Scalar(0.0 as $t), self.shape(), location);
                // (`f64` has no scalar comparisons: the zero to compare with is the flag times 0.)
                let flag = flag.max();
                let any_nan = flag.gt(&(&flag * 0.0));
                pick(&any_nan, Rhs::Scalar(<$t>::NAN), Rhs::Array(&self.max()), Shape::from(1usize), location)
            }

            /// The elements of `self` as a sum over the whole array (`axis` `None`) or along `axis`.
            fn sum_over(&self, axis: Option<usize>) -> Array<$t> {
                match axis {
                    Some(axis) => self.sum_axis(axis),
                    None => self.sum(),
                }
            }

            /// The cells reduced together: all elements, or the extent along `axis`.
            fn count_over(&self, axis: Option<usize>) -> usize {
                match axis {
                    Some(axis) => self.shape().dims().get(axis).copied().unwrap_or(1),
                    None => self.len(),
                }
            }

            /// The shift (the first element, of every line along `axis`) and the shifted data.
            fn shifted(&self, axis: Option<usize>) -> (Array<$t>, Array<$t>) {
                let shift = match axis {
                    Some(axis) => self.slice(axis, ..1),
                    None => first_cell(self),
                };
                let shifted = self - &shift;
                (shift, shifted)
            }

            /// `(sum, sum of squares)` of `self` over the whole array or along `axis`, one pass.
            fn two_moments(&self, axis: Option<usize>) -> (Array<$t>, Array<$t>) {
                match axis {
                    Some(axis) => {
                        let [s1, s2] = self.reduce_axis(axis, [Reduction::Sum, Reduction::SumOfSquares]);
                        (s1, s2)
                    }
                    None => {
                        let both = self.reduce([Reduction::Sum, Reduction::SumOfSquares]);
                        (both.slice(0, ..1), both.slice(0, 1..2))
                    }
                }
            }

            /// The mean and variance over the whole array or along `axis`, weighted or not.
            fn meanvar_at(&self, axis: Option<usize>, weights: Option<&Array<$t>>, bias: Bias, name: &'static str, location: &'static Location<'static>) -> (Array<$t>, Array<$t>) {
                let shape = reduced_shape(self.shape(), axis);
                let ddof = bias.ddof() as $t;
                let (shift, shifted) = self.shifted(axis);
                match weights {
                    None => {
                        let n = self.count_over(axis) as $t;
                        let (s1, s2) = shifted.two_moments(axis);
                        let mean = &shift + &(&s1 / n);
                        let var = &(&s2 - &(&(&s1 * &s1) / n)) / (n - ddof);
                        (mean, clamp_at_zero!(var))
                    }
                    Some(weights) => {
                        if weights.shape().dims().iter().zip(self.shape().dims()).any(|(&w, x)| w != x && w != 1) {
                            let bad = mismatch(self, weights, shape, name, location);
                            return (bad.clone(), bad);
                        }
                        let weights = if weights.shape() == self.shape() { weights.clone() } else { weights.broadcast_to(self.shape()) };
                        let weighted = &weights * &shifted;
                        let s0 = weights.sum_over(axis);
                        let s1 = weighted.sum_over(axis);
                        let s2 = (&weighted * &shifted).sum_over(axis);
                        let mean = &shift + &(&s1 / &s0);
                        let var = &(&s2 - &(&(&s1 * &s1) / &s0)) / &(&s0 - ddof);
                        (mean, clamp_at_zero!(var))
                    }
                }
            }

            /// The variance of all elements as a lazy one-element array (see [`Array::sum`]),
            /// with the divisor `bias` says. One pass over the data: the sum and the sum of
            /// squares of the elements minus the first one (computed inside the reduction,
            /// never stored) give `(S2 - S1^2 / n) / divisor`, which is shift invariant (see
            /// [`Array::var_axis`] for the accuracy: data far from zero with a spread
            /// comparable to its offset from the first element keeps its digits; a constant
            /// array has variance exactly 0). Rounding can leave the difference slightly
            /// negative, so the result is clamped at 0 (a NaN or infinity anywhere gives NaN).
            /// A single element has variance `0` with [`Bias::Population`] and NaN with
            /// [`Bias::Sample`].
            #[track_caller]
            pub fn var(&self, bias: Bias) -> Array<$t> {
                self.meanvar_at(None, None, bias, "var", Location::caller()).1
            }

            /// The standard deviation of all elements: the square root of [`Array::var`].
            #[track_caller]
            pub fn std(&self, bias: Bias) -> Array<$t> {
                self.meanvar_at(None, None, bias, "std", Location::caller()).1.sqrt()
            }

            /// The mean and the variance of all elements, as two lazy one-element arrays
            /// computed together from one pass over the data (see [`Array::var`]).
            #[track_caller]
            pub fn meanvar(&self, bias: Bias) -> (Array<$t>, Array<$t>) {
                self.meanvar_at(None, None, bias, "meanvar", Location::caller())
            }

            /// The weighted mean and variance of all elements: `weights` has the shape of
            /// `self` (or extents of 1 that are broadcast), the mean is `sum(w x) / sum(w)`, the
            /// variance `sum(w (x - mean)^2) / (sum(w) - ddof)` where `ddof` is 0 for
            /// [`Bias::Population`] and 1 for [`Bias::Sample`]. One pass over `self` and
            /// `weights` (the three sums are recorded in one kernel). A shape that does not
            /// match poisons both results with [`LumaError::ShapeMismatch`].
            #[track_caller]
            pub fn meanvar_weighted(&self, weights: &Array<$t>, bias: Bias) -> (Array<$t>, Array<$t>) {
                self.meanvar_at(None, Some(weights), bias, "meanvar_weighted", Location::caller())
            }

            /// The weighted mean of all elements, `sum(w x) / sum(w)`, as a lazy one-element
            /// array: `weights` has the shape of `self` (or extents of 1 that are broadcast).
            /// One pass over both (the two sums are recorded in one kernel). A shape that does
            /// not match poisons the result with [`LumaError::ShapeMismatch`].
            #[track_caller]
            pub fn mean_weighted(&self, weights: &Array<$t>) -> Array<$t> {
                self.weighted_mean_at(None, weights, "mean_weighted", Location::caller())
            }

            fn weighted_mean_at(&self, axis: Option<usize>, weights: &Array<$t>, name: &'static str, location: &'static Location<'static>) -> Array<$t> {
                if weights.shape().dims().iter().zip(self.shape().dims()).any(|(&w, x)| w != x && w != 1) {
                    return mismatch(self, weights, reduced_shape(self.shape(), axis), name, location);
                }
                let weights = if weights.shape() == self.shape() { weights.clone() } else { weights.broadcast_to(self.shape()) };
                &(&weights * self).sum_over(axis) / &weights.sum_over(axis)
            }

            /// The covariance of all elements of `self` and `y` (arrays of one shape, paired
            /// element by element), as a lazy one-element array:
            /// `sum((x - mean x) (y - mean y)) / divisor`, the divisor `n` or `n - 1` as `bias`
            /// says. One pass over both arrays: the sums of `x - x0`, `y - y0` and their
            /// product (`x0`, `y0` the first elements, which makes the formula
            /// shift invariant, see [`Array::var`]) are recorded in one kernel. Arrays of
            /// different shapes poison the result with [`LumaError::ShapeMismatch`].
            #[track_caller]
            pub fn cov(&self, y: &Array<$t>, bias: Bias) -> Array<$t> {
                self.cross_at(None, y, bias, false, "cov", Location::caller())
            }

            /// Pearson's correlation coefficient of all elements of `self` and `y` (arrays of
            /// one shape), as a lazy one-element array: the covariance over the square root of
            /// the product of the variances (the divisors cancel), in one pass over both arrays.
            /// It is NaN where an array is constant. Arrays of different shapes poison the result
            /// with [`LumaError::ShapeMismatch`].
            #[track_caller]
            pub fn corrcoef(&self, y: &Array<$t>) -> Array<$t> {
                self.cross_at(None, y, Bias::Population, true, "corrcoef", Location::caller())
            }

            fn cross_at(&self, axis: Option<usize>, y: &Array<$t>, bias: Bias, correlation: bool, name: &'static str, location: &'static Location<'static>) -> Array<$t> {
                if self.shape() != y.shape() {
                    return mismatch(self, y, reduced_shape(self.shape(), axis), name, location);
                }
                let n = self.count_over(axis) as $t;
                let (_, dx) = self.shifted(axis);
                let (_, dy) = y.shifted(axis);
                let sx = dx.sum_over(axis);
                let sy = dy.sum_over(axis);
                let sxy = (&dx * &dy).sum_over(axis);
                let cxy = &sxy - &(&(&sx * &sy) / n);
                if !correlation {
                    return &cxy / (n - bias.ddof() as $t);
                }
                let sxx = (&dx * &dx).sum_over(axis);
                let syy = (&dy * &dy).sum_over(axis);
                let cxx = &sxx - &(&(&sx * &sx) / n);
                let cyy = &syy - &(&(&sy * &sy) / n);
                &cxy / &(&cxx * &cyy).sqrt()
            }

            /// The norm `norm` of the array as a lazy one-element array. The absolute value, the
            /// power and the products are computed inside the reduction's first pass, so a norm
            /// reads the data once ([`Norm::Two`] and [`Norm::Frobenius`] are the passes of
            /// [`Array::norm`], with the square root in the last one). The sums are over
            /// all elements, they do not guard against overflow of the squares.
            ///
            /// A matrix norm needs an array of two dimensions at most (see [`Norm`]), else the
            /// result is poisoned with [`LumaError::InvalidShape`]; [`Norm::P`] of an array that is
            /// not `f32` with [`LumaError::Unsupported`].
            #[track_caller]
            pub fn norm_p(&self, norm: Norm) -> Array<$t> {
                let location = Location::caller();
                let matrix = |axis: usize| {
                    let dims = self.shape().dims();
                    if dims[2] != 1 || dims[3] != 1 {
                        return Array::resolve(
                            self.context(),
                            Shape::from(1usize),
                            Err(Failure::New(LumaError::InvalidShape(format!("matrix norm of an array of shape {}", self.shape())))),
                            "norm_p",
                            location,
                        );
                    }
                    self.abs().sum_axis(axis).max_with_nan()
                };
                let norm = match norm {
                    Norm::P(p) if p == 1.0 => Norm::One,
                    Norm::P(p) if p == 2.0 => Norm::Two,
                    Norm::P(p) if p == f32::INFINITY => Norm::Inf,
                    norm => norm,
                };
                match norm {
                    Norm::One => self.abs().sum(),
                    Norm::Two | Norm::Frobenius => self.reduce([Reduction::SumOfSquares]).sqrt(),
                    Norm::Inf => self.abs().max_with_nan(),
                    Norm::P(p) => match ($pow)(&self.abs(), p) {
                        Some(powered) => ($root)(&powered.sum(), p),
                        None => unsupported(self, "norm_p(Norm::P) needs an f32 array", "norm_p", location),
                    },
                    Norm::MatrixOne => matrix(0),
                    Norm::MatrixInf => matrix(1),
                }
            }
        }
    };
}

float_stats!(
    f32,
    |x: &Array<f32>, p: f32| Some(x.powf(p)),
    |s: &Array<f32>, p: f32| s.powf(1.0 / p)
);
float_stats!(f64, |_: &Array<f64>, _: f32| None::<Array<f64>>, |s: &Array<f64>, _: f32| s.clone());

/// The axis forms exist for `f32` arrays: reductions along an axis are not available for `f64`.
impl Array<f32> {
    /// The variance along `axis` with the divisor `bias` says; the result has extent 1
    /// along it. [`Array::var_axis`] is this with [`Bias::Population`].
    #[track_caller]
    pub fn var_axis_bias(&self, axis: usize, bias: Bias) -> Array<f32> {
        self.meanvar_at(Some(axis), None, bias, "var_axis_bias", Location::caller()).1
    }

    /// The standard deviation along `axis`: the square root of
    /// [`var_axis_bias`](Array::var_axis_bias).
    #[track_caller]
    pub fn std_axis_bias(&self, axis: usize, bias: Bias) -> Array<f32> {
        self.meanvar_at(Some(axis), None, bias, "std_axis_bias", Location::caller()).1.sqrt()
    }

    /// The mean and the variance along `axis` (extent 1 along it), from one pass over
    /// the data.
    #[track_caller]
    pub fn meanvar_axis(&self, axis: usize, bias: Bias) -> (Array<f32>, Array<f32>) {
        self.meanvar_at(Some(axis), None, bias, "meanvar_axis", Location::caller())
    }

    /// [`meanvar_weighted`](Array::meanvar_weighted) along `axis` (extent 1 along it).
    #[track_caller]
    pub fn meanvar_axis_weighted(&self, axis: usize, weights: &Array<f32>, bias: Bias) -> (Array<f32>, Array<f32>) {
        self.meanvar_at(Some(axis), Some(weights), bias, "meanvar_axis_weighted", Location::caller())
    }

    /// The weighted mean along `axis` (extent 1 along it), see
    /// [`mean_weighted`](Array::mean_weighted).
    #[track_caller]
    pub fn mean_axis_weighted(&self, axis: usize, weights: &Array<f32>) -> Array<f32> {
        self.weighted_mean_at(Some(axis), weights, "mean_axis_weighted", Location::caller())
    }

    /// The covariance along `axis` of `self` and `y` (extent 1 along it), see
    /// [`cov`](Array::cov).
    #[track_caller]
    pub fn cov_axis(&self, axis: usize, y: &Array<f32>, bias: Bias) -> Array<f32> {
        self.cross_at(Some(axis), y, bias, false, "cov_axis", Location::caller())
    }

    /// The correlation coefficient along `axis` (extent 1 along it), see
    /// [`corrcoef`](Array::corrcoef).
    #[track_caller]
    pub fn corrcoef_axis(&self, axis: usize, y: &Array<f32>) -> Array<f32> {
        self.cross_at(Some(axis), y, Bias::Population, true, "corrcoef_axis", Location::caller())
    }
}

impl Array<f64> {
    /// The mean of all elements, as a lazy one-element array: the sum divided by the length (see
    /// [`Array<f32>::mean`](Array::mean)).
    #[track_caller]
    pub fn mean(&self) -> Array<f64> {
        &self.sum() / (self.len() as f64)
    }
}

impl<T: Numeric> Array<T> {
    /// The product along `axis`; the result has extent 1 along it (`1` for a line of no
    /// elements). Fused like [`Array::sum_axis`]: a pending operand is computed inside the first
    /// pass of the reduction. See [`Array::prod`] for the semantics of float and integer
    /// products.
    #[track_caller]
    pub fn prod_axis(&self, axis: usize) -> Array<T> {
        let [product] = self.reduce_axis(axis, [Reduction::Product]);
        product
    }

    /// The product along `axis`: the same as [`Array::prod_axis`] (ArrayFire's name).
    #[track_caller]
    pub fn product_axis(&self, axis: usize) -> Array<T> {
        let [product] = self.reduce_axis(axis, [Reduction::Product]);
        product
    }
}

impl Array<bool> {
    /// The number of true elements along `axis`, as `u32`; the result has extent 1 along it.
    /// The same as [`Array<bool>::sum_axis`](Array::sum_axis) (ArrayFire's `count(x, dim)`).
    #[track_caller]
    pub fn count_axis(&self, axis: usize) -> Array<u32> {
        self.sum_axis(axis)
    }

    /// Whether every element along `axis` is true (ArrayFire's `allTrue(x, dim)`); the result
    /// has extent 1 along it. The same as [`Array<bool>::min_axis`](Array::min_axis).
    #[track_caller]
    pub fn all_axis(&self, axis: usize) -> Array<bool> {
        self.min_axis(axis)
    }

    /// Whether any element along `axis` is true (ArrayFire's `anyTrue(x, dim)`); the result has
    /// extent 1 along it. The same as [`Array<bool>::max_axis`](Array::max_axis).
    #[track_caller]
    pub fn any_axis(&self, axis: usize) -> Array<bool> {
        self.max_axis(axis)
    }
}
