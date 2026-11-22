//! The complex single-precision element type [`C32`].

use bytemuck::{Pod, Zeroable};

use crate::gpu_type::{ScalarType, sealed};

/// A complex number of two `f32`, stored as `(re, im)` (8 bytes, the layout of WGSL's
/// `vec2<f32>`, of `cufftComplex` and of the C99 `float _Complex`).
///
/// `Array<C32>` is the array of complex numbers the FFTs ([`Array::fft`](crate::Array::fft) and
/// friends) work on; it has `+ - * /`, negation, `conj`, `real`, `imag`, `abs` and `arg`
/// like the other element types, lazy and fused. The host arithmetic of `C32` (the operators,
/// [`C32::abs`], [`C32::arg`]) exists for building and checking arrays on the host.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct C32 {
    /// The real part.
    pub re: f32,
    /// The imaginary part.
    pub im: f32,
}

// SAFETY: `#[repr(C)]` over two `f32`: no padding, every bit pattern valid.
unsafe impl Zeroable for C32 {}
// SAFETY: see above.
unsafe impl Pod for C32 {}

impl sealed::Sealed for C32 {}

impl C32 {
    /// Zero.
    pub const ZERO: C32 = C32 { re: 0.0, im: 0.0 };
    /// One.
    pub const ONE: C32 = C32 { re: 1.0, im: 0.0 };
    /// The imaginary unit.
    pub const I: C32 = C32 { re: 0.0, im: 1.0 };

    /// `re + im i`.
    pub const fn new(re: f32, im: f32) -> Self {
        C32 { re, im }
    }

    /// `r (cos theta + i sin theta)`.
    pub fn from_polar(r: f32, theta: f32) -> Self {
        C32 { re: r * theta.cos(), im: r * theta.sin() }
    }

    /// The complex conjugate.
    pub fn conj(self) -> Self {
        C32 { re: self.re, im: -self.im }
    }

    /// The modulus `sqrt(re^2 + im^2)`.
    pub fn abs(self) -> f32 {
        self.re.hypot(self.im)
    }

    /// The argument `atan2(im, re)`, in `[-pi, pi]`.
    pub fn arg(self) -> f32 {
        self.im.atan2(self.re)
    }
}

impl From<f32> for C32 {
    fn from(re: f32) -> Self {
        C32 { re, im: 0.0 }
    }
}

impl From<(f32, f32)> for C32 {
    fn from((re, im): (f32, f32)) -> Self {
        C32 { re, im }
    }
}

impl std::ops::Add for C32 {
    type Output = C32;
    fn add(self, o: C32) -> C32 {
        C32 { re: self.re + o.re, im: self.im + o.im }
    }
}

impl std::ops::Sub for C32 {
    type Output = C32;
    fn sub(self, o: C32) -> C32 {
        C32 { re: self.re - o.re, im: self.im - o.im }
    }
}

impl std::ops::Mul for C32 {
    type Output = C32;
    fn mul(self, o: C32) -> C32 {
        C32 { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
}

impl std::ops::Div for C32 {
    type Output = C32;
    fn div(self, o: C32) -> C32 {
        let d = o.re * o.re + o.im * o.im;
        C32 { re: (self.re * o.re + self.im * o.im) / d, im: (self.im * o.re - self.re * o.im) / d }
    }
}

impl std::ops::Neg for C32 {
    type Output = C32;
    fn neg(self) -> C32 {
        C32 { re: -self.re, im: -self.im }
    }
}

impl ScalarType for C32 {
    fn wgsl_type() -> &'static str {
        "vec2<f32>"
    }

    fn reduce_type() -> &'static str {
        "c32"
    }
}
