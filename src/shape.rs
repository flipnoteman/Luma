use std::fmt;

use crate::error::{LumaError, Result};

/// The extent of an [`Array`](crate::Array) in up to four dimensions.
///
/// Stored as four extents, the first dimension varying fastest in memory; missing trailing
/// dimensions are 1. Anything that is the shape of an array converts into it, so constructors
/// take `3`, `[2, 3]`, `[2, 3, 4]` or `[2, 3, 4, 5]` directly:
///
/// ```
/// use luma::Shape;
/// assert_eq!(Shape::from(3), Shape::from([3, 1, 1, 1]));
/// assert_eq!(Shape::from([2, 3]).len(), 6);
/// ```
///
/// A `Shape` itself is not validated; constructors reject zero-sized dimensions and shapes that
/// do not fit `u32`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Shape([usize; 4]);

impl Shape {
    /// A shape from all four extents.
    pub const fn new(dims: [usize; 4]) -> Shape {
        Shape(dims)
    }

    /// The four extents (trailing dimensions are 1).
    pub const fn dims(&self) -> [usize; 4] {
        self.0
    }

    /// The number of elements (the product of the extents), saturating on overflow.
    pub fn len(&self) -> usize {
        self.0.iter().fold(1usize, |acc, &d| acc.saturating_mul(d))
    }

    /// `true` if some extent is zero (such a shape cannot be used for an array).
    pub fn is_empty(&self) -> bool {
        self.0.contains(&0)
    }

    /// Checks the shape against what the GPU kernels support: every extent is non-zero, and every
    /// extent and the element count fit `u32`. Returns the element count.
    pub(crate) fn validate(&self) -> Result<usize> {
        let too_large = || LumaError::DimensionTooLarge { shape: self.0 };
        let mut len = 1u32;
        for &extent in &self.0 {
            if extent == 0 {
                return Err(LumaError::InvalidShape(format!(
                    "Shape {:?} has a zero-sized dimension",
                    self.0
                )));
            }
            let extent = u32::try_from(extent).map_err(|_| too_large())?;
            len = len.checked_mul(extent).ok_or_else(too_large)?;
        }
        Ok(len as usize)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl PartialEq<[usize; 4]> for Shape {
    fn eq(&self, other: &[usize; 4]) -> bool {
        self.0 == *other
    }
}

impl From<usize> for Shape {
    fn from(n: usize) -> Shape {
        Shape([n, 1, 1, 1])
    }
}

impl From<[usize; 1]> for Shape {
    fn from(d: [usize; 1]) -> Shape {
        Shape([d[0], 1, 1, 1])
    }
}

impl From<[usize; 2]> for Shape {
    fn from(d: [usize; 2]) -> Shape {
        Shape([d[0], d[1], 1, 1])
    }
}

impl From<[usize; 3]> for Shape {
    fn from(d: [usize; 3]) -> Shape {
        Shape([d[0], d[1], d[2], 1])
    }
}

impl From<[usize; 4]> for Shape {
    fn from(d: [usize; 4]) -> Shape {
        Shape(d)
    }
}

/// Per-dimension offsets of [`Array::shift`](crate::Array::shift). Converts from an `isize` and
/// from `[isize; 1]` ..= `[isize; 4]`; missing dimensions are not shifted.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Offsets(pub(crate) [isize; 4]);

impl From<isize> for Offsets {
    fn from(n: isize) -> Offsets {
        Offsets([n, 0, 0, 0])
    }
}

impl From<[isize; 1]> for Offsets {
    fn from(d: [isize; 1]) -> Offsets {
        Offsets([d[0], 0, 0, 0])
    }
}

impl From<[isize; 2]> for Offsets {
    fn from(d: [isize; 2]) -> Offsets {
        Offsets([d[0], d[1], 0, 0])
    }
}

impl From<[isize; 3]> for Offsets {
    fn from(d: [isize; 3]) -> Offsets {
        Offsets([d[0], d[1], d[2], 0])
    }
}

impl From<[isize; 4]> for Offsets {
    fn from(d: [isize; 4]) -> Offsets {
        Offsets(d)
    }
}
