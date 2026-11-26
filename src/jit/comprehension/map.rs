//! Affine maps: the point of a comprehension's space to the coordinates of a source.
//!
//! A [`ViewMap`] is exactly an affine map with at most one non-zero coefficient per source row, four rows and four output axes;
//! an [`AffineMap`] drops the "at most one" (a window reads `stride * o + t - pad`, two coefficients) and the four output axes
//! (the space has up to [`MAX_DIMS`] dims).

use super::{MAX_DIMS, MAX_ROWS};
use crate::jit::{AxisMap, ViewMap};

/// Source coordinate along one axis: `offset + sum(coeff[d] * point[d])`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct AffineRow {
    pub(crate) coeff: [i32; MAX_DIMS],
    pub(crate) offset: i64,
}

impl AffineRow {
    pub(crate) const ZERO: AffineRow = AffineRow { coeff: [0; MAX_DIMS], offset: 0 };

    /// The coordinate of dim `d` itself.
    pub(crate) const fn dim(d: usize) -> AffineRow {
        let mut row = AffineRow::ZERO;
        row.coeff[d] = 1;
        row
    }

    /// The dims with a non-zero coefficient.
    fn used(&self) -> impl Iterator<Item = (usize, i32)> + '_ {
        self.coeff.iter().copied().enumerate().filter(|&(_, c)| c != 0)
    }
}

/// Row `a` is the source coordinate along axis `a`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct AffineMap {
    pub(crate) rows: [AffineRow; MAX_ROWS],
}

impl AffineMap {
    pub(crate) const ZERO: AffineMap = AffineMap { rows: [AffineRow::ZERO; MAX_ROWS] };

    /// Rows `0..dims.len()` follow the dims `dims[row]` with coefficient 1, the others are 0.
    pub(crate) const fn following<const N: usize>(dims: [usize; N]) -> AffineMap {
        let mut map = AffineMap::ZERO;
        let mut row = 0;
        while row < N {
            map.rows[row] = AffineRow::dim(dims[row]);
            row += 1;
        }
        map
    }

    /// The map of a [`ViewMap`]: `axes[o]` is the space dim output axis `o` stands for. A row whose step is 0 has no coefficient.
    pub(crate) fn from_view(view: &ViewMap, axes: [u8; 4]) -> AffineMap {
        let mut map = AffineMap::ZERO;
        for (row, axis) in map.rows.iter_mut().zip(&view.axes) {
            row.offset = i64::from(axis.start);
            if axis.step != 0 {
                row.coeff[usize::from(axes[usize::from(axis.from)])] = axis.step;
            }
        }
        map
    }

    /// The view of this map over `src` and `out` extents; `axes[o]` is the space dim output axis `o` stands for. `None` when a row
    /// has more than one non-zero coefficient, or a coefficient on a dim no output axis stands for, or [`ViewMap::fits`] fails.
    /// A row without coefficient is a step 0 on its own axis (the convention of `broadcast_view`).
    pub(crate) fn to_view(self, axes: [Option<u8>; 4], src: [u32; 4], out: [u32; 4]) -> Option<ViewMap> {
        let mut view = ViewMap { src, out, axes: [AxisMap { from: 0, start: 0, step: 0 }; 4] };
        for (a, (row, slot)) in self.rows.iter().zip(&mut view.axes).enumerate() {
            let start = i32::try_from(row.offset).ok()?;
            let mut used = row.used();
            *slot = match (used.next(), used.next()) {
                (None, _) => AxisMap { from: a as u8, start, step: 0 },
                (Some((d, step)), None) => {
                    let from = axes.iter().position(|&axis| axis == Some(d as u8))?;
                    AxisMap { from: from as u8, start, step }
                }
                _ => return None,
            };
        }
        view.fits().then_some(view)
    }

    /// This map applied to the point `inner` gives: the coordinates `inner` produces are the dims of this map's space (row `d` of
    /// `inner` is the coordinate of dim `d`, the way `outer.src == self.out` composes two views). `None` if a dim past the rows of
    /// `inner` has a coefficient, or an `i32`/`i64` overflows.
    pub(crate) fn then(&self, inner: &AffineMap) -> Option<AffineMap> {
        let mut out = AffineMap::ZERO;
        for (row, slot) in self.rows.iter().zip(&mut out.rows) {
            let mut offset = row.offset;
            for (d, c) in row.used() {
                let source = inner.rows.get(d)?;
                offset = offset.checked_add(i64::from(c).checked_mul(source.offset)?)?;
                for (acc, &s) in slot.coeff.iter_mut().zip(&source.coeff) {
                    *acc = i32::try_from(i64::from(*acc).checked_add(i64::from(c).checked_mul(i64::from(s))?)?).ok()?;
                }
            }
            slot.offset = offset;
        }
        Some(out)
    }

    /// The flat index of the source element (read row-major, axis 0 fastest, of an array of `shape`) as the sum of one coefficient
    /// per dim (for `Edge::None`, where the rows are not folded back into range), and its constant part.
    pub(crate) fn strides(&self, shape: [u32; 4]) -> ([i64; MAX_DIMS], i64) {
        let mut strides = [0i64; MAX_DIMS];
        let mut base = 0i64;
        let mut axis_stride = 1i64;
        for (row, &extent) in self.rows.iter().zip(&shape) {
            for (acc, &c) in strides.iter_mut().zip(&row.coeff) {
                *acc += i64::from(c) * axis_stride;
            }
            base += row.offset * axis_stride;
            axis_stride *= i64::from(extent);
        }
        (strides, base)
    }

    /// The least and greatest coordinate of row `a` over a space of the extents `extents` (dim `d` runs `0..extents[d]`), `None`
    /// if the space is empty. Exact (the sums are in `i128`).
    pub(crate) fn row_range(&self, a: usize, extents: &[u64]) -> Option<(i128, i128)> {
        let row = &self.rows[a];
        let (mut lo, mut hi) = (i128::from(row.offset), i128::from(row.offset));
        for (&c, &extent) in row.coeff.iter().zip(extents.iter().chain(std::iter::repeat(&1))) {
            if extent == 0 {
                return None;
            }
            let reach = i128::from(c) * (i128::from(extent) - 1);
            lo += reach.min(0);
            hi += reach.max(0);
        }
        Some((lo, hi))
    }
}

/// A map that most values share (an API reduction's identity map is a constant), or one of its own.
#[derive(Clone, Debug)]
pub(crate) enum MapRef {
    Static(&'static AffineMap),
    Owned(Box<AffineMap>),
}

impl MapRef {
    pub(crate) fn owned(map: AffineMap) -> MapRef {
        MapRef::Owned(Box::new(map))
    }
}

impl std::ops::Deref for MapRef {
    type Target = AffineMap;

    #[inline(always)]
    fn deref(&self) -> &AffineMap {
        match self {
            MapRef::Static(map) => map,
            MapRef::Owned(map) => map,
        }
    }
}

impl PartialEq for MapRef {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Eq for MapRef {}
