//! Matrix properties of values that are known from the construction that made them (W3-R6).
//!
//! Properties are facts about the entries of the matrix on dimensions 0 and 1 (dimensions 2 and 3 are batches: the fact holds for
//! every matrix of the batch): entries outside the pattern are exact zeros for finite inputs (a `-0`, or a NaN made by `inf * 0`
//! inside a product, is outside the claim). They come only from exact constructions: the identity, a diagonal matrix, a triangle,
//! the factors of a decomposition that are triangles, and the few operations that keep a pattern (see the table in the spec).
//!
//! **Orientation.** The flags are stated in the convention of the dense routines ([`crate::matmul`], `solve`, `lu`,
//! `solve_triangular`): a matrix is `[cols, rows]`, so the *row* is dimension 1 and the *column* dimension 0. `LOWER` means entries
//! `(c, r)` with `c > r` are zero, `UPPER` those with `c < r`; the band `(lo, hi)` counts the sub-diagonals (`r > c`) and the
//! super-diagonals (`c > r`). This is the transpose of the reading of `src/structure.rs` (whose `lower` keeps `j <= i` with `i`
//! along dimension 0): `Array::lower` is `UPPER` here, and `diag_matrix(k)` for `k > 0` is `LOWER` with `lo = k`.
//!
//! The handle carries a compact copy ([`HandleProps`]) on lazy and structured arrays only: a valid array's buffer is shared by its
//! clones, so a write through one handle would leave the others' property stale.

use std::sync::atomic::{AtomicU32, Ordering};
#[cfg(debug_assertions)]
use std::sync::atomic::AtomicU64;

use crate::jit::comprehension::Props;

/// Properties lost to sharing so far (debug builds).
#[cfg(debug_assertions)]
static LOST: AtomicU64 = AtomicU64::new(0);
/// Uses of a claim so far (debug builds): one in 16 is re-derived from the values.
#[cfg(debug_assertions)]
static USES: AtomicU32 = AtomicU32::new(0);

/// The pattern bits (the ones about where the zeros are).
const PATTERN: u16 = Props::DIAGONAL | Props::LOWER | Props::UPPER | Props::SYMMETRIC;

impl Props {
    /// Whether every bit of `bits` is set.
    pub(crate) fn has(self, bits: u16) -> bool {
        self.flags & bits == bits
    }

    /// Whether no fact is known.
    pub(crate) fn is_none(self) -> bool {
        self.flags == 0 && self.band.is_none()
    }

    /// The facts the others imply: `IDENTITY` is `DIAGONAL`, `DIAGONAL` is `LOWER` and `UPPER` and a band of `(0, 0)` (and the
    /// reverse), a band with `hi = 0` is `LOWER`, with `lo = 0` is `UPPER`, and `BANDED` is exactly "the band is known".
    pub(crate) fn closed(mut self) -> Props {
        for _ in 0..2 {
            if self.has(Props::IDENTITY) {
                self.flags |= Props::DIAGONAL;
            }
            if self.has(Props::LOWER | Props::UPPER) {
                self.flags |= Props::DIAGONAL;
            }
            if self.has(Props::DIAGONAL) {
                self.flags |= Props::LOWER | Props::UPPER;
                self.band = Some((0, 0));
            }
            if let Some((lo, hi)) = self.band {
                if hi == 0 {
                    self.flags |= Props::LOWER;
                }
                if lo == 0 {
                    self.flags |= Props::UPPER;
                }
            }
            if self.has(Props::LOWER) && self.band.is_some_and(|(_, hi)| hi != 0) || self.has(Props::UPPER) && self.band.is_some_and(|(lo, _)| lo != 0) {
                // Contradictory (a bug of the caller): know nothing rather than something wrong.
                return Props::NONE;
            }
        }
        match self.band {
            Some(_) => self.flags |= Props::BANDED,
            None => self.flags &= !Props::BANDED,
        }
        self
    }

    /// The identity of a matrix with `square` shape (a rectangular one is only diagonal).
    pub(crate) fn identity(square: bool) -> Props {
        let flags = if square { Props::IDENTITY | Props::PERMUTATION | Props::SYMMETRIC | Props::UNIT_DIAG } else { Props::DIAGONAL };
        Props { flags, band: Some((0, 0)), block: None, rank: None }.closed()
    }

    /// A permutation matrix (a single one in every row and column).
    pub(crate) fn permutation() -> Props {
        Props { flags: Props::PERMUTATION, band: None, block: None, rank: None }
    }

    /// A diagonal matrix.
    pub(crate) fn diagonal(square: bool) -> Props {
        let flags = if square { Props::DIAGONAL | Props::SYMMETRIC } else { Props::DIAGONAL };
        Props { flags, band: Some((0, 0)), block: None, rank: None }.closed()
    }

    /// A matrix with `lo` sub-diagonals and `hi` super-diagonals.
    pub(crate) fn banded(lo: u32, hi: u32) -> Props {
        Props { flags: 0, band: Some((lo, hi)), block: None, rank: None }.closed()
    }

    /// A lower or upper triangle of a matrix with `rows` rows (`unit`: with ones on the diagonal).
    pub(crate) fn triangle(lower: bool, unit: bool, rows: usize, cols: usize) -> Props {
        let flags = if lower { Props::LOWER } else { Props::UPPER } | if unit { Props::UNIT_DIAG } else { 0 };
        let wide = |extent: usize| u32::try_from(extent.saturating_sub(1)).unwrap_or(u32::MAX);
        let band = if lower { (wide(rows), 0) } else { (0, wide(cols)) };
        Props { flags, band: Some(band), block: None, rank: None }.closed()
    }

    /// The matrix with its two axes swapped.
    pub(crate) fn transposed(self) -> Props {
        let mut flags = self.flags & !(Props::LOWER | Props::UPPER);
        if self.has(Props::LOWER) {
            flags |= Props::UPPER;
        }
        if self.has(Props::UPPER) {
            flags |= Props::LOWER;
        }
        Props { flags, band: self.band.map(|(lo, hi)| (hi, lo)), block: None, rank: None }.closed()
    }

    /// `a + b` and `a - b` of two matrices of one shape: the pattern both have, the wider band of the two.
    pub(crate) fn summed(self, other: Props) -> Props {
        let band = self.band.zip(other.band).map(|(a, b)| (a.0.max(b.0), a.1.max(b.1)));
        Props { flags: self.flags & other.flags & PATTERN, band, block: None, rank: None }.closed()
    }

    /// `a * s` and `-a` for a scalar `s`: the zeros stay zeros; the ones of the diagonal do not.
    pub(crate) fn scaled(self) -> Props {
        let flags = self.flags & (PATTERN | Props::BANDED | Props::DIAGONAL);
        Props { flags, band: self.band, block: None, rank: None }.closed()
    }

    /// The matrix product `self * other` (the row-major product of the dense routines, see the module documentation).
    pub(crate) fn product(self, other: Props) -> Props {
        if self.has(Props::IDENTITY) {
            return other;
        }
        if other.has(Props::IDENTITY) {
            return self;
        }
        let mut out = Props::NONE;
        let sym = Props::DIAGONAL | Props::SYMMETRIC;
        if self.has(Props::DIAGONAL) || other.has(Props::DIAGONAL) {
            // A diagonal factor scales rows or columns of the other, and keeps its zeros where they are.
            let (diagonal, x) = if self.has(Props::DIAGONAL) { (self, other) } else { (other, self) };
            out.flags |= x.flags & (Props::LOWER | Props::UPPER | Props::DIAGONAL);
            out.band = x.band;
            if diagonal.has(sym) && x.has(sym) {
                out.flags |= Props::SYMMETRIC;
            }
        } else {
            for bit in [Props::LOWER, Props::UPPER] {
                if self.has(bit) && other.has(bit) {
                    out.flags |= bit;
                    if self.has(Props::UNIT_DIAG) && other.has(Props::UNIT_DIAG) {
                        out.flags |= Props::UNIT_DIAG;
                    }
                }
            }
            out.band = self.band.zip(other.band).and_then(|(a, b)| Some((a.0.checked_add(b.0)?, a.1.checked_add(b.1)?)));
        }
        if self.has(Props::PERMUTATION) && other.has(Props::PERMUTATION) {
            out.flags |= Props::PERMUTATION;
        }
        out.closed()
    }

    /// The text of the properties, for the decision log and tests.
    pub(crate) fn describe(self) -> String {
        const NAMES: [(u16, &str); 8] = [
            (Props::IDENTITY, "IDENTITY"),
            (Props::DIAGONAL, "DIAGONAL"),
            (Props::PERMUTATION, "PERMUTATION"),
            (Props::LOWER, "LOWER"),
            (Props::UPPER, "UPPER"),
            (Props::SYMMETRIC, "SYMMETRIC"),
            (Props::UNIT_DIAG, "UNIT_DIAG"),
            (Props::BANDED, "BANDED"),
        ];
        let mut parts: Vec<String> = NAMES.iter().filter(|(bit, _)| self.flags & bit != 0).map(|(_, name)| (*name).to_string()).collect();
        if let Some((lo, hi)) = self.band {
            parts.push(format!("band=({lo},{hi})"));
        }
        if parts.is_empty() { "NONE".to_string() } else { parts.join("|") }
    }
}

/// The compact copy of [`Props`] a handle carries: 4 bytes, in the padding after `read_view`. Bands wider than 255 are not
/// recorded (the triangle bits say what matters then).
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub(crate) struct HandleProps {
    flags: u16,
    band_lo: u8,
    band_hi: u8,
}

impl HandleProps {
    pub(crate) fn props(self) -> Props {
        let band = (self.flags & Props::BANDED != 0).then_some((u32::from(self.band_lo), u32::from(self.band_hi)));
        Props { flags: self.flags, band, block: None, rank: None }
    }

    pub(crate) fn of(props: &Props) -> Self {
        let mut flags = props.flags;
        let (mut band_lo, mut band_hi) = (0, 0);
        match props.band.map(|(lo, hi)| (u8::try_from(lo), u8::try_from(hi))) {
            Some((Ok(lo), Ok(hi))) => (band_lo, band_hi) = (lo, hi),
            _ => flags &= !Props::BANDED,
        }
        HandleProps { flags, band_lo, band_hi }
    }

    #[inline(always)]
    pub(crate) fn is_none(self) -> bool {
        self.flags == 0
    }

    fn bits(self) -> u32 {
        u32::from(self.flags) | u32::from(self.band_lo) << 16 | u32::from(self.band_hi) << 24
    }

    fn from_bits(bits: u32) -> Self {
        HandleProps { flags: bits as u16, band_lo: (bits >> 16) as u8, band_hi: (bits >> 24) as u8 }
    }
}

/// The [`HandleProps`] of a handle, in a cell: a handle that shares its data with another (a clone, a reshape) loses its
/// properties when the sharing starts, because a write through the other would leave them stale and `clone` only has `&self`.
#[derive(Default)]
pub(crate) struct PropsCell(AtomicU32);

impl PropsCell {
    #[inline(always)]
    pub(crate) fn get(&self) -> HandleProps {
        HandleProps::from_bits(self.0.load(Ordering::Relaxed))
    }

    pub(crate) fn set(&self, props: HandleProps) {
        self.0.store(props.bits(), Ordering::Relaxed);
    }

    /// Forgets the properties (the data is about to be written).
    #[inline(always)]
    pub(crate) fn clear(&self) {
        self.0.store(0, Ordering::Relaxed);
    }

    /// Forgets the properties because the data is about to be shared (a clone, a reshape): the value stays what it was, but a
    /// write through either handle would leave the other's claim stale. Debug builds count what is lost, to find the paths worth
    /// fixing (see [`Array::lost_properties_debug`]).
    #[inline(always)]
    pub(crate) fn lose(&self) {
        if self.0.load(Ordering::Relaxed) != 0 {
            #[cfg(debug_assertions)]
            LOST.fetch_add(1, Ordering::Relaxed);
            self.0.store(0, Ordering::Relaxed);
        }
    }
}

impl std::fmt::Debug for PropsCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.get().fmt(f)
    }
}

impl crate::array::Array<f32> {
    /// How many properties debug builds dropped because their handle was cloned or reshaped (0 in release builds).
    #[doc(hidden)]
    pub fn lost_properties_debug() -> u64 {
        #[cfg(debug_assertions)]
        return LOST.load(Ordering::Relaxed);
        #[cfg(not(debug_assertions))]
        0
    }

    /// The properties of this handle, for a consumer that is about to act on them. Debug builds re-derive one claim in 16 from
    /// the values (a read-back) and panic if the matrix breaks it: a stale or wrong property is the correctness risk of the
    /// whole feature. Matrices with a non-finite entry are not checked (the claims are about finite inputs).
    #[inline(always)]
    pub(crate) fn props_used(&self) -> Props {
        let props = self.props();
        #[cfg(debug_assertions)]
        if !props.is_none() && USES.fetch_add(1, Ordering::Relaxed).is_multiple_of(16) {
            self.check_claim(props);
        }
        props
    }

    #[cfg(debug_assertions)]
    #[inline(never)]
    fn check_claim(&self, props: Props) {
        let Ok(values) = self.to_vec() else { return };
        if values.iter().any(|v| !v.is_finite()) {
            return;
        }
        let [n, m, b2, b3] = self.shape().dims();
        let (lo, hi) = props.band.map_or((usize::MAX, usize::MAX), |(lo, hi)| (lo as usize, hi as usize));
        let label = props.describe();
        for z in 0..b2 * b3 {
            let at = |r: usize, c: usize| values[z * n * m + r * n + c];
            for r in 0..m {
                let mut ones = 0;
                for c in 0..n {
                    let v = at(r, c);
                    let zero_needed = (c > r && props.has(Props::LOWER))
                        || (c < r && props.has(Props::UPPER))
                        || (c != r && props.has(Props::DIAGONAL))
                        || (r > c && r - c > lo)
                        || (c > r && c - r > hi)
                        || (props.has(Props::PERMUTATION) && v != 1.0);
                    assert!(!zero_needed || v == 0.0, "matrix property claim {label} is false: entry (row {r}, col {c}) is {v}");
                    assert!(c != r || !props.has(Props::UNIT_DIAG) || v == 1.0, "matrix property claim {label} is false: diagonal entry {r} is {v}");
                    ones += usize::from(v == 1.0);
                    if props.has(Props::SYMMETRIC) && m == n && c < r {
                        let w = at(c, r);
                        // Bit-equal when the symmetric half-work product ran; a few ulps otherwise (the full kernel's two
                        // triangles may differ in the last bits on tensor cores).
                        assert!((v - w).abs() <= 8.0 * f32::EPSILON * v.abs().max(w.abs()), "matrix property claim {label} is false: entries ({r}, {c}) and ({c}, {r}) are {v} and {w}");
                    }
                }
                assert!(!props.has(Props::PERMUTATION) || ones == 1 || n == 0, "matrix property claim {label} is false: row {r} has {ones} ones");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has(props: Props, names: u16) -> bool {
        props.has(names)
    }

    #[test]
    fn products_follow_the_pattern_rules() {
        let lower = Props::triangle(true, false, 6, 6);
        let unit = Props::triangle(true, true, 6, 6);
        let upper = Props::triangle(false, false, 6, 6);
        let p = lower.product(unit);
        assert!(has(p, Props::LOWER | Props::BANDED) && !has(p, Props::UNIT_DIAG) && p.band == Some((10, 0)));
        let p = unit.product(unit);
        assert!(has(p, Props::LOWER | Props::UNIT_DIAG));
        let full = lower.product(upper);
        assert!(!has(full, Props::LOWER) && !has(full, Props::UPPER) && full.band == Some((5, 5)));
        let d = Props::diagonal(true);
        let p = d.product(lower);
        assert!(has(p, Props::LOWER) && !has(p, Props::SYMMETRIC | Props::DIAGONAL | Props::UNIT_DIAG));
        assert_eq!(lower.product(d).flags & (Props::LOWER | Props::SYMMETRIC), Props::LOWER);
        let p = d.product(d);
        assert!(has(p, Props::DIAGONAL | Props::SYMMETRIC));
        let i = Props::identity(true);
        assert_eq!(i.product(lower), lower);
        assert_eq!(upper.product(i), upper);
        assert!(has(i.product(i), Props::IDENTITY));
        let b = Props::banded(1, 0).product(Props::banded(1, 2));
        assert_eq!(b.band, Some((2, 2)));
        let perm = Props::permutation();
        assert!(has(perm.product(perm), Props::PERMUTATION));
        assert_eq!(perm.product(lower).flags & Props::PERMUTATION, 0);
    }

    #[test]
    fn the_lattice_is_closed() {
        let i = Props::identity(true);
        assert!(has(i, Props::IDENTITY | Props::DIAGONAL | Props::LOWER | Props::UPPER | Props::SYMMETRIC | Props::BANDED | Props::PERMUTATION));
        assert_eq!(i.band, Some((0, 0)));
        assert!(!has(Props::identity(false), Props::IDENTITY));
        assert!(has(Props::banded(0, 3), Props::UPPER) && !has(Props::banded(0, 3), Props::LOWER));
        assert!(has(Props::banded(0, 0), Props::DIAGONAL));
        // A contradiction knows nothing.
        let bad = Props { flags: Props::LOWER, band: Some((1, 2)), block: None, rank: None };
        assert_eq!(bad.closed(), Props::NONE);
        // Lower and upper together are diagonal.
        let both = Props { flags: Props::LOWER | Props::UPPER, band: None, block: None, rank: None }.closed();
        assert!(has(both, Props::DIAGONAL));
    }

    #[test]
    fn a_handle_keeps_what_fits_in_four_bytes() {
        assert_eq!(std::mem::size_of::<HandleProps>(), 4);
        let small = Props::banded(3, 5);
        assert_eq!(HandleProps::of(&small).props(), small);
        let wide = Props::banded(300, 2);
        let kept = HandleProps::of(&wide).props();
        assert_eq!(kept.band, None);
        assert!(!has(kept, Props::BANDED));
        let tri = Props::triangle(true, true, 1000, 1000);
        let kept = HandleProps::of(&tri).props();
        assert!(has(kept, Props::LOWER | Props::UNIT_DIAG) && kept.band.is_none());
    }

    #[test]
    fn transposes_and_sums() {
        let lower = Props::triangle(true, true, 5, 5);
        let t = lower.transposed();
        assert!(has(t, Props::UPPER | Props::UNIT_DIAG) && !has(t, Props::LOWER) && t.band == Some((0, 4)));
        assert_eq!(t.transposed(), lower);
        let sum = lower.summed(Props::diagonal(true));
        assert!(has(sum, Props::LOWER) && !has(sum, Props::UNIT_DIAG | Props::DIAGONAL));
        let scaled = Props::identity(true).scaled();
        assert!(has(scaled, Props::DIAGONAL | Props::SYMMETRIC) && !has(scaled, Props::IDENTITY | Props::PERMUTATION | Props::UNIT_DIAG));
    }
}
