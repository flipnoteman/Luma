//! The numerics policy of a context: which departures from exact `f32` semantics it allows.
//!
//! One value, [`Numerics`], replaces the separate switches for fast math, TF32 matrix products
//! and TF32 in the factorisations. Set it with
//! [`ContextOptions::numerics`](crate::ContextOptions::numerics) or `LUMA_NUMERICS`; read it with
//! [`Context::numerics`](crate::Context::numerics).

use std::fmt;

/// The reduced-precision features a context may use, a bit set (see [`Numerics::reduced`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Reduced(u32);

impl Reduced {
    /// No reduced-precision feature.
    pub const NONE: Self = Self(0);
    /// [`Array::matmul`](crate::Array::matmul) runs on tensor cores in TF32 where the device has them.
    pub const TF32_MATMUL: Self = Self(1 << 0);
    /// The trailing-update products of the factorisations (`lu`, `qr`, `cholesky`, ...) run in TF32
    /// where the device has tensor cores.
    pub const TF32_LINALG: Self = Self(1 << 1);
    /// The cheaper `Df32` additions (see [`ContextOptions::numerics`](crate::ContextOptions::numerics)). The algebraic rewrites
    /// that change values are not this bit's: they are the flags [`Numerics::exact_finite`], [`Numerics::reassociate`] and
    /// [`Numerics::reformulate`], which [`Numerics::fast`] and [`Numerics::algebraic`] set.
    pub const FAST_MATH: Self = Self(1 << 2);
    /// A pending program that reads spilled (host-resident) data may run on the host's CPU when that is cheaper than uploading
    /// it, even if the CPU's result may differ in the last bits from the device's (library math, float sums, products, scans).
    /// Without it only programs whose result is bit-identical wherever it is computed may run there. In [`Numerics::algebraic`]
    /// and [`Numerics::fast`], not in `default` or `strict`; `LUMA_NUMERICS=...+host`.
    pub const HOST_PLACEMENT: Self = Self(1 << 3);
    /// A product may run on a matrix unit whose accumulation differs from the sequential `f32` chain (hardware order and width)
    /// **without rounding any operand below its stored precision**: `f16` and `bf16` arrays on the half-precision tensor cores. The
    /// unit multiplies exactly and accumulates in `f32`; the result is rounded once to the array's type. In `default`, `algebraic` and
    /// `fast`, not in `strict` (which widens and uses the `f32` kernels); `LUMA_NUMERICS=...+matrix-accumulate`.
    pub const MATRIX_ACCUMULATE: Self = Self(1 << 4);
    /// `f32` operands of a product may be **rounded to `f16`** (11 significant bits; a magnitude over 65504 becomes infinite) so that
    /// it runs on the half-precision matrix units. In no preset: `LUMA_NUMERICS=...+f16-matmul`.
    pub const F16_OPERANDS: Self = Self(1 << 5);

    /// No feature.
    pub const fn empty() -> Self {
        Self(0)
    }
    /// Whether every feature of `other` is set.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    /// The features of both.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    /// The features of `self` that `other` does not have.
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
    /// Whether no feature is set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
    /// The raw bits.
    pub const fn bits(self) -> u32 {
        self.0
    }
}

impl std::ops::BitOr for Reduced {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl std::ops::BitOrAssign for Reduced {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// The numerics policy of a [`Context`](crate::Context): a constant of the context, part of
/// anything keyed by its options.
///
/// The three flags are measured against **exact-`f32` semantics** (IEEE `f32` operations, no TF32,
/// no fast math), whatever `reduced` says; a rule's class never depends on the context's `reduced`
/// bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Numerics {
    /// Rewrites that change how an expression rounds are allowed: the constants of a chain fold (`(x * a) * b`), `x / c` becomes
    /// `x * (1 / c)`, a power becomes a product, a quotient or a root. Float sums may change their order.
    pub reassociate: bool,
    /// Rewrites that compute something mathematically equal in a different way are allowed (`x + 0`, `x * 0`, `x / x`,
    /// `x.powf(0)`): the two sides differ at a signed zero, an infinity or NaN.
    pub reformulate: bool,
    /// Rewrites that are exact unless an input is infinite or NaN are allowed (`x - x`, `x == x`, `x + (-y)`): the two sides are
    /// bit-equal on every finite input, signed zeros included.
    pub exact_finite: bool,
    /// The reduced-precision features in use.
    pub reduced: Reduced,
}

impl Numerics {
    /// The default: TF32 matrix products where the device has tensor cores, `f16` / `bf16` products on the half-precision matrix
    /// units (operands never rounded below their stored precision), everything else exact.
    pub const fn default() -> Self {
        Numerics { reassociate: false, reformulate: false, exact_finite: false, reduced: Reduced::TF32_MATMUL.union(Reduced::MATRIX_ACCUMULATE) }
    }

    /// Exact `f32` semantics: no TF32, no fast math, no rewrite that changes a bit.
    pub const fn strict() -> Self {
        Numerics { reassociate: false, reformulate: false, exact_finite: false, reduced: Reduced::NONE }
    }

    /// [`Numerics::default`] and the algebraic rewrites that change rounding or are exact only
    /// for finite inputs. Among them, on matrix products: `a * identity` is `a`, `a * diag_matrix(v)`
    /// is `a * v`, `sum(a * b)` and `sum_axis(a * b, j)` are sums of the operands and a vector
    /// product, and a chain `(a * b) * v` is multiplied in the cheapest order. To let those rules see
    /// a product before it is computed, a product (and the sum of one) is not launched at the call
    /// but when its value is used, so a GPU starts a little later in a loop than with the default
    /// numerics.
    pub const fn algebraic() -> Self {
        Numerics { reassociate: true, reformulate: true, exact_finite: true, reduced: Reduced::TF32_MATMUL.union(Reduced::MATRIX_ACCUMULATE).union(Reduced::HOST_PLACEMENT) }
    }

    /// [`Numerics::algebraic`] and fast math.
    pub const fn fast() -> Self {
        Numerics { reduced: Reduced::TF32_MATMUL.union(Reduced::MATRIX_ACCUMULATE).union(Reduced::FAST_MATH).union(Reduced::HOST_PLACEMENT), ..Numerics::algebraic() }
    }

    /// Whether fast math is on.
    pub const fn fast_math(&self) -> bool {
        self.reduced.contains(Reduced::FAST_MATH)
    }

    /// Whether `matmul` may run in TF32.
    pub const fn tf32_matmul(&self) -> bool {
        self.reduced.contains(Reduced::TF32_MATMUL)
    }

    /// Whether the factorisations' products may run in TF32.
    pub const fn tf32_linalg(&self) -> bool {
        self.reduced.contains(Reduced::TF32_LINALG)
    }

    /// Whether results that may differ from the device's in the last bits may be computed on the host (see [`Reduced::HOST_PLACEMENT`]).
    pub const fn host_placement(&self) -> bool {
        self.reduced.contains(Reduced::HOST_PLACEMENT)
    }

    /// `self` with the reduced-precision features `on` set or cleared.
    pub const fn with_reduced(mut self, features: Reduced, on: bool) -> Self {
        self.reduced = if on { self.reduced.union(features) } else { self.reduced.difference(features) };
        self
    }

    /// Parses `LUMA_NUMERICS`: a preset (`strict`, `default`, `algebraic`, `fast`), optionally
    /// followed by modifiers `+name` / `-name` with `name` one of `tf32-matmul`, `tf32-linalg`,
    /// `fast-math`, `reassociate`, `reformulate`, `exact-finite`, e.g. `strict+tf32-linalg` or
    /// `default-tf32-matmul`. Case and surrounding blanks are ignored; the preset may be omitted
    /// (`default`).
    pub fn parse(text: &str) -> Result<Numerics, String> {
        let text = text.trim().to_ascii_lowercase();
        let cut = text.find(['+', '-']).unwrap_or(text.len());
        let (preset, mut rest) = text.split_at(cut);
        let mut numerics = match preset.trim() {
            "" | "default" => Numerics::default(),
            "strict" => Numerics::strict(),
            "algebraic" => Numerics::algebraic(),
            "fast" => Numerics::fast(),
            other => return Err(format!("unknown preset `{other}` (expected strict, default, algebraic or fast)")),
        };
        while !rest.is_empty() {
            let on = rest.starts_with('+');
            let body = &rest[1..];
            // A modifier ends at the next sign that starts a modifier: names use `-` inside, so
            // cut at a sign followed by a known name's first letter only after matching names.
            let (name, tail) = Self::MODIFIERS
                .iter()
                .find_map(|m| body.strip_prefix(m).filter(|t| t.is_empty() || t.starts_with(['+', '-'])).map(|t| (*m, t)))
                .ok_or_else(|| format!("unknown modifier `{rest}` (expected one of {})", Self::MODIFIERS.join(", ")))?;
            match name {
                "tf32-matmul" => numerics = numerics.with_reduced(Reduced::TF32_MATMUL, on),
                "tf32-linalg" => numerics = numerics.with_reduced(Reduced::TF32_LINALG, on),
                "fast-math" => numerics = numerics.with_reduced(Reduced::FAST_MATH, on),
                "host" => numerics = numerics.with_reduced(Reduced::HOST_PLACEMENT, on),
                "matrix-accumulate" => numerics = numerics.with_reduced(Reduced::MATRIX_ACCUMULATE, on),
                "f16-matmul" => numerics = numerics.with_reduced(Reduced::F16_OPERANDS, on),
                "reassociate" => numerics.reassociate = on,
                "reformulate" => numerics.reformulate = on,
                _ => numerics.exact_finite = on,
            }
            rest = tail;
        }
        Ok(numerics)
    }

    const MODIFIERS: [&'static str; 9] = ["tf32-matmul", "tf32-linalg", "fast-math", "host", "matrix-accumulate", "f16-matmul", "reassociate", "reformulate", "exact-finite"];
}

impl Default for Numerics {
    fn default() -> Self {
        Numerics::default()
    }
}

impl fmt::Display for Numerics {
    /// The preset's name with the modifiers that make it this value, in the syntax of
    /// `LUMA_NUMERICS`; [`Numerics::parse`] reads it back.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let presets = [("strict", Numerics::strict()), ("default", Numerics::default()), ("algebraic", Numerics::algebraic()), ("fast", Numerics::fast())];
        let distance = |p: &Numerics| {
            usize::from(p.reassociate != self.reassociate)
                + usize::from(p.reformulate != self.reformulate)
                + usize::from(p.exact_finite != self.exact_finite)
                + (p.reduced.bits() ^ self.reduced.bits()).count_ones() as usize
        };
        let (name, base) = presets.iter().min_by_key(|(_, p)| distance(p)).unwrap();
        f.write_str(name)?;
        let sign = |on: bool| if on { '+' } else { '-' };
        if base.reassociate != self.reassociate {
            write!(f, "{}reassociate", sign(self.reassociate))?;
        }
        if base.reformulate != self.reformulate {
            write!(f, "{}reformulate", sign(self.reformulate))?;
        }
        if base.exact_finite != self.exact_finite {
            write!(f, "{}exact-finite", sign(self.exact_finite))?;
        }
        for (flag, name) in [(Reduced::TF32_MATMUL, "tf32-matmul"), (Reduced::TF32_LINALG, "tf32-linalg"), (Reduced::FAST_MATH, "fast-math"), (Reduced::HOST_PLACEMENT, "host"), (Reduced::MATRIX_ACCUMULATE, "matrix-accumulate"), (Reduced::F16_OPERANDS, "f16-matmul")] {
            if base.reduced.contains(flag) != self.reduced.contains(flag) {
                write!(f, "{}{name}", sign(self.reduced.contains(flag)))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_round_trip_through_the_text_form() {
        for preset in [Numerics::strict(), Numerics::default(), Numerics::algebraic(), Numerics::fast()] {
            assert_eq!(Numerics::parse(&preset.to_string()), Ok(preset), "{preset}");
        }
        assert_eq!(Numerics::parse(" FAST "), Ok(Numerics::fast()));
        assert_eq!(Numerics::parse(""), Ok(Numerics::default()));
    }

    #[test]
    fn modifiers_apply_in_order() {
        let n = Numerics::parse("strict+tf32-linalg").unwrap();
        assert!(n.tf32_linalg() && !n.tf32_matmul() && !n.fast_math());
        let n = Numerics::parse("default-tf32-matmul").unwrap();
        assert_eq!(n, Numerics::strict());
        let n = Numerics::parse("algebraic-tf32-matmul+tf32-linalg-reassociate").unwrap();
        assert!(!n.tf32_matmul() && n.tf32_linalg() && !n.reassociate && n.reformulate && n.exact_finite);
        assert_eq!(Numerics::parse(&n.to_string()), Ok(n));
        let n = Numerics::parse("+tf32-linalg").unwrap();
        assert!(n.tf32_matmul() && n.tf32_linalg());
        assert!(Numerics::parse("strict+fast-math-fast-math+fast-math").unwrap().fast_math());
    }

    #[test]
    fn unknown_text_is_rejected() {
        assert!(Numerics::parse("quick").is_err());
        assert!(Numerics::parse("strict+tf32").is_err());
        assert!(Numerics::parse("strict+").is_err());
    }
}
