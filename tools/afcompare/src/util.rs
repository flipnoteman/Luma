//! Deterministic inputs and result comparison.

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [0, 1) with 24 random bits.
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// `n` uniform values in `[lo, hi)`.
pub fn uniform(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| lo + (hi - lo) * r.unit()).collect()
}

/// `n` random values of `0..m`.
pub fn indices(n: usize, seed: u64, m: u32) -> Vec<u32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| ((r.next_u64() >> 32) % u64::from(m)) as u32).collect()
}

/// Random signs: every partial sum of them is an exact `f32` integer.
pub fn signs(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| if r.next_u64() >> 63 == 0 { 1.0 } else { -1.0 }).collect()
}

pub fn rms(v: &[f32]) -> f64 {
    (v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / v.len().max(1) as f64).sqrt()
}

#[derive(Clone, Debug)]
pub struct Verdict {
    pub ok: bool,
    pub text: String,
}

impl Verdict {
    pub fn yes(text: impl Into<String>) -> Verdict {
        Verdict { ok: true, text: text.into() }
    }
    pub fn no(text: impl Into<String>) -> Verdict {
        Verdict { ok: false, text: text.into() }
    }
    /// Both verdicts must hold.
    pub fn and(self, other: Verdict) -> Verdict {
        Verdict { ok: self.ok && other.ok, text: format!("{}; {}", self.text, other.text) }
    }
}

/// `luma` against `af`: every element within `abs + rel * |af|`.
pub fn compare(luma: &[f32], af: &[f32], rel: f64, abs: f64) -> Verdict {
    if luma.len() != af.len() {
        return Verdict::no(format!("length: luma {} vs ArrayFire {}", luma.len(), af.len()));
    }
    let (mut worst, mut worst_i, mut worst_ratio) = (0.0f64, 0usize, 0.0f64);
    for (i, (&l, &a)) in luma.iter().zip(af).enumerate() {
        let d = if l.is_nan() && a.is_nan() { 0.0 } else { (f64::from(l) - f64::from(a)).abs() };
        let d = if d.is_nan() { f64::INFINITY } else { d };
        let tol = abs + rel * f64::from(a).abs();
        let ratio = if tol > 0.0 { d / tol } else if d == 0.0 { 0.0 } else { f64::INFINITY };
        if d > worst {
            worst = d;
        }
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst_i = i;
        }
    }
    if worst_ratio <= 1.0 {
        Verdict::yes(format!("yes (max Δ {worst:.1e})"))
    } else {
        Verdict::no(format!("NO: max Δ {worst:.3e}; at [{worst_i}] luma {} vs ArrayFire {} (tol {:.1e} + {:.1e}·abs(x))", luma[worst_i], af[worst_i], abs, rel))
    }
}

/// Exact equality.
pub fn exact(luma: &[f32], af: &[f32]) -> Verdict {
    compare(luma, af, 0.0, 0.0)
}

/// `got` against an `f64` reference.
pub fn compare_ref(who: &str, got: &[f32], want: &[f64], rel: f64, abs: f64) -> Verdict {
    if got.len() != want.len() {
        return Verdict::no(format!("{who}: length {} vs reference {}", got.len(), want.len()));
    }
    let mut worst = 0.0f64;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (f64::from(g) - w).abs();
        let tol = abs + rel * w.abs();
        if d.is_nan() || d > tol {
            return Verdict::no(format!("NO: {who} vs f64 reference at [{i}]: {g} vs {w} (tol {tol:.1e})"));
        }
        worst = worst.max(d);
    }
    Verdict::yes(format!("{who} vs f64 ref max Δ {worst:.1e}"))
}

/// Whether `v` is non-decreasing.
pub fn is_sorted(v: &[f32]) -> bool {
    v.windows(2).all(|w| w[0] <= w[1])
}
