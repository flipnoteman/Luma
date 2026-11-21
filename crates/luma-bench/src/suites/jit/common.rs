//! Shared pieces of the `jit` suite: random expression recipes (with a host evaluation of the same
//! recipe), stencils with host references, and size selection.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::harness::Harness;
use luma::*;

pub type A = Array<f32>;

/// `quick` on `--quick`, `cpu` on the CPU backend, else `full`.
pub fn size(h: &Harness, quick: usize, cpu: usize, full: usize) -> usize {
    if h.quick {
        if h.ctx.is_cpu() { quick.min(cpu) } else { quick }
    } else if h.ctx.is_cpu() {
        cpu
    } else {
        full
    }
}

/// A small deterministic generator.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.next() >> 40) as f32 / (1u64 << 24) as f32 * (hi - lo)
    }
}

/// A seed that differs between processes: kernels built from "fresh" recipes were never compiled
/// before, not even by an earlier run of this benchmark (compiled kernels are cached on disk).
fn salt() -> u64 {
    static SALT: AtomicU64 = AtomicU64::new(0);
    let s = SALT.load(Ordering::Relaxed);
    if s != 0 {
        return s;
    }
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64) | 1;
    let _ = SALT.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
    SALT.load(Ordering::Relaxed)
}

/// One elementwise step of a recipe. Every variant is another kernel structure; the constants
/// are scalars (uniforms) and not part of the structure.
#[derive(Clone, Copy, Debug)]
pub enum Step {
    Add(f32),
    Mul(f32),
    SqrtAbs,
    Abs,
    Max,
    Min,
    Sub,
    MulY,
    Sin,
    Tanh,
}

impl Step {
    pub fn random(rng: &mut Rng) -> Step {
        match rng.below(10) {
            0 => Step::Add(rng.range(-0.5, 0.5)),
            1 => Step::Mul(rng.range(0.6, 1.3)),
            2 => Step::SqrtAbs,
            3 => Step::Abs,
            4 => Step::Max,
            5 => Step::Min,
            6 => Step::Sub,
            7 => Step::MulY,
            8 => Step::Sin,
            _ => Step::Tanh,
        }
    }

    /// The same step with other constants.
    pub fn rescaled(self, rng: &mut Rng) -> Step {
        match self {
            Step::Add(_) => Step::Add(rng.range(-0.5, 0.5)),
            Step::Mul(_) => Step::Mul(rng.range(0.6, 1.3)),
            other => other,
        }
    }

    pub fn dev(self, x: &A, y: &A) -> A {
        match self {
            Step::Add(c) => x + c,
            Step::Mul(c) => x * c,
            Step::SqrtAbs => (x.abs() + 0.05).sqrt(),
            Step::Abs => x.abs(),
            Step::Max => x.maximum(y),
            Step::Min => x.minimum(y * 0.9),
            Step::Sub => x - y,
            Step::MulY => &(x * y) * 0.5,
            Step::Sin => x.sin(),
            Step::Tanh => x.tanh(),
        }
    }

    pub fn host(self, x: f64, y: f64) -> f64 {
        match self {
            Step::Add(c) => x + f64::from(c),
            Step::Mul(c) => x * f64::from(c),
            Step::SqrtAbs => (x.abs() + 0.05).sqrt(),
            Step::Abs => x.abs(),
            Step::Max => x.max(y),
            Step::Min => x.min(y * 0.9),
            Step::Sub => x - y,
            Step::MulY => x * y * 0.5,
            Step::Sin => x.sin(),
            Step::Tanh => x.tanh(),
        }
    }
}

/// `len` random steps from a fixed seed.
pub fn recipe(seed: u64, len: usize) -> Vec<Step> {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    (0..len).map(|_| Step::random(&mut rng)).collect()
}

/// A generator seeded differently at every call, in this process and the next.
fn fresh_rng() -> Rng {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let k = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut rng = Rng((salt() ^ k.wrapping_mul(0xD6E8_FEB8_6659_FD93)).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    for _ in 0..4 {
        rng.next();
    }
    rng
}

/// `len` random steps never built before in this process or an earlier one.
pub fn fresh_recipe(len: usize) -> Vec<Step> {
    let mut rng = fresh_rng();
    (0..len).map(|_| Step::random(&mut rng)).collect()
}

/// The recipe applied to `x` (with `y` as its second operand).
pub fn apply(steps: &[Step], x: &A, y: &A) -> A {
    let mut a = x.clone();
    for s in steps {
        a = s.dev(&a, y);
    }
    a
}

/// The recipe on the host, in `f64`.
pub fn apply_host(steps: &[Step], x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter()
        .zip(y)
        .map(|(&x, &y)| {
            let y = f64::from(y);
            steps.iter().fold(f64::from(x), |a, s| s.host(a, y)) as f32
        })
        .collect()
}

/// `sum of the four neighbours - 4 * centre` of a `[w, h]` array with the edge values repeated.
pub fn lap_edge(e: &A, w: usize, h: usize) -> A {
    let p = e.pad([(1, 1), (1, 1)], PadMode::Edge);
    let centre = p.slice(0, 1..w + 1).slice(1, 1..h + 1);
    p.slice(0, 0..w).slice(1, 1..h + 1)
        + p.slice(0, 2..w + 2).slice(1, 1..h + 1)
        + p.slice(0, 1..w + 1).slice(1, 0..h)
        + p.slice(0, 1..w + 1).slice(1, 2..h + 2)
        - centre * 4.0
}

/// [`lap_edge`] on the host (`axis 0` varies fastest).
pub fn lap_edge_host(e: &[f64], w: usize, h: usize) -> Vec<f64> {
    let at = |i: isize, j: isize| e[i.clamp(0, w as isize - 1) as usize + w * j.clamp(0, h as isize - 1) as usize];
    let mut out = vec![0.0; w * h];
    for j in 0..h {
        for i in 0..w {
            let (a, b) = (i as isize, j as isize);
            out[i + w * j] = at(a - 1, b) + at(a + 1, b) + at(a, b - 1) + at(a, b + 1) - 4.0 * at(a, b);
        }
    }
    out
}

/// Circular neighbours of a `[w, h]` array: `sum of the four neighbours - 4 * centre`, by `shift`.
pub fn lap_wrap(u: &A) -> A {
    u.shift([1, 0]) + u.shift([-1, 0]) + u.shift([0, 1]) + u.shift([0, -1]) - u * 4.0
}

pub fn lap_wrap_host(e: &[f64], w: usize, h: usize) -> Vec<f64> {
    let at = |i: isize, j: isize| e[i.rem_euclid(w as isize) as usize + w * j.rem_euclid(h as isize) as usize];
    let mut out = vec![0.0; w * h];
    for j in 0..h {
        for i in 0..w {
            let (a, b) = (i as isize, j as isize);
            out[i + w * j] = at(a - 1, b) + at(a + 1, b) + at(a, b - 1) + at(a, b + 1) - 4.0 * at(a, b);
        }
    }
    out
}

pub fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

pub fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

/// `n` units of work per call as a rate in units per second (the harness divides items by 1e6).
pub fn per_sec(n: usize, unit: &'static str) -> crate::harness::Work {
    crate::harness::Work::Items(n as f64 * 1e6, unit)
}

/// Evaluates `a` and waits for it.
pub fn force(a: &A) {
    a.eval().unwrap();
    a.context().sync().unwrap();
}

/// A 24-operation elementwise expression: four rounds of `max(|sqrt(e * 0.9 + 0.2)|, e * 0.5)`.
pub fn heavy(x: &A) -> A {
    let mut e = x.clone();
    for _ in 0..4 {
        e = (&e * 0.9 + 0.2).sqrt().abs().maximum(&e * 0.5);
    }
    e
}

pub fn heavy_host(x: f64) -> f64 {
    let mut e = x;
    for _ in 0..4 {
        e = (e * 0.9 + 0.2).sqrt().abs().max(e * 0.5);
    }
    e
}

pub type D = Array<Df32>;

impl Step {
    /// A step of the subset `Df32` arrays support (no `sin` or `tanh`).
    pub fn random_df(rng: &mut Rng) -> Step {
        loop {
            let s = Step::random(rng);
            if !matches!(s, Step::Sin | Step::Tanh) {
                return s;
            }
        }
    }

    pub fn dev_df(self, x: &D, y: &D) -> D {
        let c = |v: f32| Df32::from(v);
        match self {
            Step::Add(v) => x + c(v),
            Step::Mul(v) => x * c(v),
            Step::SqrtAbs => (x.abs() + c(0.05)).sqrt(),
            Step::Abs => x.abs(),
            Step::Max => x.maximum(y),
            Step::Min => x.minimum(y * c(0.9)),
            Step::Sub => x - y,
            Step::MulY => &(x * y) * c(0.5),
            Step::Sin | Step::Tanh => unreachable!("not a Df32 step"),
        }
    }
}

/// `len` `Df32` steps never built before.
pub fn fresh_df_recipe(len: usize) -> Vec<Step> {
    let mut rng = fresh_rng();
    (0..len).map(|_| Step::random_df(&mut rng)).collect()
}

/// The recipe in `Df32` arithmetic on `f32` inputs and output.
pub fn apply_df(steps: &[Step], x: &A, y: &A) -> A {
    let (y, mut a) = (y.to_df32(), x.to_df32());
    for s in steps {
        a = s.dev_df(&a, &y);
    }
    a.to_f32()
}

/// [`heavy`] in `Df32` arithmetic.
pub fn heavy_df32(x: &A) -> A {
    let (a, b, c) = (Df32::from(0.9f32), Df32::from(0.2f32), Df32::from(0.5f32));
    let mut e = x.to_df32();
    for _ in 0..4 {
        e = (&e * a + b).sqrt().abs().maximum(&e * c);
    }
    e.to_f32()
}
