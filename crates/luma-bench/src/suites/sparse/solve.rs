//! Conjugate gradients on the 2-D Poisson matrix (5-point Laplacian of a `g x g` grid, Dirichlet
//! boundaries), with `A p` as an atomic CSR product, an ELLPACK product and a matrix-free stencil
//! of shifts. A fixed number of iterations runs as an `iterate_all` loop (state `x, r, p` and the
//! one-element `r.r`), a run to a tolerance as an `iterate_until_all` loop that stops on the GPU.
//!
//! The device iterate is verified by its true residual `|b - A x| / |b|` (computed in `f64` on
//! the host), which must not exceed the residual of `f64` CG after the same number of iterations
//! by more than a factor of two.

use luma::*;

use super::data::{self, Csr};
use super::*;

type A = Array<f32>;
type Apply<'a> = &'a dyn Fn(&A) -> A;

/// One CG iteration on the state `[x, r, p, r.r]`.
fn step(apply: Apply, s: &[A; 4]) -> [A; 4] {
    let [x, r, p, rs] = s;
    let ap = apply(p);
    let alpha = rs / (p * &ap).sum();
    let x2 = x + &alpha * p;
    let r2 = r - &alpha * &ap;
    let rs2 = (&r2 * &r2).sum();
    let p2 = &r2 + (&rs2 / rs) * p;
    [x2, r2, p2, rs2]
}

/// `f64` CG from `x = 0` for at most `iters` iterations, stopping when `r.r <= stop`; returns `x` and
/// the iterations run.
fn cg_host(a: &Csr, b: &[f32], iters: usize, stop: f64) -> (Vec<f64>, usize) {
    let n = a.rows;
    let mut x = vec![0.0f64; n];
    let mut r: Vec<f64> = b.iter().map(|&v| f64::from(v)).collect();
    let mut p = r.clone();
    let mut rs: f64 = r.iter().map(|v| v * v).sum();
    let mut ap = vec![0.0f64; n];
    let mut done = 0;
    while done < iters && rs > stop {
        for (i, out) in ap.iter_mut().enumerate() {
            *out = (a.ptr[i] as usize..a.ptr[i + 1] as usize).map(|k| f64::from(a.val[k]) * p[a.col[k] as usize]).sum();
        }
        let alpha = rs / p.iter().zip(&ap).map(|(a, b)| a * b).sum::<f64>();
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs2: f64 = r.iter().map(|v| v * v).sum();
        let beta = rs2 / rs;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs = rs2;
        done += 1;
    }
    (x, done)
}

/// `|b - A x| / |b|` in `f64`.
fn residual(a: &Csr, b: &[f32], x: &[f64]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for i in 0..a.rows {
        let ax: f64 = (a.ptr[i] as usize..a.ptr[i + 1] as usize).map(|k| f64::from(a.val[k]) * x[a.col[k] as usize]).sum();
        num += (f64::from(b[i]) - ax).powi(2);
        den += f64::from(b[i]).powi(2);
    }
    (num / den).sqrt()
}

/// FLOPs of one iteration: the product (2 per entry), two dot products and three vector updates.
fn flops(a: &Csr) -> f64 {
    2.0 * a.nnz() as f64 + 10.0 * a.rows as f64
}

/// How a vector is shaped for an operator: `[n, 1]` (CSR), `[1, n]` (ELL), `[g, g]` (grid).
#[derive(Clone, Copy)]
enum Layout {
    Csr,
    /// The `Sparse` type (CSR, `sum_segments` over the row pointer).
    Sparse,
    Ell,
    Grid,
}

impl Layout {
    fn shape(self, g: usize) -> [usize; 2] {
        match self {
            Layout::Csr | Layout::Sparse => [g * g, 1],
            Layout::Ell => [1, g * g],
            Layout::Grid => [g, g],
        }
    }

    fn name(self) -> &'static str {
        match self {
            Layout::Csr => "csr",
            Layout::Sparse => "sparse",
            Layout::Ell => "ell",
            Layout::Grid => "stencil",
        }
    }
}

fn stencil(p: &A) -> A {
    p * 4.0f32 - p.shift_fill([1, 0], 0.0) - p.shift_fill([-1, 0], 0.0) - p.shift_fill([0, 1], 0.0) - p.shift_fill([0, -1], 0.0)
}

struct Problem {
    g: usize,
    a: Csr,
    b: Vec<f32>,
    dev: spmv::Dev,
    ell: spmv::DevEll,
    csr: Sparse,
}

impl Problem {
    fn new(h: &Harness, g: usize) -> Problem {
        let a = data::poisson2d(g);
        let b = values(a.rows, 41, 0.0, 1.0);
        let (dev, ell) = (spmv::upload(h, &a), spmv::upload_ell(h, &a));
        let csr = Sparse::from_csr(a.rows, a.cols, &dev.ptr, &dev.col, &dev.val);
        Problem { g, a, b, dev, ell, csr }
    }

    fn apply(&self, layout: Layout) -> Box<dyn Fn(&A) -> A + '_> {
        match layout {
            Layout::Csr => Box::new(|p| spmv::atomic(&self.dev, p)),
            Layout::Sparse => Box::new(|p| self.csr.matmul(p)),
            Layout::Ell => Box::new(|p| spmv::ell(&self.ell, p)),
            Layout::Grid => Box::new(stencil),
        }
    }

    fn initial(&self, h: &Harness, layout: Layout) -> [A; 4] {
        let shape = layout.shape(self.g);
        let rs0: f64 = self.b.iter().map(|&v| f64::from(v).powi(2)).sum();
        [
            Array::zeros_on(&h.ctx, shape).unwrap(),
            up_f32(h, &self.b, shape),
            up_f32(h, &self.b, shape),
            up_f32(h, &[rs0 as f32], 1),
        ]
    }

    fn x_of(state: &[A; 4]) -> Vec<f64> {
        state[0].to_vec().unwrap().into_iter().map(f64::from).collect()
    }
}

pub fn run(h: &mut Harness) {
    let small = 128;
    let large = scale(h, 256, 512, 1536);
    for g in [small, large] {
        h.guarded("cg", |h| fixed_cases(h, g));
    }
    h.guarded("cg", |h| until_cases(h, scale(h, 128, 256, 512)));
}

/// `K` iterations in one `iterate_all` loop.
fn fixed_cases(h: &mut Harness, g: usize) {
    if !wants(h, "cg/fixed") {
        return;
    }
    let iters = 50;
    let p = Problem::new(h, g);
    let (host_x, _) = cg_host(&p.a, &p.b, iters, 0.0);
    let want = residual(&p.a, &p.b, &host_x);
    let size = format!("{}x{} grid {} nnz x{iters}", g, g, label(p.a.nnz()));
    for layout in [Layout::Csr, Layout::Sparse, Layout::Ell, Layout::Grid] {
        let name = format!("cg/fixed_{}", layout.name());
        if !h.want(&name) {
            continue;
        }
        let apply = p.apply(layout);
        let init = p.initial(h, layout);
        let go = || iterate_all([&init[0], &init[1], &init[2], &init[3]], iters, |s| step(&*apply, s));
        let got = residual(&p.a, &p.b, &Problem::x_of(&go()));
        assert!(got <= 2.0 * want + 1e-6, "{name}: residual {got:.3e} after {iters} iterations, host CG reaches {want:.3e}");
        case_trim(h, &name, &size, Work::Flops(flops(&p.a) * iters as f64), &mut || {
            for s in go() {
                s.eval().unwrap();
            }
        });
    }
}

/// Iterate until `r.r <= (2e-2 |b|)^2`, with the stop rule evaluated on the GPU (the `f32` iterate
/// cannot be pushed much below `eps * cond(A)`, a few 1e-3, in true residual).
fn until_cases(h: &mut Harness, g: usize) {
    if !wants(h, "cg/until") {
        return;
    }
    let p = Problem::new(h, g);
    let tol = 2e-2;
    let rs0: f64 = p.b.iter().map(|&v| f64::from(v).powi(2)).sum();
    let stop = (tol * tol * rs0) as f32;
    let (host_x, host_steps) = cg_host(&p.a, &p.b, 10_000, f64::from(stop));
    let want = residual(&p.a, &p.b, &host_x);
    for layout in [Layout::Csr, Layout::Grid] {
        let name = format!("cg/until_{}", layout.name());
        if !h.want(&name) {
            continue;
        }
        let apply = p.apply(layout);
        let init = p.initial(h, layout);
        let go = || {
            iterate_until_all(init.clone(), Until::new(4 * host_steps + 16).record("rs", |s: &[A; 4]| s[3].clone()).stop_when(|h| h["rs"].last().le(stop)), |s, _| step(&*apply, s)).unwrap()
        };
        let run = go();
        let got = residual(&p.a, &p.b, &Problem::x_of(&run.state));
        assert!(got <= 2.0 * tol, "{name}: residual {got:.3e} above 2x the tolerance {tol:.0e} (host CG reaches {want:.3e})");
        let steps = run.steps;
        assert!(steps.abs_diff(host_steps) <= 3 + host_steps / 10, "{name}: stopped after {steps} iterations, host CG after {host_steps}");
        let size = format!("{}x{} grid {} nnz x{steps}", g, g, label(p.a.nnz()));
        case_trim(h, &name, &size, Work::Flops(flops(&p.a) * steps as f64), &mut || {
            go().state[0].eval().unwrap();
        });
    }
}
