//! The `cfd` suite: the workloads of a line-implicit compressible-flow solver (the bowshock
//! pattern) on 2-D and 3-D structured grids, as the regression ledger for the PDE side of Luma.
//!
//! * `heat`, `euler`, `muscl`: explicit stencil steps replayed by `iterate_all` (ghost cells by
//!   record assigns, Rusanov and MUSCL residuals, 5- and 7-point heat), plus the host-driven loop
//!   (`*_per_step`) and, for `euler` and `muscl`, the host cost of recording a step alone
//!   (`*_build_step`: the lazy graph of the steps built and dropped without evaluating them).
//! * `adi`: implicit heat, a tridiagonal solve along every axis in the delta form.
//! * `lines`, `parts`: block-tridiagonal line solves of blocks 1 to 5 along contiguous and strided
//!   axes on stacked arrays and on parts (evaluated and lazy coefficients).
//! * `implicit`, `bc`: the pieces of a bowshock-like implicit step (state -> block coefficients ->
//!   solves along every axis -> defect correction -> update) and the ghost-cell assigns on all faces.
//! * `converge`: `iterate_until_all` and `compile_until_all` loops, and the fixed cost per point
//!   of a run (trace, kernel build, first chunk) with compiled loops against fresh traces.
//! * `poisson`, `mg`: Jacobi, red-black and CG pressure solves (stencil matvec, dot reductions,
//!   `item()` readbacks) and multigrid transfers and V-cycles through reshaped views.
//!
//! Every case is verified once against a double precision host reference (see `host.rs`), or for
//! the big line solves against the host residual of the returned solution, before it is timed.
//! "small" sizes are wedge-like (about 20k cells, latency bound), "large" ones sector-like
//! (300k to 700k cells). On the CPU backend the large sizes shrink.

mod euler;
mod host;

use crate::harness::*;
use euler::*;
use host::{Cell, Grid};
use luma::*;

/// Sizes of the run: they shrink with `--quick` and on the CPU backend.
struct Z {
    quick: bool,
    cpu: bool,
    s2: [usize; 2],
    l2: [usize; 2],
    s3: [usize; 3],
    l3: [usize; 3],
    /// The 3-D grid of the line solves on parts (the sector grid of bench_solve).
    g3: [usize; 3],
    /// The 3-D grid of the stacked line solves.
    lines3: [usize; 3],
    /// Wedge-like 2-D grid for the multigrid cycles (divisible by 4).
    v2: [usize; 2],
}

impl Z {
    fn new(h: &Harness) -> Z {
        let (quick, cpu) = (h.quick, h.ctx.is_cpu());
        Z {
            quick,
            cpu,
            s2: [156, 122],
            l2: if cpu { [256, 192] } else if quick { [320, 256] } else { [700, 512] },
            s3: [48, 40, 12],
            l3: if cpu { [48, 40, 24] } else if quick { [64, 48, 20] } else { [128, 96, 32] },
            g3: if cpu { [96, 40, 16] } else if quick { [94, 36, 12] } else { [187, 70, 24] },
            lines3: if cpu { [64, 32, 24] } else if quick { [128, 64, 24] } else { [256, 64, 48] },
            v2: [160, 128],
        }
    }

    /// Steps of a loop: `full` on the GPU, fewer in quick runs and on the CPU.
    fn steps(&self, full: usize) -> usize {
        if self.quick {
            (full / 5).max(2)
        } else if self.cpu {
            (full / 2).max(2)
        } else {
            full
        }
    }
}

fn dims_str(d: &[usize]) -> String {
    d.iter().map(usize::to_string).collect::<Vec<_>>().join("x")
}

const AXES: [char; 3] = ['i', 'j', 'k'];

fn cells_of(d: &[usize]) -> usize {
    d.iter().product()
}

/// `check` of every field of a record.
fn check_rec(name: &str, got: &[Vec<f32>], want: &[Vec<f32>], rel: f32, abs: f32) {
    assert_eq!(got.len(), want.len(), "{name}: fields");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        check(&format!("{name}[{k}]"), g, w, rel, abs);
    }
}

fn refs<const N: usize>(q: &[A; N]) -> [&A; N] {
    std::array::from_fn(|k| &q[k])
}

fn eval_rec<const N: usize>(q: &[A; N]) {
    Fields::eval(q).unwrap();
}

pub fn run(h: &mut Harness) {
    let z = Z::new(h);
    heat(h, &z);
    explicit_euler::<4>(h, &z, "euler/2d", false);
    explicit_euler::<4>(h, &z, "muscl/2d", true);
    explicit_euler::<5>(h, &z, "euler/3d", false);
    explicit_euler::<5>(h, &z, "muscl/3d", true);
    adi(h, &z);
    lines(h, &z);
    parts(h, &z);
    bc(h, &z);
    implicit::<4>(h, &z, "implicit/2d");
    implicit::<5>(h, &z, "implicit/3d");
    converge(h, &z);
    poisson(h, &z);
    multigrid(h, &z);
}

// =================================================================================================
// Explicit heat and Euler
// =================================================================================================

/// Zero-flux (edge) padding of `u` by 1 on its `d` axes.
fn pad1(u: &A, d: usize, mode: PadMode) -> A {
    match d {
        2 => u.pad([(1, 1); 2], mode),
        3 => u.pad([(1, 1); 3], mode),
        _ => unreachable!("2-D or 3-D"),
    }
}

/// The neighbours of `p` (padded by one) at `+-1` along `axis`, as arrays of the interior `n`.
fn neighbours(p: &A, axis: usize, n: &[usize]) -> [A; 2] {
    let at = |o: usize| {
        let mut v = p.clone();
        for (b, &len) in n.iter().enumerate() {
            v = if b == axis { v.slice(b, o..o + len) } else { v.slice(b, 1..1 + len) };
        }
        v
    };
    [at(0), at(2)]
}

fn heat_step(u: &A, alpha: f32) -> A {
    let n: Vec<usize> = u.shape().dims().iter().copied().filter(|&e| e > 1).collect();
    let d = n.len();
    let p = pad1(u, d, PadMode::Edge);
    let mut sum: Option<A> = None;
    for a in 0..d {
        for v in neighbours(&p, a, &n) {
            sum = Some(match sum {
                None => v,
                Some(s) => s + v,
            });
        }
    }
    u + alpha * (sum.unwrap() - 2.0 * d as f32 * u)
}

fn heat(h: &mut Harness, z: &Z) {
    for (d, small, big, alpha, steps) in [(2usize, z.s2.to_vec(), z.l2.to_vec(), 0.2f32, z.steps(100)), (3, z.s3.to_vec(), z.l3.to_vec(), 0.12, z.steps(50))] {
        let group = format!("heat/{d}d");
        h.guarded(&group, |h| {
            let ctx = h.ctx.clone();
            let vn: Vec<usize> = if d == 2 { vec![24, 18] } else { vec![14, 12, 10] };
            let hv = values(cells_of(&vn), 5, 0.0, 1.0);
            let u = Array::from_slice_on(&ctx, &hv, shape(&vn)).unwrap();
            let [got] = iterate_all([&u], 10, |s| [heat_step(&s[0], alpha)]);
            check(&group, &got.to_vec().unwrap(), &host::h_heat(&hv, Grid::new(&vn), f64::from(alpha), 10), 1e-4, 1e-5);
            for (label, dims) in [("small", &small), ("large", &big)] {
                let u = Array::from_slice_on(&ctx, &values(cells_of(dims), 6, 0.0, 1.0), shape(dims)).unwrap();
                u.eval().unwrap();
                let _ = label;
                let mut f = || {
                    let [r] = iterate_all([&u], steps, |s| [heat_step(&s[0], alpha)]);
                    r.eval().unwrap();
                };
                h.case(&format!("{group}_replay"), &format!("{}:{steps}st", dims_str(dims)), Work::Cells((cells_of(dims) * steps) as f64), &mut f);
            }
            // The host loop: one launch (or more) per step, evaluated every step.
            let u = Array::from_slice_on(&ctx, &values(cells_of(&small), 7, 0.0, 1.0), shape(&small)).unwrap();
            u.eval().unwrap();
            let n = z.steps(50);
            let mut f = || {
                let mut cur = heat_step(&u, alpha);
                cur.eval().unwrap();
                for _ in 1..n {
                    let next = heat_step(&cur, alpha);
                    next.eval().unwrap();
                    cur = next;
                }
            };
            h.case(&format!("{group}_per_step"), &format!("{}:{n}st", dims_str(&small)), Work::Cells((cells_of(&small) * n) as f64), &mut f);
        });
    }
}

/// Records `n` steps off `q` and drops them unevaluated.
#[inline(never)]
fn build_steps<const N: usize>(q: &[A; N], muscl: bool, n: usize) {
    for _ in 0..n {
        std::hint::black_box(explicit_step(q, muscl, true));
    }
}

/// Explicit Euler (first order, or MUSCL for `muscl/..` groups) with the low-i wall: `iterate_all` replay at two sizes and the host loop.
fn explicit_euler<const N: usize>(h: &mut Harness, z: &Z, group: &str, muscl: bool) {
    let d = N - 2;
    let (small, big, steps) = if d == 2 { (z.s2.to_vec(), z.l2.to_vec(), z.steps(if muscl { 20 } else { 50 })) } else { (z.s3.to_vec(), z.l3.to_vec(), z.steps(if muscl { 10 } else { 20 })) };
    h.guarded(group, |h| {
        let ctx = h.ctx.clone();
        let vn: Vec<usize> = if d == 2 { vec![20, 16] } else { vec![14, 12, 10] };
        let hv = init_host::<N>(&vn, 1.0);
        let vsteps = 3;
        let want = host::h_explicit_run::<N>(&hv, Grid::new(&vn), &|a, high| a == 0 && !high, muscl, f64::from(DT), vsteps);
        let q = upload::<N>(&ctx, &hv, &vn);
        let got = iterate_all(refs(&q), vsteps, |s| explicit_step(s, muscl, true));
        check_rec(group, &download(&got), &want, 2e-3, 2e-3);
        for dims in [&small, &big] {
            let q = upload::<N>(&ctx, &init_host::<N>(dims, 1.0), dims);
            let mut f = || {
                let r = iterate_all(refs(&q), steps, |s| explicit_step(s, muscl, true));
                eval_rec(&r);
            };
            h.case(&format!("{group}_iterate"), &format!("{}:{steps}st", dims_str(dims)), Work::Cells((cells_of(dims) * steps) as f64), &mut f);
        }
        let q = upload::<N>(&ctx, &init_host::<N>(&small, 1.0), &small);
        let n = steps.min(20);
        let mut f = || {
            let mut cur = explicit_step(&q, muscl, true);
            eval_rec(&cur);
            for _ in 1..n {
                let next = explicit_step(&cur, muscl, true);
                eval_rec(&next);
                cur = next;
            }
        };
        h.case(&format!("{group}_per_step"), &format!("{}:{n}st", dims_str(&small)), Work::Cells((cells_of(&small) * n) as f64), &mut f);
        // Host cost of recording a step alone: build the lazy graph of `n` steps off the same state and drop it unevaluated.
        let mut f = || {
            build_steps(&q, muscl, n);
        };
        h.build_case(&format!("{group}_build_step"), &format!("{}:{n}st", dims_str(&small)), Work::Cells((cells_of(&small) * n) as f64), &mut f);
    });
}

// =================================================================================================
// Implicit heat (ADI)
// =================================================================================================

fn adi_step(u: &A, kappa: &A, r: f32) -> A {
    let n: Vec<usize> = u.shape().dims().iter().copied().filter(|&e| e > 1).collect();
    let d = n.len();
    let p = pad1(u, d, PadMode::Zero);
    let mut sum: Option<A> = None;
    for a in 0..d {
        for v in neighbours(&p, a, &n) {
            sum = Some(match sum {
                None => v,
                Some(s) => s + v,
            });
        }
    }
    let rk = r * kappa;
    let mut du = &rk * (sum.unwrap() - 2.0 * d as f32 * u);
    let off = -&rk;
    let diag = 1.0 + 2.0 * &rk;
    for axis in 0..d {
        du = solve_tridiagonal(axis, &off, &diag, &off, &du);
    }
    u + du
}

fn adi(h: &mut Harness, z: &Z) {
    for (d, small, big, steps) in [(2usize, z.s2.to_vec(), z.l2.to_vec(), z.steps(10)), (3, z.s3.to_vec(), z.l3.to_vec(), z.steps(10))] {
        let group = format!("adi/heat{d}d");
        h.guarded(&group, |h| {
            let ctx = h.ctx.clone();
            let r = 2.0f32;
            let vn: Vec<usize> = if d == 2 { vec![30, 22] } else { vec![14, 12, 10] };
            let (hv, hk) = (values(cells_of(&vn), 11, 0.0, 1.0), values(cells_of(&vn), 12, 0.5, 1.5));
            let kappa = Array::from_slice_on(&ctx, &hk, shape(&vn)).unwrap();
            let u = Array::from_slice_on(&ctx, &hv, shape(&vn)).unwrap();
            let [got] = iterate_all([&u], 3, |s| [adi_step(&s[0], &kappa, r)]);
            check(&group, &got.to_vec().unwrap(), &host::h_adi(&hv, &hk, Grid::new(&vn), f64::from(r), 3), 1e-3, 1e-4);
            for dims in [&small, &big] {
                let kappa = Array::from_slice_on(&ctx, &values(cells_of(dims), 13, 0.5, 1.5), shape(dims)).unwrap();
                let u = Array::from_slice_on(&ctx, &values(cells_of(dims), 14, 0.0, 1.0), shape(dims)).unwrap();
                kappa.eval().unwrap();
                u.eval().unwrap();
                let mut f = || {
                    let [res] = iterate_all([&u], steps, |s| [adi_step(&s[0], &kappa, r)]);
                    res.eval().unwrap();
                };
                h.case(&group, &format!("{}:{steps}st", dims_str(dims)), Work::Cells((cells_of(dims) * steps) as f64), &mut f);
            }
        });
    }
}

// =================================================================================================
// Block-tridiagonal line solves
// =================================================================================================

/// Blocks of an implicit step: strongly diagonally dominant `b x b` blocks per cell (`lo`, `diag`, `up`, `rhs`; entry `(row, col)` at `row * b + col`).
fn block_system(cells: usize, b: usize, seed: u64) -> [Vec<f32>; 4] {
    let bb = b * b;
    let mut diag = values(cells * bb, seed + 1, -1.0, 1.0);
    for (k, v) in diag.iter_mut().enumerate() {
        if (k % bb) / b == k % b {
            *v += 2.0 * b as f32 + 2.0;
        }
    }
    [values(cells * bb, seed, -1.0, 1.0), diag, values(cells * bb, seed + 2, -1.0, 1.0), values(cells * b, seed + 3, -1.0, 1.0)]
}

fn grid3(dims: &[usize]) -> [usize; 3] {
    let mut g = [1; 3];
    g[..dims.len()].copy_from_slice(dims);
    g
}

/// Panics unless the residual of a line solve is at most `tol` (the data are of order 1).
fn check_solve(name: &str, b: usize, dims: &[usize], axis: usize, sys: &[Vec<f32>; 4], x: &[f32]) {
    let res = host::line_residual(b, grid3(dims), axis, sys, x);
    assert!(res < 3e-3 && res.is_finite(), "{name}: residual of the solve {res:e} (axis {axis}, grid {})", dims_str(dims));
}

/// Stacked-array solves (`solve_tridiagonal` for `b == 1`, `solve_block_tridiagonal` otherwise) along each of `axes` of every grid in `configs`.
fn stacked(h: &mut Harness, case: &str, b: usize, configs: &[(Vec<usize>, Vec<usize>)]) {
    h.guarded(case, |h| {
        let ctx = h.ctx.clone();
        for (dims, axes) in configs {
            let cells = cells_of(dims);
            let bb = b * b;
            let sys = block_system(cells, b, 31 + b as u64);
            let mk = |k: usize| -> A {
                let c = if k == 3 { b } else { bb };
                let full: Vec<usize> = if b == 1 { dims.clone() } else { std::iter::once(c).chain(dims.iter().copied()).collect() };
                let a = Array::from_slice_on(&ctx, &sys[k], shape(&full)).unwrap();
                a.eval().unwrap();
                a
            };
            let [lo, diag, up, rhs] = [0, 1, 2, 3].map(mk);
            for &axis in axes {
                let run = || if b == 1 { solve_tridiagonal(axis, &lo, &diag, &up, &rhs) } else { solve_block_tridiagonal(b, axis + 1, &lo, &diag, &up, &rhs) };
                check_solve(case, b, dims, axis, &sys, &run().to_vec().unwrap());
                let mut f = || run().eval().unwrap();
                let bytes = ((3 * bb + 2 * b) * 4 * cells) as f64;
                h.case(case, &format!("{}:{}", dims_str(dims), AXES[axis]), Work::Bytes(bytes), &mut f);
            }
        }
    });
}

fn lines(h: &mut Harness, z: &Z) {
    for b in 1..=5usize {
        let mut configs = vec![(z.s2.to_vec(), vec![0, 1])];
        if b == 1 || b == 4 {
            configs.push((z.lines3.to_vec(), vec![0, 1, 2]));
        }
        stacked(h, &format!("lines/stack_b{b}"), b, &configs);
    }
    // Few, long lines: a workgroup per line (cyclic reduction) rather than an invocation per line.
    let (few1, few4) = if z.quick || z.cpu { (vec![1024, 64], vec![256, 32]) } else { (vec![4096, 200], vec![1024, 100]) };
    stacked(h, "lines/few_b1", 1, &[(few1, vec![0, 1])]);
    stacked(h, "lines/few_b4", 4, &[(few4, vec![0, 1])]);
}

/// Line solves on parts along `axes` of every grid in `configs`: `3 b^2 + b` arrays, evaluated or (`lazy`) each an expression
/// of two of 50 face fields, computed inside the solver kernel.
fn parts_solves(h: &mut Harness, case: &str, b: usize, lazy: bool, configs: &[(Vec<usize>, Vec<usize>)]) {
    h.guarded(case, |h| {
        let ctx = h.ctx.clone();
        let (bb, per) = (b * b, 3 * b * b + b);
        for (dims, axes) in configs {
            let cells = cells_of(dims);
            let faces: Vec<Vec<f32>> = (0..50).map(|k| values(cells, 900 + k as u64, -0.5, 0.5)).collect();
            let shift = |k: usize| if (bb..2 * bb).contains(&k) && (k - bb) / b == (k - bb) % b { 2.0 * b as f32 + 1.0 } else { 0.0 };
            let weight = |k: usize| 1.0 + k as f32 * 0.01;
            let host_part = |k: usize| -> Vec<f32> { (0..cells).map(|q| faces[k % 50][q] * weight(k) + faces[(k * 7 + 3) % 50][q] * 0.5 + shift(k)).collect() };
            let face_arrays: Vec<A> = faces.iter().map(|f| Array::from_slice_on(&ctx, f, shape(dims)).unwrap()).collect();
            let parts: Vec<A> = (0..per)
                .map(|k| {
                    if lazy {
                        &(&face_arrays[k % 50] * weight(k)) + &(&face_arrays[(k * 7 + 3) % 50] * 0.5) + shift(k)
                    } else {
                        Array::from_slice_on(&ctx, &host_part(k), shape(dims)).unwrap()
                    }
                })
                .collect();
            face_arrays.iter().for_each(|a| a.eval().unwrap());
            if !lazy {
                parts.iter().for_each(|a| a.eval().unwrap());
            }
            // The system in the cell-major layout of the host references.
            let interleave = |range: std::ops::Range<usize>| -> Vec<f32> {
                let host: Vec<Vec<f32>> = range.map(host_part).collect();
                (0..cells * host.len()).map(|i| host[i % host.len()][i / host.len()]).collect()
            };
            let sys = [interleave(0..bb), interleave(bb..2 * bb), interleave(2 * bb..3 * bb), interleave(3 * bb..per)];
            let (lo, diag, up, rhs) = (&parts[..bb], &parts[bb..2 * bb], &parts[2 * bb..3 * bb], &parts[3 * bb..]);
            for &axis in axes {
                let run = || solve_parts(axis, b, lo, diag, up, rhs);
                let comps: Vec<Vec<f32>> = run().iter().map(|c| c.to_vec().unwrap()).collect();
                let got: Vec<f32> = (0..cells * b).map(|i| comps[i % b][i / b]).collect();
                check_solve(case, b, dims, axis, &sys, &got);
                let mut f = || eval_all(&run().iter().collect::<Vec<_>>()).unwrap();
                let bytes = ((3 * bb + 2 * b) * 4 * cells) as f64;
                h.case(case, &format!("{}:{}", dims_str(dims), AXES[axis]), Work::Bytes(bytes), &mut f);
            }
        }
    });
}

fn parts(h: &mut Harness, z: &Z) {
    for b in 1..=5usize {
        let mut configs = vec![(z.g3.to_vec(), vec![0, 1, 2])];
        if b == 4 {
            configs.push((z.s2.to_vec(), vec![0, 1]));
            configs.push((z.l2.to_vec(), vec![0, 1]));
        }
        parts_solves(h, &format!("parts/b{b}"), b, false, &configs);
    }
    parts_solves(h, "parts/b4_lazy", 4, true, &[(z.g3.to_vec(), vec![0, 1, 2])]);
}

// =================================================================================================
// Ghost cells and the implicit step
// =================================================================================================

fn walls(axis: usize, high: bool) -> bool {
    implicit_bc(axis, high) == Bc::Wall
}

fn bc(h: &mut Harness, z: &Z) {
    for (d, small, big) in [(2usize, z.s2.to_vec(), z.l2.to_vec()), (3, z.s3.to_vec(), z.l3.to_vec())] {
        let group = format!("bc/ghost{d}d");
        h.guarded(&group, |h| if d == 2 { ghost_cases::<4>(h, &group, &small, &big) } else { ghost_cases::<5>(h, &group, &small, &big) });
    }
    face_apply(h, z);
    ghost_region(h, z);
    h.guarded("bc/assign_record", |h| {
        let ctx = h.ctx.clone();
        let (vw, vh) = (12, 9);
        let host4: Vec<Vec<f32>> = (0..4).map(|k| values(vw * vh, 70 + k, 0.2, 1.2)).collect();
        let mut q = upload::<4>(&ctx, &host4, &[vw, vh]);
        let ghost = Fields::flip(&Fields::slice(&q, 0, 2..4), 0).map(|a| a * -1.0);
        q.slice_mut(0, ..2).assign(ghost).unwrap();
        let got = download(&q);
        for k in 0..4 {
            let mut want = host4[k].clone();
            for y in 0..vh {
                want[vw * y] = -host4[k][3 + vw * y];
                want[1 + vw * y] = -host4[k][2 + vw * y];
            }
            check("bc/assign_record", &got[k], &want, 0.0, 0.0);
        }
        // 100 mirror assigns of a four-field record in a row: launch latency of boundary updates.
        let side = if z.quick { 128 } else { 256 };
        let mut q = upload::<4>(&ctx, &(0..4).map(|_| vec![1.0f32; side * side]).collect::<Vec<_>>(), &[side, side]);
        let mut f = || {
            for _ in 0..100 {
                let ghost = Fields::flip(&Fields::slice(&q, 0, 2..4), 0).map(|a| a * -1.0);
                q.slice_mut(0, ..2).assign(ghost).unwrap();
            }
        };
        h.case("bc/assign_record", &format!("4x{side}x{side}:x100"), Work::Items(100.0, "assign"), &mut f);
    });
}

/// `v` with its momentum (fields 1..=3) reflected about the unit normal `n`: `v - 2 (v.n) n`.
fn reflect_rec(v: &[A; 5], n: &[A; 3]) -> [A; 5] {
    let dot = &v[1] * &n[0] + &v[2] * &n[1] + &v[3] * &n[2];
    std::array::from_fn(|k| if (1..=3).contains(&k) { &v[k] - 2.0 * &dot * &n[k - 1] } else { v[k].clone() })
}

/// The data of one `face_apply` pass over a grid `dims` along `dir`: the face matrices `fl`, `fr` (25 arrays each over the
/// `n + 1` faces of every line), the vector `z` over the cells, the unit normals of the two line ends (one cell along `dir`),
/// the solid-neighbour masks and normals of the faces. Host copies are kept for the reference.
struct FaceApply {
    dir: usize,
    dims: Vec<usize>,
    host: FaceHost,
    fl: Vec<A>,
    fr: Vec<A>,
    z: [A; 5],
    ends: [[A; 3]; 2],
    wall: [Array<bool>; 2],
    nw: [A; 3],
}

struct FaceHost {
    fl: Vec<Vec<f32>>,
    fr: Vec<Vec<f32>>,
    z: Vec<Vec<f32>>,
    ends: [Vec<Vec<f32>>; 2],
    wall: [Vec<bool>; 2],
    nw: Vec<Vec<f32>>,
}

impl FaceApply {
    fn new(ctx: &Context, dims: &[usize], dir: usize) -> FaceApply {
        let (mut fdims, mut edims) = (dims.to_vec(), dims.to_vec());
        fdims[dir] += 1;
        edims[dir] = 1;
        let (cells, faces, edge) = (cells_of(dims), cells_of(&fdims), cells_of(&edims));
        let mut seed = 5000 + dir as u64;
        let mut data = |n: usize, lo: f32, hi: f32| {
            seed += 1;
            values(n, seed, lo, hi)
        };
        let fl: Vec<Vec<f32>> = (0..25).map(|_| data(faces, -0.5, 0.5)).collect();
        let fr: Vec<Vec<f32>> = (0..25).map(|_| data(faces, -0.5, 0.5)).collect();
        let z: Vec<Vec<f32>> = (0..5).map(|_| data(cells, 0.5, 1.5)).collect();
        let mut unit = |n: usize| -> Vec<Vec<f32>> {
            let raw: Vec<Vec<f32>> = (0..3).map(|_| data(n, 0.2, 1.0)).collect();
            let norm: Vec<f32> = (0..n).map(|q| (raw[0][q] * raw[0][q] + raw[1][q] * raw[1][q] + raw[2][q] * raw[2][q]).sqrt()).collect();
            raw.iter().map(|r| (0..n).map(|q| r[q] / norm[q]).collect()).collect()
        };
        let ends = [unit(edge), unit(edge)];
        let nw = unit(faces);
        let flags = |seed: u64| -> Vec<bool> { ints(faces, seed, 16).into_iter().map(|v| v == 0).collect() };
        let wall = [flags(31), flags(32)];
        let up = |h: &[Vec<f32>], d: &[usize]| -> Vec<A> { h.iter().map(|v| Array::from_slice_on(ctx, v, shape(d)).unwrap()).collect() };
        let fa = FaceApply {
            dir,
            dims: dims.to_vec(),
            fl: up(&fl, &fdims),
            fr: up(&fr, &fdims),
            z: std::array::from_fn(|k| Array::from_slice_on(ctx, &z[k], shape(dims)).unwrap()),
            ends: std::array::from_fn(|e| std::array::from_fn(|q| Array::from_slice_on(ctx, &ends[e][q], shape(&edims)).unwrap())),
            wall: std::array::from_fn(|e| Array::from_slice_on(ctx, &wall[e], shape(&fdims)).unwrap()),
            nw: std::array::from_fn(|q| Array::from_slice_on(ctx, &nw[q], shape(&fdims)).unwrap()),
            host: FaceHost { fl, fr, z, ends, wall, nw },
        };
        let inputs = fa.fl.iter().chain(&fa.fr).chain(&fa.z).chain(&fa.nw).chain(fa.ends.iter().flatten());
        inputs.for_each(|a| a.eval().unwrap());
        fa.wall.iter().for_each(|a| a.eval().unwrap());
        fa
    }

    /// The pass of a block line solve over the faces of the direction that reads the state across a boundary: the face matrices applied to the cells
    /// below and above each face, `zl` and `zr`, the vector `z` extended by the reflected ghost at the line ends (a concat of the end cell) and, at
    /// the faces of a solid neighbour, by the reflection of the other side (a select of one mask over five fields). 50 coefficient reads and 5
    /// writes per face.
    fn apply(&self) -> [A; 5] {
        let (dir, z) = (self.dir, &self.z);
        let n = self.dims[dir];
        let lo_end = reflect_rec(&std::array::from_fn(|k| z[k].slice(dir, 0..1)), &self.ends[0]);
        let hi_end = reflect_rec(&std::array::from_fn(|k| z[k].slice(dir, n - 1..n)), &self.ends[1]);
        let zl: [A; 5] = std::array::from_fn(|k| concat(dir, [&lo_end[k], &z[k]]));
        let zr: [A; 5] = std::array::from_fn(|k| concat(dir, [&z[k], &hi_end[k]]));
        let (r_of_r, r_of_l) = (reflect_rec(&zr, &self.nw), reflect_rec(&zl, &self.nw));
        let zl: [A; 5] = std::array::from_fn(|k| select(&self.wall[0], &r_of_r[k], &zl[k]));
        let zr: [A; 5] = std::array::from_fn(|k| select(&self.wall[1], &r_of_l[k], &zr[k]));
        std::array::from_fn(|q| {
            let mut acc = &self.fl[5 * q] * &zl[0] + &self.fr[5 * q] * &zr[0];
            for j in 1..5 {
                acc = acc + &self.fl[5 * q + j] * &zl[j] + &self.fr[5 * q + j] * &zr[j];
            }
            acc
        })
    }

    /// The double precision mirror of [`FaceApply::apply`], component-major over the faces.
    fn reference(&self) -> Vec<Vec<f32>> {
        let (dir, hd) = (self.dir, &self.host);
        let g = Grid::new(&self.dims);
        let (mut fd, mut ed) = (g.n, g.n);
        fd[dir] += 1;
        ed[dir] = 1;
        let (fg, eg) = (Grid { n: fd, d: g.d }, Grid { n: ed, d: g.d });
        let last = g.n[dir] - 1;
        let refl = |v: [f64; 5], n: [f64; 3]| -> [f64; 5] {
            let dot = v[1] * n[0] + v[2] * n[1] + v[3] * n[2];
            [v[0], v[1] - 2.0 * dot * n[0], v[2] - 2.0 * dot * n[1], v[3] - 2.0 * dot * n[2], v[4]]
        };
        let cell = |c: [usize; 3]| -> [f64; 5] { std::array::from_fn(|k| f64::from(hd.z[k][g.flat(c)])) };
        let mut out = vec![vec![0.0f32; fg.cells()]; 5];
        for f in 0..fg.cells() {
            let c = fg.coords(f);
            let ghost = |side: usize, at: usize| -> [f64; 5] {
                let (mut e, mut line) = (c, c);
                e[dir] = at;
                line[dir] = 0;
                refl(cell(e), std::array::from_fn(|q| f64::from(hd.ends[side][q][eg.flat(line)])))
            };
            let mut below = if c[dir] == 0 {
                ghost(0, 0)
            } else {
                let mut b = c;
                b[dir] -= 1;
                cell(b)
            };
            let mut above = if c[dir] == last + 1 { ghost(1, last) } else { cell(c) };
            let n: [f64; 3] = std::array::from_fn(|q| f64::from(hd.nw[q][f]));
            let (r_of_above, r_of_below) = (refl(above, n), refl(below, n));
            if hd.wall[0][f] {
                below = r_of_above;
            }
            if hd.wall[1][f] {
                above = r_of_below;
            }
            for q in 0..5 {
                out[q][f] = (0..5).map(|j| f64::from(hd.fl[5 * q + j][f]) * below[j] + f64::from(hd.fr[5 * q + j][f]) * above[j]).sum::<f64>() as f32;
            }
        }
        out
    }
}

/// `FaceApply` on every direction of a sector-like grid: the `n_apply` pass of a line solve, a record extended across the line ends
/// (reflected ghost concats and wall selects of one mask over five fields) fused with the 5 x 5 face matrix products.
fn face_apply(h: &mut Harness, z: &Z) {
    h.guarded("bc/face_apply", |h| {
        let ctx = h.ctx.clone();
        for dir in 0..3 {
            let small = FaceApply::new(&ctx, &[9, 8, 7], dir);
            let got: Vec<Vec<f32>> = small.apply().iter().map(|a| a.to_vec().unwrap()).collect();
            check_rec("bc/face_apply", &got, &small.reference(), 2e-3, 2e-3);
        }
        for dir in 0..3 {
            let dims = sector_dims(z);
            let fa = FaceApply::new(&ctx, &dims, dir);
            let faces = cells_of(&dims) / dims[dir] * (dims[dir] + 1);
            let mut f = || {
                let gf = fa.apply();
                eval_all(&gf.iter().collect::<Vec<_>>()).unwrap();
            };
            h.case("bc/face_apply", &format!("{}:{}", dims_str(&dims), AXES[dir]), Work::Bytes((faces * 4 * 55) as f64), &mut f);
        }
    });
}

/// The grid of the bowshock sector case (`bench3 --case sector`) on the GPU; the `g3` grid on the CPU and in quick runs.
fn sector_dims(z: &Z) -> Vec<usize> {
    if z.cpu || z.quick { z.g3.to_vec() } else { vec![196, 76, 18] }
}

const FREESTREAM: [f32; 5] = [1.0, 2.0, 0.1, 0.05, 0.714];

/// The characteristic far-field ghost of the interior primitives `w` for the outward unit normal `n` (the bowshock `characteristic_ghost3`):
/// Riemann invariants on the normal velocity, entropy and tangential velocity from the interior or the free stream, supersonic faces
/// take one of them. Every field selects between several expressions that share the invariants.
fn char_ghost(w: &[A; 5], n: &[A; 3]) -> [A; 5] {
    let (g, g1) = (GAMMA, GAMMA - 1.0);
    let fs = FREESTREAM;
    let ci = (g * &w[4] / &w[0]).sqrt();
    let uni = &w[1] * &n[0] + &w[2] * &n[1] + &w[3] * &n[2];
    let m = &uni / &ci;
    let ce = (g * fs[4] / fs[0]).sqrt();
    let une = fs[1] * &n[0] + fs[2] * &n[1] + fs[3] * &n[2];
    let r_out = &uni + 2.0 * &ci / g1;
    let r_in = &une - 2.0 * ce / g1;
    let un_b = 0.5 * (&r_out + &r_in);
    let c_b = (0.25 * g1 * (&r_out - &r_in)).maximum(1e-8f32);
    let from_interior = un_b.ge(0.0f32);
    let rf: [A; 4] = std::array::from_fn(|k| select(&from_interior, &w[k], fs[k]));
    let un_ref = select(&from_interior, &uni, &une);
    let c_ref = select(&from_interior, &ci, ce);
    let q = &c_b / &c_ref;
    let q2 = &q * &q;
    let rho = &rf[0] * (&q2 * &q2 * &q);
    let dn = &un_b - &un_ref;
    let riemann = [rho.clone(), &rf[1] + &dn * &n[0], &rf[2] + &dn * &n[1], &rf[3] + &dn * &n[2], &rho * &c_b * &c_b / g];
    let super_in = m.le(-1.0f32);
    let super_out = m.ge(1.0f32);
    std::array::from_fn(|k| select(&super_in, fs[k], select(&super_out, &w[k], &riemann[k])))
}

/// Host mirror of [`char_ghost`] for one cell.
fn h_char_ghost(w: [f64; 5], n: [f64; 3]) -> [f64; 5] {
    let g = f64::from(GAMMA);
    let fs = FREESTREAM.map(f64::from);
    let ci = (g * w[4] / w[0]).sqrt();
    let uni = w[1] * n[0] + w[2] * n[1] + w[3] * n[2];
    let m = uni / ci;
    let ce = (g * fs[4] / fs[0]).sqrt();
    let une = fs[1] * n[0] + fs[2] * n[1] + fs[3] * n[2];
    let r_out = uni + 2.0 * ci / (g - 1.0);
    let r_in = une - 2.0 * ce / (g - 1.0);
    let un_b = 0.5 * (r_out + r_in);
    let c_b = (0.25 * (g - 1.0) * (r_out - r_in)).max(1e-8);
    let from_interior = un_b >= 0.0;
    let rf: [f64; 4] = std::array::from_fn(|k| if from_interior { w[k] } else { fs[k] });
    let un_ref = if from_interior { uni } else { une };
    let c_ref = if from_interior { ci } else { ce };
    let q = c_b / c_ref;
    let rho = rf[0] * q.powi(5);
    let dn = un_b - un_ref;
    let riemann = [rho, rf[1] + dn * n[0], rf[2] + dn * n[1], rf[3] + dn * n[2], rho * c_b * c_b / g];
    std::array::from_fn(|k| if m <= -1.0 { fs[k] } else if m >= 1.0 { w[k] } else { riemann[k] })
}

/// The padded primitive record of the 3-D implicit step with its ghost layers written as region assigns on a padded copy (the bowshock
/// `assign_ghosts3`): the conserved state `u` to primitives, edge pad by two, then the characteristic far field at `i = 0`, `i = ni` and
/// `j = nj` (one expression of the edge cells over every field) and the reflection about the symmetry planes at `k = 0`, `k = nk`.
/// Evaluating the record is one kernel that writes the five padded fields.
struct GhostRegion {
    dims: Vec<usize>,
    u: [A; 5],
    /// Normals of the `i` low, `i` high, `j` high far-field faces (one layer along their axis) and of the two `k` planes.
    n: [[A; 3]; 5],
    host_u: Vec<Vec<f32>>,
    host_n: Vec<Vec<Vec<f32>>>,
}

impl GhostRegion {
    fn new(ctx: &Context, dims: &[usize]) -> GhostRegion {
        let (ni, nj, nk) = (dims[0], dims[1], dims[2]);
        let host_u = init_host::<5>(dims, 1.0);
        let planes = [[1, nj, nk], [1, nj, nk], [ni, 1, nk], [ni, nj, 1], [ni, nj, 1]];
        let mut seed = 777u64;
        let host_n: Vec<Vec<Vec<f32>>> = planes
            .iter()
            .map(|p| {
                let len = cells_of(p);
                let raw: Vec<Vec<f32>> = (0..3)
                    .map(|_| {
                        seed += 1;
                        values(len, seed, -1.0, 1.0)
                    })
                    .collect();
                let norm: Vec<f32> = (0..len).map(|q| (raw[0][q] * raw[0][q] + raw[1][q] * raw[1][q] + raw[2][q] * raw[2][q]).sqrt().max(0.1)).collect();
                raw.iter().map(|r| (0..len).map(|q| r[q] / norm[q]).collect()).collect()
            })
            .collect();
        let n = std::array::from_fn(|s| std::array::from_fn(|a| Array::from_slice_on(ctx, &host_n[s][a], shape(&planes[s])).unwrap()));
        let gr = GhostRegion { dims: dims.to_vec(), u: upload::<5>(ctx, &host_u, dims), n, host_u, host_n };
        gr.n.iter().flatten().for_each(|a| a.eval().unwrap());
        gr
    }

    fn record(&self) -> [A; 5] {
        let (ni, nj, nk) = (self.dims[0], self.dims[1], self.dims[2]);
        let ok = "ghost layer assign";
        let p = prim(&self.u);
        let mut w = Fields::pad(&p, [(2, 2), (2, 2), (2, 2)], PadMode::Edge);
        let edge_lo: [A; 5] = Fields::slice(&Fields::pad(&p, [(2, 0)], PadMode::Edge), 0, 0..2);
        w.slice_mut(0, ..2).slice(1, 2..nj + 2).slice(2, 2..nk + 2).assign(char_ghost(&edge_lo, &self.n[0])).expect(ok);
        let edge_hi: [A; 5] = Fields::slice(&Fields::pad(&p, [(0, 2)], PadMode::Edge), 0, ni..ni + 2);
        w.slice_mut(0, ni + 2..).slice(1, 2..nj + 2).slice(2, 2..nk + 2).assign(char_ghost(&edge_hi, &self.n[1])).expect(ok);
        let edge_top: [A; 5] = Fields::slice(&Fields::pad(&p, [(0, 0), (0, 2)], PadMode::Edge), 1, nj..nj + 2);
        w.slice_mut(1, nj + 2..).slice(0, 2..ni + 2).slice(2, 2..nk + 2).assign(char_ghost(&edge_top, &self.n[2])).expect(ok);
        let mirror_lo: [A; 5] = Fields::slice(&Fields::pad(&p, [(0, 0), (0, 0), (2, 0)], PadMode::Mirror), 2, 0..2);
        w.slice_mut(2, ..2).slice(0, 2..ni + 2).slice(1, 2..nj + 2).assign(reflect_rec(&mirror_lo, &self.n[3])).expect(ok);
        let mirror_hi: [A; 5] = Fields::slice(&Fields::pad(&p, [(0, 0), (0, 0), (0, 2)], PadMode::Mirror), 2, nk..nk + 2);
        w.slice_mut(2, nk + 2..).slice(0, 2..ni + 2).slice(1, 2..nj + 2).assign(reflect_rec(&mirror_hi, &self.n[4])).expect(ok);
        w
    }

    /// The double precision mirror of [`GhostRegion::record`], component-major over the padded grid.
    fn reference(&self) -> Vec<Vec<f32>> {
        let (ni, nj, nk) = (self.dims[0], self.dims[1], self.dims[2]);
        let g = Grid::new(&self.dims);
        let pg = Grid::new(&[ni + 4, nj + 4, nk + 4]);
        let cells: Vec<Cell<5>> = host::to_cells::<5>(&self.host_u).into_iter().map(host::h_prim).collect();
        let at = |c: [usize; 3]| cells[g.flat(c)];
        let normal = |s: usize, c: [usize; 3], plane: [usize; 3]| -> [f64; 3] {
            let pgrid = Grid::new(&plane);
            std::array::from_fn(|a| f64::from(self.host_n[s][a][pgrid.flat(c)]))
        };
        let reflect = |v: [f64; 5], n: [f64; 3]| -> [f64; 5] {
            let dot = v[1] * n[0] + v[2] * n[1] + v[3] * n[2];
            [v[0], v[1] - 2.0 * dot * n[0], v[2] - 2.0 * dot * n[1], v[3] - 2.0 * dot * n[2], v[4]]
        };
        let clamp = |c: usize, n: usize| c.saturating_sub(2).min(n - 1);
        let mut out = vec![vec![0.0f32; pg.cells()]; 5];
        for q in 0..pg.cells() {
            let [i, j, k] = pg.coords(q);
            let (ci, cj, ck) = (clamp(i, ni), clamp(j, nj), clamp(k, nk));
            let (in_j, in_k) = ((2..nj + 2).contains(&j), (2..nk + 2).contains(&k));
            let in_i = (2..ni + 2).contains(&i);
            let v = if i < 2 && in_j && in_k {
                h_char_ghost(at([0, cj, ck]), normal(0, [0, cj, ck], [1, nj, nk]))
            } else if i >= ni + 2 && in_j && in_k {
                h_char_ghost(at([ni - 1, cj, ck]), normal(1, [0, cj, ck], [1, nj, nk]))
            } else if j >= nj + 2 && in_i && in_k {
                h_char_ghost(at([ci, nj - 1, ck]), normal(2, [ci, 0, ck], [ni, 1, nk]))
            } else if k < 2 && in_i && in_j {
                reflect(at([ci, cj, 1 - k]), normal(3, [ci, cj, 0], [ni, nj, 1]))
            } else if k >= nk + 2 && in_i && in_j {
                reflect(at([ci, cj, 2 * nk + 1 - k]), normal(4, [ci, cj, 0], [ni, nj, 1]))
            } else {
                at([ci, cj, ck])
            };
            for (field, &x) in out.iter_mut().zip(&v) {
                field[q] = x as f32;
            }
        }
        out
    }
}

/// The ghost kernel of the 3-D implicit step: a padded copy of the primitives with five region assigns of shared expressions.
fn ghost_region(h: &mut Harness, z: &Z) {
    h.guarded("bc/ghost_region", |h| {
        let ctx = h.ctx.clone();
        let small = GhostRegion::new(&ctx, &[9, 8, 7]);
        let got: Vec<Vec<f32>> = small.record().iter().map(|a| a.to_vec().unwrap()).collect();
        check_rec("bc/ghost_region", &got, &small.reference(), 3e-3, 3e-3);
        let dims = sector_dims(z);
        let gr = GhostRegion::new(&ctx, &dims);
        let padded: usize = dims.iter().map(|d| d + 4).product();
        let mut f = || {
            let w = gr.record();
            eval_all(&w.iter().collect::<Vec<_>>()).unwrap();
        };
        h.case("bc/ghost_region", &dims_str(&dims), Work::Bytes(((cells_of(&dims) + padded) * 5 * 4) as f64), &mut f);
    });
}

/// The padded primitive record of the implicit step: pad, then every face of the grid assigned (walls and extrapolation).
fn ghost_cases<const N: usize>(h: &mut Harness, group: &str, small: &[usize], big: &[usize]) {
    let ctx = h.ctx.clone();
    let vn: Vec<usize> = if N == 4 { vec![14, 11] } else { vec![9, 8, 7] };
    let hv = init_host::<N>(&vn, 1.0);
    let q = upload::<N>(&ctx, &hv, &vn);
    let padded = ghosted(&prim(&q), 1, &implicit_bc);
    let got = download(&padded);
    let g = Grid::new(&vn);
    let prims: Vec<Cell<N>> = host::to_cells::<N>(&hv).into_iter().map(host::h_prim).collect();
    let field = host::Field { grid: g, prim: &prims, wall: &walls };
    let pn: Vec<usize> = vn.iter().map(|n| n + 2).collect();
    let pg = Grid::new(&pn);
    for pq in 0..pg.cells() {
        let co = pg.coords(pq);
        let outside = (0..N - 2).filter(|&a| co[a] == 0 || co[a] == pn[a] - 1).count();
        if outside >= 2 {
            continue; // corners are not read by a first-order stencil
        }
        let want = field.at([co[0] as isize - 1, co[1] as isize - 1, co[2] as isize - 1]);
        for (w, comp) in want.iter().zip(&got) {
            let have = comp[pq];
            assert!((f64::from(have) - w).abs() < 1e-5 * w.abs().max(1.0), "{group}: padded at {co:?}: {have} vs {w}");
        }
    }
    for dims in [small, big] {
        let q = upload::<N>(&ctx, &init_host::<N>(dims, 1.0), dims);
        let mut f = || eval_rec(&ghosted(&prim(&q), 1, &implicit_bc));
        let padded_cells: usize = dims.iter().map(|n| n + 2).product();
        h.case(group, &dims_str(dims), Work::Bytes((N * 4 * (cells_of(dims) + padded_cells)) as f64), &mut f);
    }
}

fn implicit_state_check<const N: usize>(ctx: &Context, group: &str) {
    let vn: Vec<usize> = if N == 4 { vec![24, 18] } else { vec![12, 10, 8] };
    let hv = init_host::<N>(&vn, 1.0);
    let cfl = Array::from_slice_on(ctx, &[2.0f32], 1).unwrap();
    let zero = zeros(ctx, &vn);
    let mut q = upload::<N>(ctx, &hv, &vn);
    let mut cells: Vec<Cell<N>> = host::to_cells(&hv);
    for sub in [2usize, 1] {
        let (un, _) = implicit_step(&q, &cfl, &zero, sub);
        cells = host::h_implicit_step(&cells, Grid::new(&vn), &walls, 2.0, sub);
        q = Fields::cache(&un);
    }
    check_rec(group, &download(&q), &host::from_cells(&cells), 3e-3, 3e-3);
}

/// An evaluated array of zeros of a grid.
fn zeros(ctx: &Context, dims: &[usize]) -> A {
    let z = Array::<f32>::zeros_on(ctx, shape(dims)).unwrap();
    z.eval().unwrap();
    z
}

fn implicit<const N: usize>(h: &mut Harness, z: &Z, group: &str) {
    let d = N - 2;
    let (small, big) = if d == 2 { (z.s2.to_vec(), z.l2.to_vec()) } else { (z.s3.to_vec(), z.l3.to_vec()) };
    h.guarded(group, |h| {
        let ctx = h.ctx.clone();
        implicit_state_check::<N>(&ctx, group);
        let cfl = Array::from_slice_on(&ctx, &[2.0f32], 1).unwrap();
        for dims in [&small, &big] {
            let q = upload::<N>(&ctx, &init_host::<N>(dims, 1.0), dims);
            let zero = zeros(&ctx, dims);
            let first = std::cell::Cell::new(true);
            let mut f = || {
                let (un, rn) = implicit_step(&q, &cfl, &zero, 2);
                let mut all: Vec<&A> = un.iter().collect();
                all.push(&rn);
                eval_all(&all).unwrap();
                if first.replace(false) {
                    let rho_min = un[0].min().item().unwrap();
                    assert!(rho_min.is_finite() && rho_min > 0.0, "{group}: minimum density {rho_min} after a step on {}", dims_str(dims));
                }
            };
            h.case(&format!("{group}_step"), &dims_str(dims), Work::Cells(cells_of(dims) as f64), &mut f);
        }
        let q = upload::<N>(&ctx, &init_host::<N>(&small, 1.0), &small);
        let zero = zeros(&ctx, &small);
        let mut f = || {
            let (un, rn) = implicit_step(&q, &cfl, &zero, 1);
            let mut all: Vec<&A> = un.iter().collect();
            all.push(&rn);
            eval_all(&all).unwrap();
        };
        h.case(&format!("{group}_step_sub1"), &dims_str(&small), Work::Cells(cells_of(&small) as f64), &mut f);
        // The coefficients alone: ghost cells, residual, time step and the 3 N^2 block entries along every axis, evaluated.
        for dims in [&small, &big] {
            let q = upload::<N>(&ctx, &init_host::<N>(dims, 1.0), dims);
            let zero = zeros(&ctx, dims);
            let mut f = || {
                let l = assemble(&q, &cfl, &zero);
                l.rn.eval().unwrap();
            };
            h.case(&format!("{group}_coeffs"), &dims_str(dims), Work::Cells(cells_of(dims) as f64), &mut f);
        }
    });
}

// =================================================================================================
// Convergence loops
// =================================================================================================

/// The loop state of the implicit solver: conserved record, density residual (one element), CFL (one element).
const GROWTH: f32 = 1.1;
const CFL_MAX: f32 = 20.0;

fn pack<const N: usize, const M: usize>(q: [A; N], rn: A, cfl: A) -> [A; M] {
    let mut it = q.into_iter().chain([rn, cfl]);
    std::array::from_fn(|_| it.next().unwrap())
}

fn init_state<const N: usize, const M: usize>(ctx: &Context, dims: &[usize], amp: f64, cfl0: f32) -> [A; M] {
    let q = upload::<N>(ctx, &init_host::<N>(dims, amp), dims);
    let rn = Array::<f32>::zeros_on(ctx, 1).unwrap();
    let cfl = Array::from_slice_on(ctx, &[cfl0], 1).unwrap();
    pack(q, rn, cfl)
}

fn force_state<const M: usize>(s: &[A; M]) {
    Fields::eval(s).unwrap();
}

/// The implicit solver's loop on `[q.., residual, cfl]` states of one grid: the step (two sweeps of the approximate factorisation,
/// the CFL ramped by the state), the record of the residual and the stop rule, run fresh or compiled.
struct Loop<const N: usize, const M: usize> {
    zero: A,
}

impl<const N: usize, const M: usize> Loop<N, M> {
    fn new(ctx: &Context, dims: &[usize]) -> Self {
        Loop { zero: zeros(ctx, dims) }
    }

    fn step(&self, s: &[A; M]) -> [A; M] {
        let q: [A; N] = std::array::from_fn(|k| s[k].clone());
        let (un, rn) = implicit_step(&q, &s[N + 1], &self.zero, 2);
        pack(un, rn, (&s[N + 1] * GROWTH).minimum(CFL_MAX))
    }

    fn until<'a>(steps: usize) -> Until<'a, [A; M]> {
        Until::new(steps).record("res", |s: &[A; M]| s[N].clone()).stop_when(|h| h["res"].last().lt(1e-20))
    }

    fn fresh(&self, s0: &[A; M], steps: usize) -> Run<[A; M]> {
        let run = iterate_until_all(s0.clone(), Self::until(steps), |s, _| self.step(s)).unwrap();
        assert_eq!(run.steps, steps, "iterate_until_all stopped early");
        run
    }

    fn compile(&self, s0: &[A; M], steps: usize) -> CompiledUntil<[A; M]> {
        compile_until_all(s0, Self::until(steps), |s, _| self.step(s)).unwrap()
    }
}

fn implicit_loop<const N: usize, const M: usize>(h: &mut Harness, z: &Z, group: &str) {
    let d = N - 2;
    let (small, big, steps_s, steps_l) = if d == 2 { (z.s2.to_vec(), z.l2.to_vec(), z.steps(20), z.steps(10)) } else { (z.s3.to_vec(), z.l3.to_vec(), z.steps(10), z.steps(5)) };
    h.guarded(group, |h| {
        let ctx = h.ctx.clone();
        // Host: the same ramp, `vsteps` implicit steps.
        let vn: Vec<usize> = if d == 2 { vec![24, 18] } else { vec![12, 10, 8] };
        let vsteps = 4;
        let hv = init_host::<N>(&vn, 1.0);
        let mut cells: Vec<Cell<N>> = host::to_cells(&hv);
        let mut cfl = 2.0f32;
        for _ in 0..vsteps {
            cells = host::h_implicit_step(&cells, Grid::new(&vn), &walls, f64::from(cfl), 2);
            cfl = (cfl * GROWTH).min(CFL_MAX);
        }
        let lp = Loop::<N, M>::new(&ctx, &vn);
        let s0 = init_state::<N, M>(&ctx, &vn, 1.0, 2.0);
        let run = lp.fresh(&s0, vsteps);
        let got: Vec<Vec<f32>> = (0..N).map(|k| run.state[k].to_vec().unwrap()).collect();
        check_rec(group, &got, &host::from_cells(&cells), 5e-3, 5e-3);
        // The same loop compiled: identical to the fresh one.
        let mut cu = lp.compile(&s0, vsteps);
        let crun = cu.run(s0.clone()).unwrap();
        let cgot: Vec<Vec<f32>> = (0..N).map(|k| crun.state[k].to_vec().unwrap()).collect();
        check_rec(&format!("{group} compiled"), &cgot, &got, 1e-4, 1e-5);
        for (dims, steps) in [(&small, steps_s), (&big, steps_l)] {
            let lp = Loop::<N, M>::new(&ctx, dims);
            let s0 = init_state::<N, M>(&ctx, dims, 1.0, 2.0);
            // The big grids have no host reference: on the first call the density must be finite and positive after the run.
            let first = std::cell::Cell::new(true);
            let mut f = || {
                let state = lp.fresh(&s0, steps).state;
                force_state(&state);
                if first.replace(false) {
                    let rho_min = state[0].min().item().unwrap();
                    assert!(rho_min.is_finite() && rho_min > 0.0, "{group}: minimum density {rho_min} after {steps} steps on {}", dims_str(dims));
                }
            };
            h.case(group, &format!("{}:{steps}st", dims_str(dims)), Work::Cells((cells_of(dims) * steps) as f64), &mut f);
        }
    });
}

fn converge(h: &mut Harness, z: &Z) {
    // The explicit first-order Euler loop with a norm record of the density residual (the loop of the old bench_pde).
    h.guarded("converge/euler2d_until", |h| {
        let ctx = h.ctx.clone();
        let vn = [20usize, 16];
        let hv = init_host::<4>(&vn, 1.0);
        let want = host::h_explicit_run::<4>(&hv, Grid::new(&vn), &|a, high| a == 0 && !high, false, f64::from(DT), 10);
        let run_until = |q: &[A; 4], steps: usize| {
            let run = iterate_until_all(q.clone(), Until::new(steps).record("res", |s: &[A; 4]| density_residual(s, false, true)).stop_when(|h| h["res"].last().lt(-1.0)), |s, _| explicit_step(s, false, true)).unwrap();
            assert_eq!(run.steps, steps, "iterate_until_all stopped early");
            run.state
        };
        check_rec("converge/euler2d_until", &download(&run_until(&upload::<4>(&ctx, &hv, &vn), 10)), &want, 2e-3, 2e-3);
        for (dims, steps) in [(z.s2.to_vec(), z.steps(100)), (z.l2.to_vec(), z.steps(30))] {
            let q = upload::<4>(&ctx, &init_host::<4>(&dims, 1.0), &dims);
            let mut f = || eval_rec(&run_until(&q, steps));
            h.case("converge/euler2d_until", &format!("{}:{steps}st", dims_str(&dims)), Work::Cells((cells_of(&dims) * steps) as f64), &mut f);
        }
    });
    implicit_loop::<4, 6>(h, z, "converge/implicit2d_until");
    implicit_loop::<5, 7>(h, z, "converge/implicit3d_until");
    fixed_costs(h, z);
}

/// The fixed cost per point of a run: several points (initial states and CFL, the same shapes) run for a few steps each, with
/// a fresh trace per point (`iterate_until_all`), with one compiled loop reused (`compile_until_all`), with a compile per call,
/// and the chunked loop of a solver that looks at the state between chunks.
fn fixed_costs(h: &mut Harness, z: &Z) {
    h.guarded("converge/points", |h| {
        let ctx = h.ctx.clone();
        let dims = z.s2.to_vec();
        let (points, per_point) = (4usize, z.steps(10));
        let lp = Loop::<4, 6>::new(&ctx, &dims);
        let starts: Vec<[A; 6]> = (0..points).map(|p| init_state::<4, 6>(&ctx, &dims, 0.6 + 0.2 * p as f64, 1.0 + 0.5 * p as f32)).collect();
        starts.iter().for_each(force_state);
        // Verification: the compiled loop on every point equals the fresh loop.
        let mut cu = lp.compile(&starts[0], per_point);
        for s in &starts {
            let (fresh, comp) = (lp.fresh(s, per_point), cu.run(s.clone()).unwrap());
            assert_eq!(comp.steps, per_point, "compiled loop stopped early");
            check_rec("converge/points", &download::<4>(&std::array::from_fn(|k| comp.state[k].clone())), &download::<4>(&std::array::from_fn(|k| fresh.state[k].clone())), 1e-4, 1e-5);
        }
        let size = format!("{}:{points}x{per_point}st", dims_str(&dims));
        let work = Work::Cells((cells_of(&dims) * per_point * points) as f64);
        let mut f = || {
            for s in &starts {
                force_state(&lp.fresh(s, per_point).state);
            }
        };
        h.case("converge/points_fresh", &size, work, &mut f);
        let mut f = || {
            for s in &starts {
                force_state(&cu.run(s.clone()).unwrap().state);
            }
        };
        h.case("converge/points_compiled", &size, work, &mut f);
        let mut f = || {
            let mut cu = lp.compile(&starts[0], per_point);
            for s in &starts {
                force_state(&cu.run(s.clone()).unwrap().state);
            }
        };
        h.case("converge/points_compile_each", &size, work, &mut f);
        let mut f = || {
            drop(lp.compile(&starts[0], per_point));
        };
        h.build_case("converge/points_compile_only", &dims_str(&dims), Work::None, &mut f);
        // Chunks: 5 runs of `per_point` steps, each from the state the last one returned.
        let chunks = 5;
        let size = format!("{}:{chunks}x{per_point}st", dims_str(&dims));
        let work = Work::Cells((cells_of(&dims) * per_point * chunks) as f64);
        let mut f = || {
            let mut s = starts[0].clone();
            for _ in 0..chunks {
                s = lp.fresh(&s, per_point).state;
            }
            force_state(&s);
        };
        h.case("converge/points_chunks_fresh", &size, work, &mut f);
        let mut f = || {
            let mut s = starts[0].clone();
            for _ in 0..chunks {
                s = cu.run(s).unwrap().state;
            }
            force_state(&s);
        };
        h.case("converge/points_chunks_compiled", &size, work, &mut f);
    });
}

// =================================================================================================
// Pressure Poisson: Jacobi, red-black, conjugate gradients
// =================================================================================================

/// `lap u = sum of the neighbours - 2 d u` with zero outside the grid.
fn laplacian(u: &A) -> A {
    let n: Vec<usize> = u.shape().dims().iter().copied().filter(|&e| e > 1).collect();
    let d = n.len();
    let p = pad1(u, d, PadMode::Zero);
    let mut sum: Option<A> = None;
    for a in 0..d {
        for v in neighbours(&p, a, &n) {
            sum = Some(match sum {
                None => v,
                Some(s) => s + v,
            });
        }
    }
    sum.unwrap() - 2.0 * d as f32 * u
}

/// A weighted Jacobi sweep of `-lap u = b`.
fn jacobi_sweep(u: &A, b: &A, omega: f32, d: usize) -> A {
    u + omega / (2.0 * d as f32) * (b + laplacian(u))
}

fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

fn poisson(h: &mut Harness, z: &Z) {
    let omega = 0.8f32;
    for (d, small, big, sweeps) in [(2usize, z.s2.to_vec(), z.l2.to_vec(), z.steps(100)), (3, z.s3.to_vec(), z.l3.to_vec(), z.steps(50))] {
        let group = format!("poisson/jacobi{d}d");
        h.guarded(&group, |h| {
            let ctx = h.ctx.clone();
            let vn: Vec<usize> = if d == 2 { vec![24, 20] } else { vec![12, 10, 8] };
            let hb = values(cells_of(&vn), 21, 0.0, 1.0);
            let (b, u) = (Array::from_slice_on(&ctx, &hb, shape(&vn)).unwrap(), Array::<f32>::zeros_on(&ctx, shape(&vn)).unwrap());
            let [got] = iterate_all([&u], 20, |s| [jacobi_sweep(&s[0], &b, omega, d)]);
            let mut want = vec![0.0; hb.len()];
            host::h_jacobi(&mut want, &to_f64(&hb), Grid::new(&vn), f64::from(omega), 20);
            check(&group, &got.to_vec().unwrap(), &to_f32(&want), 1e-3, 1e-4);
            for dims in [&small, &big] {
                let b = Array::from_slice_on(&ctx, &values(cells_of(dims), 22, 0.0, 1.0), shape(dims)).unwrap();
                let u = Array::<f32>::zeros_on(&ctx, shape(dims)).unwrap();
                b.eval().unwrap();
                u.eval().unwrap();
                let mut f = || {
                    let [r] = iterate_all([&u], sweeps, |s| [jacobi_sweep(&s[0], &b, omega, d)]);
                    r.eval().unwrap();
                };
                h.case(&group, &format!("{}:{sweeps}sw", dims_str(dims)), Work::Cells((cells_of(dims) * sweeps) as f64), &mut f);
            }
        });
    }

    // Red-black Gauss-Seidel: two masked half sweeps per iteration.
    h.guarded("poisson/redblack2d", |h| {
        let ctx = h.ctx.clone();
        let masks = |dims: &[usize]| -> (Array<bool>, Array<bool>) {
            let g = Grid::new(dims);
            let color = |c: usize| -> Vec<bool> { (0..g.cells()).map(|q| (g.coords(q).iter().sum::<usize>()) % 2 == c).collect() };
            (Array::from_slice_on(&ctx, &color(0), shape(dims)).unwrap(), Array::from_slice_on(&ctx, &color(1), shape(dims)).unwrap())
        };
        let sweep = |u: &A, b: &A, red: &Array<bool>, black: &Array<bool>| -> A {
            let a = select(red, jacobi_sweep(u, b, 1.0, 2), u);
            select(black, jacobi_sweep(&a, b, 1.0, 2), &a)
        };
        let iters = z.steps(50);
        let vn = [24usize, 20];
        let hb = values(cells_of(&vn), 23, 0.0, 1.0);
        let (red, black) = masks(&vn);
        let b = Array::from_slice_on(&ctx, &hb, shape(&vn)).unwrap();
        let u = Array::<f32>::zeros_on(&ctx, shape(&vn)).unwrap();
        let [got] = iterate_all([&u], 10, |s| [sweep(&s[0], &b, &red, &black)]);
        let mut want = vec![0.0; hb.len()];
        host::h_redblack(&mut want, &to_f64(&hb), Grid::new(&vn), 10);
        check("poisson/redblack2d", &got.to_vec().unwrap(), &to_f32(&want), 1e-3, 1e-4);
        for dims in [z.s2.to_vec(), z.l2.to_vec()] {
            let (red, black) = masks(&dims);
            let b = Array::from_slice_on(&ctx, &values(cells_of(&dims), 24, 0.0, 1.0), shape(&dims)).unwrap();
            let u = Array::<f32>::zeros_on(&ctx, shape(&dims)).unwrap();
            b.eval().unwrap();
            u.eval().unwrap();
            let mut f = || {
                let [r] = iterate_all([&u], iters, |s| [sweep(&s[0], &b, &red, &black)]);
                r.eval().unwrap();
            };
            h.case("poisson/redblack2d", &format!("{}:{iters}it", dims_str(&dims)), Work::Cells((cells_of(&dims) * iters * 2) as f64), &mut f);
        }
    });

    for (d, small, big) in [(2usize, z.s2.to_vec(), z.l2.to_vec()), (3, z.s3.to_vec(), z.l3.to_vec())] {
        let group = format!("poisson/cg{d}d");
        let iters = z.steps(30);
        h.guarded(&group, |h| {
            let ctx = h.ctx.clone();
            let vn: Vec<usize> = if d == 2 { vec![24, 20] } else { vec![12, 10, 8] };
            let vk = 12;
            let hb = values(cells_of(&vn), 25, 0.0, 1.0);
            let want = to_f32(&host::h_cg(&to_f64(&hb), Grid::new(&vn), vk));
            let b = Array::from_slice_on(&ctx, &hb, shape(&vn)).unwrap();
            // Tolerances are relative to the largest solution value: CG in single precision drifts from the exact iterates.
            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            check(&format!("{group}_readback"), &cg_readback(&b, vk, d).to_vec().unwrap(), &want, 2e-3, 2e-3 * scale);
            check(&format!("{group}_device"), &cg_device(&b, vk, d).to_vec().unwrap(), &want, 2e-3, 2e-3 * scale);
            for dims in [&small, &big] {
                let b = Array::from_slice_on(&ctx, &values(cells_of(dims), 26, 0.0, 1.0), shape(dims)).unwrap();
                b.eval().unwrap();
                let size = format!("{}:{iters}it", dims_str(dims));
                let work = Work::Cells((cells_of(dims) * iters) as f64);
                let mut f = || cg_readback(&b, iters, d).eval().unwrap();
                h.case(&format!("{group}_readback"), &size, work, &mut f);
                let mut f = || cg_device(&b, iters, d).eval().unwrap();
                h.case(&format!("{group}_device"), &size, work, &mut f);
            }
            // The loop as one `iterate_all` replay on device scalars.
            let b = Array::from_slice_on(&ctx, &hb, shape(&vn)).unwrap();
            check(&format!("{group}_iterate"), &cg_iterate(&b, vk, d).to_vec().unwrap(), &want, 2e-3, 2e-3 * scale);
            for dims in [&small, &big] {
                let b = Array::from_slice_on(&ctx, &values(cells_of(dims), 26, 0.0, 1.0), shape(dims)).unwrap();
                b.eval().unwrap();
                let mut f = || cg_iterate(&b, iters, d).eval().unwrap();
                h.case(&format!("{group}_iterate"), &format!("{}:{iters}it", dims_str(dims)), Work::Cells((cells_of(dims) * iters) as f64), &mut f);
            }
        });
    }
}

/// `-lap p`.
fn neg_lap(p: &A) -> A {
    -laplacian(p)
}

/// CG on `-lap x = b` with host-side scalars: two `item()` readbacks per iteration.
fn cg_readback(b: &A, iters: usize, _d: usize) -> A {
    let mut x = b * 0.0;
    let mut r = b * 1.0;
    let mut p = b * 1.0;
    let mut rr = r.norm().item().unwrap().powi(2);
    for _ in 0..iters {
        let ap = neg_lap(&p);
        let pap = (&p * &ap).sum().item().unwrap();
        let alpha = rr / pap;
        let (nx, nr) = (&x + alpha * &p, &r - alpha * &ap);
        let rr_new = (&nr * &nr).sum().item().unwrap();
        let beta = rr_new / rr;
        let np = &nr + beta * &p;
        eval_all(&[&nx, &nr, &np]).unwrap();
        (x, r, p, rr) = (nx, nr, np, rr_new);
    }
    x
}

/// CG with the scalars kept on the device as arrays of one element, evaluated every iteration without a readback.
fn cg_device(b: &A, iters: usize, _d: usize) -> A {
    let mut x = b * 0.0;
    let mut r = b * 1.0;
    let mut p = b * 1.0;
    let mut rr = (&r * &r).sum();
    for _ in 0..iters {
        let ap = neg_lap(&p);
        let alpha = &rr / (&p * &ap).sum();
        let (nx, nr) = (&x + &alpha * &p, &r - &alpha * &ap);
        let rr_new = (&nr * &nr).sum();
        let np = &nr + (&rr_new / &rr) * &p;
        eval_all(&[&nx, &nr, &np, &rr_new]).unwrap();
        (x, r, p, rr) = (nx, nr, np, rr_new);
    }
    x
}

/// CG as one `iterate_all` loop over `[x, r, p, rr]` (`rr` of one element).
fn cg_iterate(b: &A, iters: usize, _d: usize) -> A {
    let (x, r, p) = (b * 0.0, b * 1.0, b * 1.0);
    let rr = (&r * &r).sum();
    let [x, ..] = iterate_all([&x, &r, &p, &rr], iters, |s| {
        let ap = neg_lap(&s[2]);
        let alpha = &s[3] / (&s[2] * &ap).sum();
        let (nx, nr) = (&s[0] + &alpha * &s[2], &s[1] - &alpha * &ap);
        let rr_new = (&nr * &nr).sum();
        let np = &nr + (&rr_new / &s[3]) * &s[2];
        [nx, nr, np, rr_new]
    });
    x
}

// =================================================================================================
// Multigrid transfers and cycles
// =================================================================================================

/// Average of every `2 x 2` block through the reshape `[2, nx, 2, ny]`.
fn restrict2d(f: &A) -> A {
    let d = f.shape().dims();
    let (nx, ny) = (d[0] / 2, d[1] / 2);
    let r = f.reshape([2, nx, 2, ny]);
    let pick = |a: usize, b: usize| r.slice(0, a..a + 1).slice(2, b..b + 1);
    ((pick(0, 0) + pick(1, 0)) + (pick(0, 1) + pick(1, 1))).reshape([nx, ny]) * 0.25
}

/// Piecewise-constant prolongation through a broadcast of `[1, nx, 1, ny]` to `[2, nx, 2, ny]`.
fn prolong2d_pc(c: &A) -> A {
    let d = c.shape().dims();
    let (nx, ny) = (d[0], d[1]);
    c.reshape([1, nx, 1, ny]).broadcast_to([2, nx, 2, ny]).reshape([2 * nx, 2 * ny])
}

/// Bilinear prolongation (9/16, 3/16, 3/16, 1/16) with edge replication; the four parities are joined through `concat`.
fn prolong2d_bilinear(c: &A) -> A {
    let d = c.shape().dims();
    let (nx, ny) = (d[0], d[1]);
    // The neighbour of an even (odd) fine cell is the coarse cell below (above).
    let part = |a: usize, b: usize| -> A {
        let (sx, sy) = (if a == 0 { 1isize } else { -1 }, if b == 0 { 1isize } else { -1 });
        let v = (9.0 * c + 3.0 * c.shift_clamp([sx, 0]) + 3.0 * c.shift_clamp([0, sy]) + c.shift_clamp([sx, sy])) * (1.0 / 16.0);
        v.reshape([1, nx, 1, ny])
    };
    let rows: Vec<A> = (0..2).map(|a| concat(2, [&part(a, 0), &part(a, 1)])).collect();
    concat(0, [&rows[0], &rows[1]]).reshape([2 * nx, 2 * ny])
}

/// One factor 2 along `axis` of a field of extents `dims` (halving by the mean of pairs).
fn halve(a: &A, dims: &[usize], axis: usize) -> (A, Vec<usize>) {
    let (below, above): (usize, usize) = (dims[..axis].iter().product(), dims[axis + 1..].iter().product());
    let m = dims[axis] / 2;
    let r = a.reshape([below, 2, m, above]);
    let mut out = dims.to_vec();
    out[axis] = m;
    ((r.slice(1, 0..1) + r.slice(1, 1..2)).reshape(shape(&out)) * 0.5, out)
}

fn restrict3d(f: &A) -> A {
    let mut dims: Vec<usize> = f.shape().dims()[..3].to_vec();
    let mut a = f.clone();
    for axis in 0..3 {
        (a, dims) = halve(&a, &dims, axis);
    }
    a
}

/// Doubling by repetition along every axis in turn.
fn prolong3d_pc(c: &A) -> A {
    let mut dims: Vec<usize> = c.shape().dims()[..3].to_vec();
    let mut a = c.clone();
    for axis in 0..3 {
        let (below, above): (usize, usize) = (dims[..axis].iter().product(), dims[axis + 1..].iter().product());
        let m = dims[axis];
        dims[axis] = 2 * m;
        a = a.reshape([below, 1, m, above]).broadcast_to([below, 2, m, above]).reshape(shape(&dims));
    }
    a
}

/// One V-cycle of weighted Jacobi (`pre`/`post` sweeps) with the transfers above, `levels` deep; every sweep is evaluated.
fn vcycle(u: &A, b: &A, levels: usize) -> A {
    let (omega, pre, post, coarse) = (0.8f32, 2, 2, 8);
    let smooth = |u: &A, sweeps: usize| -> A {
        let mut u = u.clone();
        for _ in 0..sweeps {
            u = jacobi_sweep(&u, b, omega, 2).cache();
        }
        u
    };
    if levels == 1 {
        return smooth(u, coarse);
    }
    let u = smooth(u, pre);
    let res = (b + laplacian(&u)).cache();
    let rc = (restrict2d(&res) * 4.0).cache();
    let e = vcycle(&(&rc * 0.0).cache(), &rc, levels - 1);
    let u = (&u + prolong2d_pc(&e)).cache();
    smooth(&u, post)
}

fn multigrid(h: &mut Harness, z: &Z) {
    let big2 = z.l2.to_vec();
    let small2 = z.v2.to_vec();
    // 2-D restriction and prolongation.
    for (name, kind) in [("mg/restrict2d", 0), ("mg/prolong2d_pc", 1), ("mg/prolong2d_bilinear", 2)] {
        h.guarded(name, |h| {
            let ctx = h.ctx.clone();
            let transfer = |f: &A| -> A {
                match kind {
                    0 => restrict2d(f),
                    1 => prolong2d_pc(f),
                    _ => prolong2d_bilinear(f),
                }
            };
            let vn = [16usize, 12];
            let hv = values(cells_of(&vn), 31, 0.0, 1.0);
            let want: Vec<f64> = match kind {
                0 => host::h_restrict(&to_f64(&hv), Grid::new(&vn)),
                1 => host::h_prolong_pc(&to_f64(&hv), Grid::new(&[2 * vn[0], 2 * vn[1]])),
                _ => host::h_prolong_bilinear(&to_f64(&hv), Grid::new(&vn)),
            };
            let got = transfer(&Array::from_slice_on(&ctx, &hv, shape(&vn)).unwrap()).to_vec().unwrap();
            check(name, &got, &to_f32(&want), 1e-5, 1e-6);
            for dims in [&small2, &big2] {
                // `dims` is the fine grid of a restriction and the coarse grid of a prolongation (a quarter or a fourth of the cells).
                let src: Vec<usize> = if kind == 0 { dims.clone() } else { dims.iter().map(|n| n / 2).collect() };
                let f = Array::from_slice_on(&ctx, &values(cells_of(&src), 32, 0.0, 1.0), shape(&src)).unwrap();
                f.eval().unwrap();
                let fine = cells_of(dims);
                let mut run = || transfer(&f).eval().unwrap();
                h.case(name, &dims_str(dims), Work::Bytes(((fine + fine / 4) * 4) as f64), &mut run);
            }
        });
    }
    // 3-D transfers one axis at a time.
    for (name, restrict) in [("mg/restrict3d", true), ("mg/prolong3d_pc", false)] {
        h.guarded(name, |h| {
            let ctx = h.ctx.clone();
            let vn = [8usize, 6, 4];
            let hv = values(cells_of(&vn), 33, 0.0, 1.0);
            let f = Array::from_slice_on(&ctx, &hv, shape(&vn)).unwrap();
            let (got, want) = if restrict {
                (restrict3d(&f).to_vec().unwrap(), host::h_restrict(&to_f64(&hv), Grid::new(&vn)))
            } else {
                (prolong3d_pc(&f).to_vec().unwrap(), host::h_prolong_pc(&to_f64(&hv), Grid::new(&[16, 12, 8])))
            };
            check(name, &got, &to_f32(&want), 1e-5, 1e-6);
            for dims in [z.s3.to_vec(), z.l3.to_vec()] {
                let src: Vec<usize> = if restrict { dims.clone() } else { dims.iter().map(|n| n / 2).collect() };
                let f = Array::from_slice_on(&ctx, &values(cells_of(&src), 34, 0.0, 1.0), shape(&src)).unwrap();
                f.eval().unwrap();
                let fine = cells_of(&dims);
                let mut run = || if restrict { restrict3d(&f).eval().unwrap() } else { prolong3d_pc(&f).eval().unwrap() };
                h.case(name, &dims_str(&dims), Work::Bytes(((fine + fine / 8) * 4) as f64), &mut run);
            }
        });
    }
    h.guarded("mg/vcycle2d", |h| {
        let ctx = h.ctx.clone();
        let cycles = z.steps(5);
        let cycle_run = |b: &A, n: usize, levels: usize| -> A {
            let mut u = b * 0.0;
            for _ in 0..n {
                u = vcycle(&u, b, levels);
            }
            u
        };
        let vn = [32usize, 24];
        let hb = values(cells_of(&vn), 35, 0.0, 1.0);
        let got = cycle_run(&Array::from_slice_on(&ctx, &hb, shape(&vn)).unwrap(), 2, 3).to_vec().unwrap();
        let (hb64, g) = (to_f64(&hb), Grid::new(&vn));
        let mut want = vec![0.0; hb.len()];
        for _ in 0..2 {
            host::h_vcycle(&mut want, &hb64, g, 3, 0.8, 2, 2, 8);
        }
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())) as f32;
        check("mg/vcycle2d", &got, &to_f32(&want), 2e-3, 2e-3 * scale);
        for dims in [&small2, &big2] {
            let b = Array::from_slice_on(&ctx, &values(cells_of(dims), 36, 0.0, 1.0), shape(dims)).unwrap();
            b.eval().unwrap();
            let mut f = || cycle_run(&b, cycles, 3).eval().unwrap();
            h.case("mg/vcycle2d", &format!("{}:{cycles}cy", dims_str(dims)), Work::Cells((cells_of(dims) * cycles) as f64), &mut f);
        }
    });
}
