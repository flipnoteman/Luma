//! Benchmarks of the array-level PDE features: views, `assign`/`put`, records, lazy reductions,
//! `iterate_all` replay, `iterate_until_all`, line primitives (scans, tridiagonal solves), `Df32`/`f64`.
//!
//! `cargo run --release --example bench_pde -- [--quick] [--only <substring>]`
//!
//! Every case is checked once against a host (f64 where it matters) reference, or the reference
//! stated at the case, before it is timed. A case that fails, panics or mismatches is reported as a
//! `SKIPPED: <case>: <reason>` line and the run continues. Throughput figures count the minimal
//! data an ideal kernel would touch (reads plus writes), not what the implementation really moves.
//! Cases marked `[P]` in the task also get one extra profiled run (`kernels`, `gpu ms`) and a
//! per-kernel breakdown after the table.

use luma::*;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Mutex;
use std::time::Instant;

// =================================================================================================
// Harness
// =================================================================================================

static PANIC: Mutex<String> = Mutex::new(String::new());

enum Tp {
    /// Cell updates (cells x steps) -> Mcell-steps/s.
    Mcell(f64),
    /// Bytes touched -> GB/s.
    Bytes(f64),
    /// `ops` repetitions inside one call -> time per op in the unit.
    PerOp(f64, &'static str),
}

struct Bench {
    ctx: Context,
    quick: bool,
    only: Option<String>,
    profiles: Vec<(String, String, Profile)>,
    skipped: Vec<(String, String)>,
}

impl Bench {
    fn want(&self, name: &str) -> bool {
        self.only.as_deref().is_none_or(|f| name.contains(f))
    }

    /// Runs `f` if the case is selected; a panic (library failure or verification mismatch) is
    /// reported as a SKIPPED line.
    fn guarded(&mut self, name: &str, f: impl FnOnce(&mut Bench)) {
        if !self.want(name) {
            return;
        }
        let result = catch_unwind(AssertUnwindSafe(|| f(self)));
        if result.is_err() {
            let msg = PANIC.lock().unwrap().clone();
            let msg = msg.replace('\n', " | ");
            println!("SKIPPED: {name}: {msg}");
            self.skipped.push((name.to_string(), msg));
            let _ = self.ctx.finish_profiling();
            let _ = self.ctx.sync();
        }
    }

    fn skip(&mut self, name: &str, why: &str) {
        if self.want(name) {
            println!("SKIPPED: {name}: {why}");
            self.skipped.push((name.to_string(), why.to_string()));
        }
    }

    /// (median ms, min ms): 2 warm-up calls, then 7+ timed calls (3 when a call takes over 1 s).
    fn measure(&self, f: &mut dyn FnMut()) -> (f64, f64) {
        let ctx = &self.ctx;
        let run = |f: &mut dyn FnMut()| {
            let t = Instant::now();
            f();
            ctx.sync().unwrap();
            t.elapsed().as_secs_f64() * 1e3
        };
        let w = run(f).max(run(f));
        let reps = if w > 1000.0 {
            3
        } else if w < 10.0 && !self.quick {
            21
        } else {
            7
        };
        let mut s: Vec<f64> = (0..reps).map(|_| run(f)).collect();
        s.sort_by(f64::total_cmp);
        (s[s.len() / 2], s[0])
    }

    fn time(&mut self, name: &str, size: &str, tp: Tp, profiled: bool, f: &mut dyn FnMut()) {
        let (ms, min) = self.measure(f);
        let (rate, unit) = match tp {
            Tp::Mcell(c) => (c / (ms * 1e-3) / 1e6, "Mcell-st/s"),
            Tp::Bytes(b) => (b / (ms * 1e-3) / 1e9, "GB/s"),
            Tp::PerOp(n, u) => (ms * 1e3 / n, u),
        };
        let rate = if rate < 10.0 { format!("{rate:.3}") } else { format!("{rate:.1}") };
        let (kernels, gpu) = if profiled {
            match self.ctx.profile(|| f()) {
                Ok(((), p)) => {
                    let row = (p.launches().to_string(), format!("{:.3}", p.total().as_secs_f64() * 1e3));
                    self.profiles.push((name.to_string(), size.to_string(), p));
                    row
                }
                Err(e) => (format!("err"), format!("{e}")),
            }
        } else {
            ("-".to_string(), "-".to_string())
        };
        println!("{name:<28} {size:>11} {ms:>10.3} {min:>10.3} {rate:>10} {unit:<11} {kernels:>7} {gpu:>9}");
    }
}

fn check(name: &str, got: &[f32], want: &[f32], rel: f32, abs: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = abs.max(rel * w.abs());
        assert!((g - w).abs() <= tol, "{name}: index {i}: got {g}, want {w} (tol {tol})");
    }
}

fn check_scalar(name: &str, got: f64, want: f64, rel: f64) {
    assert!((got - want).abs() <= rel * want.abs(), "{name}: got {got}, want {want} (rel {rel})");
}

fn values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / (1u64 << 24) as f32;
            lo + unit * (hi - lo)
        })
        .collect()
}

fn permutation(n: usize, seed: u64) -> Vec<u32> {
    let mut p: Vec<u32> = (0..n as u32).collect();
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    for i in (1..n).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        p.swap(i, (s % (i as u64 + 1)) as usize);
    }
    p
}

fn shp(w: usize, h: usize) -> Shape {
    if h == 1 { Shape::from(w) } else { Shape::from([w, h]) }
}

fn size_str(w: usize, h: usize) -> String {
    if h == 1 { format!("{w}") } else { format!("{w}x{h}") }
}

// =================================================================================================
// Generic data and the "array result" case runner (views, scans, ...)
// =================================================================================================

struct Dat {
    w: usize,
    h: usize,
    a: Array<f32>,
    b: Array<f32>,
    idx: Array<u32>,
    ha: Vec<f32>,
    hb: Vec<f32>,
    hidx: Vec<u32>,
}

impl Dat {
    fn new(ctx: &Context, w: usize, h: usize, need_b: bool, need_idx: bool) -> Dat {
        let n = w * h;
        let ha = values(n, 1, 0.2, 1.2);
        let hb = if need_b { values(n, 2, 0.2, 1.2) } else { vec![0.0] };
        let hidx = if need_idx { permutation(n, 3) } else { vec![0] };
        let a = Array::from_slice_on(ctx, &ha, shp(w, h)).unwrap();
        let b = if need_b { Array::from_slice_on(ctx, &hb, shp(w, h)).unwrap() } else { Array::from_slice_on(ctx, &hb, 1).unwrap() };
        let idx = if need_idx { Array::from_slice_on(ctx, &hidx, shp(w, h)).unwrap() } else { Array::from_slice_on(ctx, &hidx, 1).unwrap() };
        a.eval().unwrap();
        Dat { w, h, a, b, idx, ha, hb, hidx }
    }
}

struct Arr<'a> {
    name: &'a str,
    big: [usize; 2],
    small: [usize; 2],
    need_b: bool,
    need_idx: bool,
    /// Elements an ideal kernel touches, in units of `w * h`.
    touched: f64,
    build: &'a dyn Fn(&Dat) -> Array<f32>,
    host: &'a dyn Fn(&Dat) -> Vec<f32>,
    tol: (f32, f32),
    prof: bool,
}

impl Bench {
    fn arr(&mut self, c: Arr<'_>) {
        let name = c.name.to_string();
        self.guarded(&name, |b| {
            let ctx = b.ctx.clone();
            {
                let d = Dat::new(&ctx, c.small[0], c.small[1], c.need_b, c.need_idx);
                let got = (c.build)(&d).to_vec().unwrap();
                check(c.name, &got, &(c.host)(&d), c.tol.0, c.tol.1);
            }
            let d = Dat::new(&ctx, c.big[0], c.big[1], c.need_b, c.need_idx);
            let bytes = c.touched * (d.w * d.h) as f64 * 4.0;
            let mut f = || (c.build)(&d).eval().unwrap();
            b.time(c.name, &size_str(d.w, d.h), Tp::Bytes(bytes), c.prof, &mut f);
        });
    }
}

// =================================================================================================
// Euler solver
// =================================================================================================

const GAMMA: f32 = 1.4;
const DT: f32 = 0.1;

#[derive(Fields)]
struct Cons {
    rho: Array<f32>,
    mu: Array<f32>,
    mv: Array<f32>,
    e: Array<f32>,
}

#[derive(Fields)]
struct Prim {
    rho: Array<f32>,
    u: Array<f32>,
    v: Array<f32>,
    p: Array<f32>,
}

#[derive(Clone, Copy)]
struct Opts {
    /// Mirror wall on the low-i ghosts (otherwise the ghosts keep the edge values).
    wall: bool,
    /// `cache()` the padded primitive record after the ghost assign.
    cache: bool,
    /// MUSCL reconstruction (2 ghosts) instead of first order (1 ghost).
    muscl: bool,
}

type A = Array<f32>;

fn prim4(rho: &A, mu: &A, mv: &A, e: &A) -> Prim {
    let u = mu / rho;
    let v = mv / rho;
    let p = (GAMMA - 1.0) * (e - 0.5 * rho * (&u * &u + &v * &v));
    Prim { rho: rho * 1.0, u, v, p }
}

fn cons(w: &Prim) -> Cons {
    let ke = 0.5 * &w.rho * (&w.u * &w.u + &w.v * &w.v);
    Cons { rho: &w.rho * 1.0, mu: &w.rho * &w.u, mv: &w.rho * &w.v, e: &w.p / (GAMMA - 1.0) + ke }
}

fn flux(w: &Prim, dir: usize) -> Cons {
    let q = cons(w);
    if dir == 0 {
        Cons { rho: &q.mu * 1.0, mu: &q.mu * &w.u + &w.p, mv: &q.mv * &w.u, e: (&q.e + &w.p) * &w.u }
    } else {
        Cons { rho: &q.mv * 1.0, mu: &q.mu * &w.v, mv: &q.mv * &w.v + &w.p, e: (&q.e + &w.p) * &w.v }
    }
}

fn mirror(w: Prim) -> Prim {
    Prim { rho: w.rho, u: -&w.u, v: w.v, p: w.p }
}

fn rusanov(l: &Prim, r: &Prim, dir: usize) -> Cons {
    let speed = |w: &Prim| (if dir == 0 { &w.u } else { &w.v }).abs() + (GAMMA * &w.p / &w.rho).sqrt();
    let smax = speed(l).maximum(speed(r));
    let (ul, ur) = (cons(l), cons(r));
    let (fl, fr) = (flux(l, dir), flux(r, dir));
    0.5 * (&fl + &fr) - 0.5 * &smax * (&ur - &ul)
}

fn minmod(a: &A, b: &A) -> A {
    let smaller = select(&a.abs().lt(&b.abs()), a, b);
    select(&(a * b).gt(0.0), &smaller, 0.0)
}

/// Face states of a MUSCL reconstruction along `axis`, flux in direction `dir`; `w` is padded by 2
/// along `axis` and restricted to the interior along the other axis; `n` interior cells.
fn muscl_faces(w: &Prim, axis: usize, n: usize, dir: usize) -> Cons {
    let s = |k: usize| w.slice(axis, k..k + n + 1);
    let (a, b, c, d) = (s(0), s(1), s(2), s(3));
    let sl = (&b - &a).zip(&(&c - &b), |x, y| minmod(x, y));
    let sr = (&c - &b).zip(&(&d - &c), |x, y| minmod(x, y));
    let l = &b + 0.5 * &sl;
    let r = &c - 0.5 * &sr;
    rusanov(&l, &r, dir)
}

fn ghosted(w: Prim, g: usize, o: Opts) -> Prim {
    let w = w.pad([(g, g), (g, g)], PadMode::Edge);
    if o.wall {
        let src = if g == 1 { w.slice(0, 1..2) } else { w.slice(0, 2..4).flip(0) };
        w.slice(0, ..g).assign(mirror(src)).unwrap();
    }
    if o.cache { w.cache() } else { w }
}

fn residual(q: [&A; 4], o: Opts) -> Cons {
    let d = q[0].shape().dims();
    let (ni, nj) = (d[0], d[1]);
    let w0 = prim4(q[0], q[1], q[2], q[3]);
    let (fi, fj) = if o.muscl {
        let w = ghosted(w0, 2, o);
        (muscl_faces(&w.slice(1, 2..nj + 2), 0, ni, 0), muscl_faces(&w.slice(0, 2..ni + 2), 1, nj, 1))
    } else {
        let w = ghosted(w0, 1, o);
        let wi = w.slice(1, 1..nj + 1);
        let wj = w.slice(0, 1..ni + 1);
        (
            rusanov(&wi.slice(0, 0..ni + 1), &wi.slice(0, 1..ni + 2), 0),
            rusanov(&wj.slice(1, 0..nj + 1), &wj.slice(1, 1..nj + 2), 1),
        )
    };
    -(fi.slice(0, 1..) - fi.slice(0, ..ni)) - (fj.slice(1, 1..) - fj.slice(1, ..nj))
}

fn step_refs(q: [&A; 4], o: Opts) -> Cons {
    let r = residual(q, o);
    Cons { rho: q[0] + DT * &r.rho, mu: q[1] + DT * &r.mu, mv: q[2] + DT * &r.mv, e: q[3] + DT * &r.e }
}

fn step(q: &Cons, o: Opts) -> Cons {
    step_refs([&q.rho, &q.mu, &q.mv, &q.e], o)
}

fn init_host(ni: usize, nj: usize) -> [Vec<f32>; 4] {
    let n = ni * nj;
    let mut out: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(n));
    let sigma = 0.15 * ni.max(nj) as f64;
    let g = f64::from(GAMMA);
    for j in 0..nj {
        for i in 0..ni {
            let (x, y) = (i as f64, j as f64);
            let bump = |cx: f64, cy: f64| (-((x - cx).powi(2) + (y - cy).powi(2)) / (2.0 * sigma * sigma)).exp();
            let rho = 1.0 + 0.2 * bump(0.5 * ni as f64, 0.5 * nj as f64);
            let p = 1.0 + 0.2 * bump(0.3 * ni as f64, 0.6 * nj as f64);
            let u = 0.1 * (2.0 * std::f64::consts::PI * x / ni as f64).sin();
            let v = 0.05 * (2.0 * std::f64::consts::PI * y / nj as f64).cos();
            out[0].push(rho as f32);
            out[1].push((rho * u) as f32);
            out[2].push((rho * v) as f32);
            out[3].push((p / (g - 1.0) + 0.5 * rho * (u * u + v * v)) as f32);
        }
    }
    out
}

fn upload(ctx: &Context, h: &[Vec<f32>; 4], ni: usize, nj: usize) -> Cons {
    let up = |k: usize| Array::from_slice_on(ctx, &h[k], [ni, nj]).unwrap();
    let q = Cons { rho: up(0), mu: up(1), mv: up(2), e: up(3) };
    q.eval().unwrap();
    q
}

fn host_of(q: &Cons) -> [Vec<f32>; 4] {
    [q.rho.to_vec().unwrap(), q.mu.to_vec().unwrap(), q.mv.to_vec().unwrap(), q.e.to_vec().unwrap()]
}

// ---- the same in f64 on the host ----

type Cell = [f64; 4];

fn h_prim(q: Cell) -> Cell {
    let g = f64::from(GAMMA);
    let (u, v) = (q[1] / q[0], q[2] / q[0]);
    [q[0], u, v, (g - 1.0) * (q[3] - 0.5 * q[0] * (u * u + v * v))]
}

fn h_cons(w: Cell) -> Cell {
    let g = f64::from(GAMMA);
    [w[0], w[0] * w[1], w[0] * w[2], w[3] / (g - 1.0) + 0.5 * w[0] * (w[1] * w[1] + w[2] * w[2])]
}

fn h_flux(w: Cell, dir: usize) -> Cell {
    let q = h_cons(w);
    if dir == 0 {
        [q[1], q[1] * w[1] + w[3], q[2] * w[1], (q[3] + w[3]) * w[1]]
    } else {
        [q[2], q[1] * w[2], q[2] * w[2] + w[3], (q[3] + w[3]) * w[2]]
    }
}

fn h_rus(l: Cell, r: Cell, dir: usize) -> Cell {
    let g = f64::from(GAMMA);
    let speed = |w: Cell| w[1 + dir].abs() + (g * w[3] / w[0]).sqrt();
    let smax = speed(l).max(speed(r));
    let (ul, ur, fl, fr) = (h_cons(l), h_cons(r), h_flux(l, dir), h_flux(r, dir));
    std::array::from_fn(|k| 0.5 * (fl[k] + fr[k]) - 0.5 * smax * (ur[k] - ul[k]))
}

fn h_minmod(a: f64, b: f64) -> f64 {
    if a * b > 0.0 { if a.abs() < b.abs() { a } else { b } } else { 0.0 }
}

fn host_run(h: &[Vec<f32>; 4], ni: usize, nj: usize, steps: usize, o: Opts) -> [Vec<f32>; 4] {
    let n = ni * nj;
    let dt = f64::from(DT);
    let mut q: Vec<Cell> = (0..n).map(|i| std::array::from_fn(|k| f64::from(h[k][i]))).collect();
    for _ in 0..steps {
        let p: Vec<Cell> = q.iter().map(|&c| h_prim(c)).collect();
        let cellp = |c: isize, j: isize| -> Cell {
            let jj = j.clamp(0, nj as isize - 1) as usize;
            let at = |i: usize| p[i + ni * jj];
            if c < 0 {
                if o.wall {
                    let mut w = at((-c - 1) as usize);
                    w[1] = -w[1];
                    w
                } else {
                    at(0)
                }
            } else if c as usize >= ni {
                at(ni - 1)
            } else {
                at(c as usize)
            }
        };
        // Face between cells `a` and `a + 1` along the line `get`.
        let face = |get: &dyn Fn(isize) -> Cell, a: isize, dir: usize| -> Cell {
            let (l, r) = if o.muscl {
                let (m1, c0, c1, c2) = (get(a - 1), get(a), get(a + 1), get(a + 2));
                let sl: Cell = std::array::from_fn(|k| h_minmod(c0[k] - m1[k], c1[k] - c0[k]));
                let sr: Cell = std::array::from_fn(|k| h_minmod(c1[k] - c0[k], c2[k] - c1[k]));
                (std::array::from_fn(|k| c0[k] + 0.5 * sl[k]), std::array::from_fn(|k| c1[k] - 0.5 * sr[k]))
            } else {
                (get(a), get(a + 1))
            };
            h_rus(l, r, dir)
        };
        for j in 0..nj {
            for i in 0..ni {
                let gi = |c: isize| cellp(c, j as isize);
                let gj = |c: isize| cellp(i as isize, c);
                let (ii, jj) = (i as isize, j as isize);
                let (fih, fil) = (face(&gi, ii, 0), face(&gi, ii - 1, 0));
                let (fjh, fjl) = (face(&gj, jj, 1), face(&gj, jj - 1, 1));
                for k in 0..4 {
                    q[i + ni * j][k] += dt * (-(fih[k] - fil[k]) - (fjh[k] - fjl[k]));
                }
            }
        }
    }
    std::array::from_fn(|k| q.iter().map(|c| c[k] as f32).collect())
}

fn euler_variant(v: &str, init: &Cons, steps: usize, o: Opts) -> Cons {
    match v {
        "per_step" | "per_step_cache_w" => {
            let mut cur = step(init, o);
            cur.eval().unwrap();
            for _ in 1..steps {
                let next = step(&cur, o);
                next.eval().unwrap();
                cur = next;
            }
            cur
        }
        "iterate_all" => {
            let [rho, mu, mv, e] = iterate_all([&init.rho, &init.mu, &init.mv, &init.e], steps, |s| {
                let c = step_refs([&s[0], &s[1], &s[2], &s[3]], o);
                [c.rho, c.mu, c.mv, c.e]
            });
            let q = Cons { rho, mu, mv, e };
            q.eval().unwrap();
            q
        }
        "until_all" => {
            let run = iterate_until_all(
                init.map(|a| a * 1.0),
                Until::new(steps)
                    .record("res", move |s: &Cons| residual([&s.rho, &s.mu, &s.mv, &s.e], o).rho.norm())
                    .stop_when(|h| h["res"].last().lt(0.0)),
                move |s, _| step(s, o),
            )
            .unwrap();
            assert_eq!(run.steps, steps, "iterate_until_all stopped early");
            run.state.eval().unwrap();
            run.state
        }
        _ => unreachable!(),
    }
}

#[derive(encase::ShaderType)]
struct EulerParams {
    ni: u32,
    nj: u32,
    dt: f32,
    g: f32,
}

const EULER_WGSL: &str = r#"
struct P { ni: u32, nj: u32, dt: f32, g: f32 }
@group(1) @binding(0) var<storage, read> src: array<f32>;
@group(1) @binding(1) var<storage, read_write> dst: array<f32>;
@group(1) @binding(2) var<uniform> p: P;

fn prim(i: i32, j: i32) -> vec4<f32> {
    let ni = i32(p.ni);
    let nj = i32(p.nj);
    let jj = clamp(j, 0, nj - 1);
    var ii = i;
    var sgn = 1.0;
    if (i < 0) { ii = 0; sgn = -1.0; }
    if (i >= ni) { ii = ni - 1; }
    let n = p.ni * p.nj;
    let idx = u32(ii) + p.ni * u32(jj);
    let rho = src[idx];
    let u = src[n + idx] / rho;
    let v = src[2u * n + idx] / rho;
    let e = src[3u * n + idx];
    return vec4<f32>(rho, sgn * u, v, (p.g - 1.0) * (e - 0.5 * rho * (u * u + v * v)));
}

fn to_cons(w: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(w.x, w.x * w.y, w.x * w.z, w.w / (p.g - 1.0) + 0.5 * w.x * (w.y * w.y + w.z * w.z));
}

fn flux(w: vec4<f32>, dir: u32) -> vec4<f32> {
    let q = to_cons(w);
    if (dir == 0u) { return vec4<f32>(q.y, q.y * w.y + w.w, q.z * w.y, (q.w + w.w) * w.y); }
    return vec4<f32>(q.z, q.y * w.z, q.z * w.z + w.w, (q.w + w.w) * w.z);
}

fn speed(w: vec4<f32>, dir: u32) -> f32 {
    return abs(select(w.z, w.y, dir == 0u)) + sqrt(p.g * w.w / w.x);
}

fn rus(l: vec4<f32>, r: vec4<f32>, dir: u32) -> vec4<f32> {
    let s = max(speed(l, dir), speed(r, dir));
    return 0.5 * (flux(l, dir) + flux(r, dir)) - 0.5 * s * (to_cons(r) - to_cons(l));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.ni || gid.y >= p.nj) { return; }
    let i = i32(gid.x);
    let j = i32(gid.y);
    let wc = prim(i, j);
    let wl = prim(i - 1, j);
    let wr = prim(i + 1, j);
    let wd = prim(i, j - 1);
    let wu = prim(i, j + 1);
    let res = -(rus(wc, wr, 0u) - rus(wl, wc, 0u)) - (rus(wc, wu, 1u) - rus(wd, wc, 1u));
    let n = p.ni * p.nj;
    let idx = gid.x + p.ni * gid.y;
    dst[idx] = src[idx] + p.dt * res.x;
    dst[n + idx] = src[n + idx] + p.dt * res.y;
    dst[2u * n + idx] = src[2u * n + idx] + p.dt * res.z;
    dst[3u * n + idx] = src[3u * n + idx] + p.dt * res.w;
}
"#;

/// Runs `steps` of the hand-written kernel from `init` (4 * n floats), ping-ponging `bufs`;
/// returns the index of the buffer holding the result.
fn custom_run(kernel: &Kernel, init: &A, bufs: &mut [A; 2], ni: usize, nj: usize, steps: usize) -> usize {
    let params = EulerParams { ni: ni as u32, nj: nj as u32, dt: DT, g: GAMMA };
    for k in 0..steps {
        let (l, r) = bufs.split_at_mut(1);
        let (src, dst): (&A, &mut A) = match (k, k % 2) {
            (0, _) => (init, &mut l[0]),
            (_, 1) => (&l[0], &mut r[0]),
            _ => (&r[0], &mut l[0]),
        };
        kernel.launch([ni as u32, nj as u32], args![src, dst, uniform(&params)]).unwrap();
    }
    (steps - 1) % 2
}

fn euler(b: &mut Bench) {
    let (sizes, steps): (Vec<usize>, usize) = if b.quick { (vec![128], 10) } else { (vec![512, 2048], 50) };
    let (vn, vm) = (64, 48);
    let base = Opts { wall: true, cache: false, muscl: false };
    let host_small = init_host(vn, vm);
    let want_small = host_run(&host_small, vn, vm, steps, base);

    for v in ["per_step", "per_step_cache_w", "iterate_all", "until_all"] {
        let name = format!("euler/{v}");
        let o = Opts { cache: v == "per_step_cache_w", ..base };
        b.guarded(&name, |b| {
            let ctx = b.ctx.clone();
            let q = upload(&ctx, &host_small, vn, vm);
            let got = host_of(&euler_variant(v, &q, steps, o));
            for k in 0..4 {
                check(&name, &got[k], &want_small[k], 1e-3, 1e-3);
            }
            for &n in &sizes {
                let h = init_host(n, n);
                let q = upload(&ctx, &h, n, n);
                let mut f = || {
                    black_box(euler_variant(v, &q, steps, o));
                };
                b.time(&name, &format!("{n}x{n}"), Tp::Mcell((n * n * steps) as f64), true, &mut f);
            }
        });
    }

    b.guarded("euler/custom_kernel", |b| {
        let ctx = b.ctx.clone();
        let kernel = ctx.kernel(EULER_WGSL).entry("main").label("euler_step").build().unwrap();
        // Against variant 1 at the small size.
        {
            let q = upload(&ctx, &host_small, vn, vm);
            let want = host_of(&euler_variant("per_step", &q, steps, base));
            let packed: Vec<f32> = host_small.iter().flatten().copied().collect();
            let init = Array::from_slice_on(&ctx, &packed, 4 * vn * vm).unwrap();
            let mut bufs = [Array::<f32>::zeros_on(&ctx, 4 * vn * vm).unwrap(), Array::<f32>::zeros_on(&ctx, 4 * vn * vm).unwrap()];
            let r = custom_run(&kernel, &init, &mut bufs, vn, vm, steps);
            let got = bufs[r].to_vec().unwrap();
            for k in 0..4 {
                check("euler/custom_kernel", &got[k * vn * vm..(k + 1) * vn * vm], &want[k], 1e-4, 1e-5);
            }
        }
        for &n in &sizes {
            let h = init_host(n, n);
            let packed: Vec<f32> = h.iter().flatten().copied().collect();
            let init = Array::from_slice_on(&ctx, &packed, 4 * n * n).unwrap();
            let mut bufs = [Array::<f32>::zeros_on(&ctx, 4 * n * n).unwrap(), Array::<f32>::zeros_on(&ctx, 4 * n * n).unwrap()];
            let mut f = || {
                custom_run(&kernel, &init, &mut bufs, n, n, steps);
            };
            b.time("euler/custom_kernel", &format!("{n}x{n}"), Tp::Mcell((n * n * steps) as f64), true, &mut f);
        }
    });
}

fn muscl(b: &mut Bench) {
    let (sizes, steps): (Vec<usize>, usize) = if b.quick { (vec![128], 5) } else { (vec![512, 2048], 20) };
    let (vn, vm, vsteps) = (64, 48, 3);
    let o = Opts { wall: true, cache: false, muscl: true };
    let host_small = init_host(vn, vm);
    let want = host_run(&host_small, vn, vm, vsteps, o);
    for v in ["per_step", "iterate_all"] {
        let name = format!("muscl/{v}");
        b.guarded(&name, |b| {
            let ctx = b.ctx.clone();
            let q = upload(&ctx, &host_small, vn, vm);
            let got = host_of(&euler_variant(v, &q, vsteps, o));
            for k in 0..4 {
                check(&name, &got[k], &want[k], 2e-3, 2e-3);
            }
            for &n in &sizes {
                let h = init_host(n, n);
                let q = upload(&ctx, &h, n, n);
                let mut f = || {
                    black_box(euler_variant(v, &q, steps, o));
                };
                b.time(&name, &format!("{n}x{n}"), Tp::Mcell((n * n * steps) as f64), true, &mut f);
            }
        });
    }
}

// =================================================================================================
// C. Views
// =================================================================================================

fn views(b: &mut Bench) {
    let (n1, big2, small1, small2): (usize, [usize; 2], usize, [usize; 2]) =
        if b.quick { (1 << 20, [1024, 1024], 4096, [64, 48]) } else { (1 << 26, [8192, 8192], 4096, [64, 48]) };
    let one = [n1, 1];
    let s1 = [small1, 1];
    let t = (1e-4, 1e-4);

    b.arr(Arr {
        name: "view/diff1d", big: one, small: s1, need_b: false, need_idx: false, touched: 2.0, prof: true, tol: t,
        build: &|d| d.a.slice(0, 1..) - d.a.slice(0, ..d.w - 1),
        host: &|d| (0..d.w - 1).map(|i| d.ha[i + 1] - d.ha[i]).collect(),
    });
    b.arr(Arr {
        name: "view/diff2d_axis1", big: big2, small: small2, need_b: false, need_idx: false, touched: 2.0, prof: true, tol: t,
        build: &|d| d.a.slice(1, 1..) - d.a.slice(1, ..d.h - 1),
        host: &|d| (0..d.w * (d.h - 1)).map(|i| d.ha[i + d.w] - d.ha[i]).collect(),
    });
    b.arr(Arr {
        name: "view/pad_edge", big: big2, small: small2, need_b: false, need_idx: false, touched: 2.0, prof: true, tol: (0.0, 0.0),
        build: &|d| d.a.pad([(1, 1), (1, 1)], PadMode::Edge) * 1.0,
        host: &|d| {
            let (w, h) = (d.w, d.h);
            let mut out = Vec::with_capacity((w + 2) * (h + 2));
            for y in 0..h + 2 {
                for x in 0..w + 2 {
                    let (sx, sy) = ((x as isize - 1).clamp(0, w as isize - 1) as usize, (y as isize - 1).clamp(0, h as isize - 1) as usize);
                    out.push(d.ha[sx + w * sy]);
                }
            }
            out
        },
    });
    b.arr(Arr {
        name: "view/flip0", big: big2, small: small2, need_b: false, need_idx: false, touched: 2.0, prof: true, tol: (0.0, 0.0),
        build: &|d| d.a.flip(0) * 1.0,
        host: &|d| (0..d.w * d.h).map(|i| d.ha[(d.w - 1 - i % d.w) + d.w * (i / d.w)]).collect(),
    });
    b.arr(Arr {
        name: "view/transpose", big: big2, small: small2, need_b: false, need_idx: false, touched: 2.0, prof: true, tol: (0.0, 0.0),
        build: &|d| d.a.transpose() * 1.0,
        host: &|d| transpose_host(&d.ha, d.w, d.h),
    });
    b.arr(Arr {
        name: "view/take_random", big: big2, small: small2, need_b: false, need_idx: true, touched: 3.0, prof: true, tol: (0.0, 0.0),
        build: &|d| d.a.take(&d.idx),
        host: &|d| d.hidx.iter().map(|&i| d.ha[i as usize]).collect(),
    });
    b.arr(Arr {
        name: "view/concat2", big: big2, small: small2, need_b: true, need_idx: false, touched: 2.0 * 2.0, prof: true, tol: (0.0, 0.0),
        build: &|d| concat(0, [&d.a, &d.b]),
        host: &|d| {
            let (w, h) = (d.w, d.h);
            let mut out = Vec::with_capacity(2 * w * h);
            for y in 0..h {
                out.extend_from_slice(&d.ha[w * y..w * (y + 1)]);
                out.extend_from_slice(&d.hb[w * y..w * (y + 1)]);
            }
            out
        },
    });
    b.arr(Arr {
        name: "view/laplace_slices", big: big2, small: small2, need_b: false, need_idx: false, touched: 2.0, prof: true, tol: t,
        build: &|d| {
            let (w, h) = (d.w, d.h);
            let a = &d.a;
            let c = a.slice(0, 1..w - 1).slice(1, 1..h - 1);
            let l = a.slice(0, ..w - 2).slice(1, 1..h - 1);
            let r = a.slice(0, 2..).slice(1, 1..h - 1);
            let dn = a.slice(0, 1..w - 1).slice(1, ..h - 2);
            let up = a.slice(0, 1..w - 1).slice(1, 2..);
            l + r + dn + up - 4.0 * c
        },
        host: &|d| {
            let (w, h) = (d.w, d.h);
            let a = &d.ha;
            let mut out = Vec::new();
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    out.push(a[x - 1 + w * y] + a[x + 1 + w * y] + a[x + w * (y - 1)] + a[x + w * (y + 1)] - 4.0 * a[x + w * y]);
                }
            }
            out
        },
    });
    b.arr(Arr {
        name: "view/laplace_shift", big: big2, small: small2, need_b: false, need_idx: false, touched: 2.0, prof: true, tol: t,
        build: &|d| {
            let a = &d.a;
            a.shift([1isize, 0]) + a.shift([-1isize, 0]) + a.shift([0isize, 1]) + a.shift([0isize, -1]) - 4.0 * a
        },
        host: &|d| {
            let (w, h) = (d.w, d.h);
            let a = &d.ha;
            let mut out = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    out.push(
                        a[(x + w - 1) % w + w * y] + a[(x + 1) % w + w * y] + a[x + w * ((y + h - 1) % h)] + a[x + w * ((y + 1) % h)]
                            - 4.0 * a[x + w * y],
                    );
                }
            }
            out
        },
    });
    b.arr(Arr {
        name: "view/slice_of_expr", big: one, small: s1, need_b: false, need_idx: false, touched: 2.0, prof: true, tol: (1e-4, 1e-5),
        build: &|d| {
            let e = (&d.a * 2.0 + 1.0).sqrt();
            e.slice(0, 1..) - e.slice(0, ..d.w - 1)
        },
        host: &|d| {
            let e: Vec<f64> = d.ha.iter().map(|&v| (f64::from(v) * 2.0 + 1.0).sqrt()).collect();
            (0..d.w - 1).map(|i| (e[i + 1] - e[i]) as f32).collect()
        },
    });
    b.arr(Arr {
        name: "view/slice_of_expr3", big: one, small: s1, need_b: false, need_idx: false, touched: 2.0, prof: true, tol: (1e-4, 1e-5),
        build: &|d| {
            let e = (&d.a * 2.0 + 1.0).sqrt();
            e.slice(0, 2..) + e.slice(0, 1..d.w - 1) - 2.0 * e.slice(0, ..d.w - 2)
        },
        host: &|d| {
            let e: Vec<f64> = d.ha.iter().map(|&v| (f64::from(v) * 2.0 + 1.0).sqrt()).collect();
            (0..d.w - 2).map(|i| (e[i + 2] + e[i + 1] - 2.0 * e[i]) as f32).collect()
        },
    });
}

/// Transpose of a `w x h` array (x fastest): the result is `h x w`.
fn transpose_host(a: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(w * h);
    for y in 0..w {
        for x in 0..h {
            out.push(a[y + w * x]);
        }
    }
    out
}

// =================================================================================================
// D. Writes
// =================================================================================================

fn writes(b: &mut Bench) {
    let (big, small, put_n, rec): (usize, [usize; 2], usize, usize) =
        if b.quick { (1024, [64, 48], 1 << 18, 512) } else { (8192, [64, 48], 1 << 24, 4096) };
    let reps = 100.0;
    let (sw, sh) = (small[0], small[1]);

    b.guarded("assign/ghost_scalar", |b| {
        let ctx = b.ctx.clone();
        {
            let ha = values(sw * sh, 1, 0.2, 1.2);
            let w = Array::from_slice_on(&ctx, &ha, [sw, sh]).unwrap();
            w.slice(0, ..1).assign(0.0).unwrap();
            let mut want = ha.clone();
            for y in 0..sh {
                want[sw * y] = 0.0;
            }
            check("assign/ghost_scalar", &w.to_vec().unwrap(), &want, 0.0, 0.0);
        }
        let w = Array::<f32>::full_on(&ctx, [big, big], 1.0).unwrap();
        w.eval().unwrap();
        let mut f = || {
            for _ in 0..100 {
                w.slice(0, ..1).assign(0.0).unwrap();
            }
        };
        b.time("assign/ghost_scalar", &format!("{big}x{big}"), Tp::PerOp(reps, "us/assign"), false, &mut f);
    });

    b.guarded("assign/ghost_mirror", |b| {
        let ctx = b.ctx.clone();
        {
            let ha = values(sw * sh, 2, 0.2, 1.2);
            let w = Array::from_slice_on(&ctx, &ha, [sw, sh]).unwrap();
            w.slice(0, ..2).assign(w.slice(0, 2..4).flip(0) * -1.0).unwrap();
            let mut want = ha.clone();
            for y in 0..sh {
                want[sw * y] = -ha[3 + sw * y];
                want[1 + sw * y] = -ha[2 + sw * y];
            }
            check("assign/ghost_mirror", &w.to_vec().unwrap(), &want, 0.0, 0.0);
        }
        let w = Array::<f32>::full_on(&ctx, [big, big], 1.0).unwrap();
        w.eval().unwrap();
        let mut f = || {
            for _ in 0..100 {
                w.slice(0, ..2).assign(w.slice(0, 2..4).flip(0) * -1.0).unwrap();
            }
        };
        b.time("assign/ghost_mirror", &format!("{big}x{big}"), Tp::PerOp(reps, "us/assign"), false, &mut f);
    });

    b.guarded("assign/full", |b| {
        let ctx = b.ctx.clone();
        {
            let ha = values(4096, 3, 0.2, 1.2);
            let a = Array::from_slice_on(&ctx, &ha, 4096).unwrap();
            let w = Array::<f32>::zeros_on(&ctx, 4096).unwrap();
            w.assign(&a * 2.0).unwrap();
            let want: Vec<f32> = ha.iter().map(|v| v * 2.0).collect();
            check("assign/full", &w.to_vec().unwrap(), &want, 0.0, 0.0);
        }
        let n = if b.quick { 1 << 20 } else { 1 << 26 };
        let a = Array::<f32>::full_on(&ctx, n, 1.5).unwrap();
        let w = Array::<f32>::zeros_on(&ctx, n).unwrap();
        a.eval().unwrap();
        let mut f = || w.assign(&a * 2.0).unwrap();
        b.time("assign/full", &format!("{n}"), Tp::Bytes(8.0 * n as f64), false, &mut f);
    });

    b.guarded("assign/record4", |b| {
        let ctx = b.ctx.clone();
        let fields = |ctx: &Context, w: usize, h: usize, seed: u64| -> (Cons, [Vec<f32>; 4]) {
            let host: [Vec<f32>; 4] = std::array::from_fn(|k| values(w * h, seed + k as u64, 0.2, 1.2));
            (upload(ctx, &host, w, h), host)
        };
        {
            let (q, host) = fields(&ctx, sw, sh, 10);
            let ghost = q.slice(0, 2..4).flip(0) * -1.0;
            q.slice(0, ..2).assign(ghost).unwrap();
            let got = host_of(&q);
            for k in 0..4 {
                let mut want = host[k].clone();
                for y in 0..sh {
                    want[sw * y] = -host[k][3 + sw * y];
                    want[1 + sw * y] = -host[k][2 + sw * y];
                }
                check("assign/record4", &got[k], &want, 0.0, 0.0);
            }
        }
        let host: [Vec<f32>; 4] = std::array::from_fn(|_| vec![1.0; rec * rec]);
        let q = upload(&ctx, &host, rec, rec);
        let mut f = || {
            for _ in 0..100 {
                let ghost = q.slice(0, 2..4).flip(0) * -1.0;
                q.slice(0, ..2).assign(ghost).unwrap();
            }
        };
        b.time("assign/record4", &format!("4x{rec}x{rec}"), Tp::PerOp(reps, "us/assign"), false, &mut f);
    });

    b.guarded("put/scatter", |b| {
        let ctx = b.ctx.clone();
        {
            let n = 4096;
            let idx = permutation(n, 5);
            let vals = values(n, 6, 0.2, 1.2);
            let x = Array::<f32>::zeros_on(&ctx, n).unwrap();
            x.put(&Array::from_slice_on(&ctx, &idx, n).unwrap(), &Array::from_slice_on(&ctx, &vals, n).unwrap()).unwrap();
            let mut want = vec![0.0f32; n];
            for i in 0..n {
                want[idx[i] as usize] = vals[i];
            }
            check("put/scatter", &x.to_vec().unwrap(), &want, 0.0, 0.0);
        }
        let n = put_n;
        let idx = Array::from_slice_on(&ctx, &permutation(n, 7), n).unwrap();
        let vals = Array::from_slice_on(&ctx, &values(n, 8, 0.2, 1.2), n).unwrap();
        let x = Array::<f32>::zeros_on(&ctx, n).unwrap();
        let mut f = || x.put(&idx, &vals).unwrap();
        b.time("put/scatter", &format!("{n}"), Tp::Bytes(12.0 * n as f64), false, &mut f);
    });
}

// =================================================================================================
// E. Scalars
// =================================================================================================

impl Bench {
    #[allow(clippy::too_many_arguments)]
    fn scalar(
        &mut self,
        name: &str,
        big: [usize; 2],
        small: [usize; 2],
        need_b: bool,
        touched: f64,
        rel: f64,
        build: &dyn Fn(&Dat) -> f64,
        host: &dyn Fn(&Dat) -> f64,
    ) {
        self.guarded(name, |b| {
            let ctx = b.ctx.clone();
            {
                let d = Dat::new(&ctx, small[0], small[1], need_b, false);
                check_scalar(name, build(&d), host(&d), rel);
            }
            let d = Dat::new(&ctx, big[0], big[1], need_b, false);
            let mut f = || {
                black_box(build(&d));
            };
            b.time(name, &size_str(d.w, d.h), Tp::Bytes(touched * (d.w * d.h) as f64 * 4.0), false, &mut f);
        });
    }
}

fn scalars(b: &mut Bench) {
    let (n, big2, small): (usize, [usize; 2], [usize; 2]) =
        if b.quick { (1 << 20, [1024, 1024], [1 << 16, 1]) } else { (1 << 26, [8192, 8192], [1 << 16, 1]) };
    let one = [n, 1];
    let hsum = |d: &Dat| d.ha.iter().map(|&v| f64::from(v)).sum::<f64>();

    b.scalar("reduce/sum", one, small, false, 1.0, 1e-4, &|d| f64::from(d.a.sum().item().unwrap()), &hsum);
    b.scalar(
        "reduce/sum_of_expr", one, small, true, 2.0, 1e-4,
        &|d| f64::from((&d.a * &d.b + 1.0).sum().item().unwrap()),
        &|d| d.ha.iter().zip(&d.hb).map(|(&a, &b)| f64::from(a) * f64::from(b) + 1.0).sum(),
    );
    b.arr(Arr {
        name: "reduce/normalize", big: one, small, need_b: false, need_idx: false, touched: 2.0, prof: false, tol: (1e-4, 1e-8),
        build: &|d| &d.a / &d.a.sum(),
        host: &|d| {
            let s = d.ha.iter().map(|&v| f64::from(v)).sum::<f64>();
            d.ha.iter().map(|&v| (f64::from(v) / s) as f32).collect()
        },
    });

    // 200 iterations of x *= 0.999; max().item() with a readback each step.
    b.guarded("reduce/item_loop", |b| {
        let ctx = b.ctx.clone();
        let (side, iters) = if b.quick { (256, 50) } else { (1024, 200) };
        let run = |a: &A, iters: usize| -> f32 {
            let mut x = a * 1.0;
            let mut r = 0.0;
            for _ in 0..iters {
                x = &x * 0.999;
                r = x.max().item().unwrap();
            }
            r
        };
        {
            let d = Dat::new(&ctx, 64, 64, false, false);
            let max0 = d.ha.iter().copied().fold(f32::MIN, f32::max) as f64;
            check_scalar("reduce/item_loop", f64::from(run(&d.a, iters)), max0 * 0.999f64.powi(iters as i32), 1e-3);
        }
        let d = Dat::new(&ctx, side, side, false, false);
        let mut f = || {
            black_box(run(&d.a, iters));
        };
        // `PerOp` divides by 1e-3 of its count to print ms: n = iterations * 1000.
        b.time("reduce/item_loop", &format!("{side}x{side}"), Tp::PerOp(iters as f64 * 1000.0, "ms/iter"), false, &mut f);
    });

    let axis_host = |d: &Dat, axis: usize| -> Vec<f32> {
        let (w, h) = (d.w, d.h);
        if axis == 0 {
            (0..h).map(|y| d.ha[w * y..w * (y + 1)].iter().map(|&v| f64::from(v)).sum::<f64>() as f32).collect()
        } else {
            (0..w).map(|x| (0..h).map(|y| f64::from(d.ha[x + w * y])).sum::<f64>() as f32).collect()
        }
    };
    b.arr(Arr {
        name: "reduce/sum_axis0", big: big2, small: [64, 48], need_b: false, need_idx: false, touched: 1.0, prof: false, tol: (1e-4, 1e-4),
        build: &|d| d.a.sum_axis(0),
        host: &|d| axis_host(d, 0),
    });
    b.arr(Arr {
        name: "reduce/sum_axis1", big: big2, small: [64, 48], need_b: false, need_idx: false, touched: 1.0, prof: false, tol: (1e-4, 1e-4),
        build: &|d| d.a.sum_axis(1),
        host: &|d| axis_host(d, 1),
    });

    b.scalar("reduce/sum_compensated", one, small, false, 1.0, 1e-6, &|d| f64::from(d.a.sum_compensated().item().unwrap()), &hsum);
    b.scalar("reduce/df32_sum", one, small, false, 1.0, 1e-8, &|d| d.a.to_df32().sum().item().unwrap().to_f64(), &hsum);
    if b.ctx.features().contains(wgpu::Features::SHADER_F64) {
        b.scalar("reduce/f64_sum", one, small, false, 1.0, 1e-9, &|d| d.a.to_f64().sum().item().unwrap(), &hsum);
    } else {
        b.skip("reduce/f64_sum", "the device has no SHADER_F64");
    }
}

// =================================================================================================
// F. Line primitives
// =================================================================================================

fn lines(b: &mut Bench) {
    let (n1, big2, tri2, blk, v1): (usize, [usize; 2], usize, usize, usize) =
        if b.quick { (1 << 20, [1024, 1024], 512, 128, 1 << 16) } else { (1 << 26, [8192, 8192], 4096, 1024, 1 << 16) };
    let tol = (2e-3, 2e-3);

    // Host: running fold along `axis` of a `w x h` array.
    let scan_host = |d: &Dat, axis: usize, max: bool| -> Vec<f32> {
        let (w, h) = (d.w, d.h);
        let mut out = vec![0.0f32; w * h];
        let mut acc = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = x + w * y;
                let prev = match axis {
                    0 if x > 0 => Some(acc[i - 1]),
                    1 if y > 0 => Some(acc[i - w]),
                    _ => None,
                };
                let v = f64::from(d.ha[i]);
                acc[i] = match prev {
                    None => v,
                    Some(p) if max => p.max(v),
                    Some(p) => p + v,
                };
                out[i] = acc[i] as f32;
            }
        }
        out
    };

    b.arr(Arr {
        name: "scan/cumsum_1d", big: [n1, 1], small: [v1, 1], need_b: false, need_idx: false, touched: 2.0, prof: true, tol,
        build: &|d| d.a.cumsum(0),
        host: &|d| scan_host(d, 0, false),
    });
    b.arr(Arr {
        name: "scan/cumsum_axis0", big: big2, small: [1500, 9], need_b: false, need_idx: false, touched: 2.0, prof: true, tol,
        build: &|d| d.a.cumsum(0),
        host: &|d| scan_host(d, 0, false),
    });
    b.arr(Arr {
        name: "scan/cumsum_axis1", big: big2, small: [9, 1500], need_b: false, need_idx: false, touched: 2.0, prof: true, tol,
        build: &|d| d.a.cumsum(1),
        host: &|d| scan_host(d, 1, false),
    });
    b.arr(Arr {
        name: "scan/running_max_axis0", big: big2, small: [1500, 9], need_b: false, need_idx: false, touched: 2.0, prof: true, tol: (0.0, 0.0),
        build: &|d| d.a.scan(0, |x, y| x.maximum(y)),
        host: &|d| scan_host(d, 0, true),
    });

    for axis in 0..2usize {
        let name = format!("tri/solve_axis{axis}");
        b.guarded(&name, |b| {
            let ctx = b.ctx.clone();
            let mk = |w: usize, h: usize| -> ([Vec<f32>; 4], [A; 4]) {
                let n = w * h;
                let host = [
                    values(n, 21, -1.0, 1.0),
                    values(n, 22, -1.0, 1.0).iter().map(|v| 4.0 + v).collect(),
                    values(n, 23, -1.0, 1.0),
                    values(n, 24, -1.0, 1.0),
                ];
                let arr = std::array::from_fn(|k| Array::from_slice_on(&ctx, &host[k], [w, h]).unwrap());
                (host, arr)
            };
            {
                let (host, [lo, diag, up, rhs]) = mk(64, 48);
                let got = solve_tridiagonal(axis, &lo, &diag, &up, &rhs).to_vec().unwrap();
                let want = host_tri(1, [64, 48, 1], axis, &host[0], &host[1], &host[2], &host[3]);
                check(&name, &got, &want, 1e-4, 1e-4);
            }
            let (_h, [lo, diag, up, rhs]) = mk(tri2, tri2);
            let mut f = || solve_tridiagonal(axis, &lo, &diag, &up, &rhs).eval().unwrap();
            b.time(&name, &format!("{tri2}x{tri2}"), Tp::Bytes(5.0 * 4.0 * (tri2 * tri2) as f64), true, &mut f);
        });
    }

    b.guarded("tri/block4_axis1", |b| {
        let ctx = b.ctx.clone();
        let mk = |g0: usize, g1: usize| -> ([Vec<f32>; 4], [A; 4]) {
            let cells = g0 * g1;
            let mut diag = values(cells * 16, 32, -1.0, 1.0);
            for (k, v) in diag.iter_mut().enumerate() {
                if (k % 16) / 4 == k % 4 {
                    *v += 8.0;
                }
            }
            let host = [values(cells * 16, 31, -1.0, 1.0), diag, values(cells * 16, 33, -1.0, 1.0), values(cells * 4, 34, -1.0, 1.0)];
            let arr = std::array::from_fn(|k| {
                let dims = if k == 3 { [4, g0, g1] } else { [16, g0, g1] };
                Array::from_slice_on(&ctx, &host[k], dims).unwrap()
            });
            (host, arr)
        };
        {
            let (host, [lo, diag, up, rhs]) = mk(30, 5);
            let got = solve_block_tridiagonal(4, 1, &lo, &diag, &up, &rhs).to_vec().unwrap();
            let want = host_tri(4, [30, 5, 1], 0, &host[0], &host[1], &host[2], &host[3]);
            check("tri/block4_axis1", &got, &want, 1e-3, 1e-3);
        }
        let (_h, [lo, diag, up, rhs]) = mk(blk, blk);
        let mut f = || solve_block_tridiagonal(4, 1, &lo, &diag, &up, &rhs).eval().unwrap();
        b.time("tri/block4_axis1", &format!("4x{blk}x{blk}"), Tp::Bytes((3.0 * 16.0 + 2.0 * 4.0) * 4.0 * (blk * blk) as f64), true, &mut f);
    });
}

/// Dense solve of `m x = r` (`n x n`, row-major) with partial pivoting in f64.
fn dense_solve(mut m: Vec<f64>, mut r: Vec<f64>, n: usize) -> Vec<f64> {
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| m[i * n + col].abs().total_cmp(&m[j * n + col].abs())).unwrap();
        for j in 0..n {
            m.swap(col * n + j, pivot * n + j);
        }
        r.swap(col, pivot);
        for i in col + 1..n {
            let f = m[i * n + col] / m[col * n + col];
            for j in col..n {
                m[i * n + j] -= f * m[col * n + j];
            }
            r[i] -= f * r[col];
        }
    }
    for i in (0..n).rev() {
        let mut x = r[i];
        for j in i + 1..n {
            x -= m[i * n + j] * r[j];
        }
        r[i] = x / m[i * n + i];
    }
    r
}

/// Host block-tridiagonal solve of every line of `grid` along `axis`; blocks of `b` (coefficients
/// `b * b` per cell, `rhs` `b`).
#[allow(clippy::too_many_arguments)]
fn host_tri(b: usize, grid: [usize; 3], axis: usize, lo: &[f32], diag: &[f32], up: &[f32], rhs: &[f32]) -> Vec<f32> {
    let cells: usize = grid.iter().product();
    let n = grid[axis];
    let stride: usize = grid[..axis].iter().product();
    let mut x = vec![0.0f32; cells * b];
    for l in 0..cells / n {
        let base = (l % stride) + (l / stride) * stride * n;
        let size = n * b;
        let mut matrix = vec![0.0f64; size * size];
        let mut r = vec![0.0f64; size];
        for s in 0..n {
            let q = base + s * stride;
            for i in 0..b {
                r[s * b + i] = f64::from(rhs[q * b + i]);
                for j in 0..b {
                    matrix[(s * b + i) * size + s * b + j] = f64::from(diag[q * b * b + i * b + j]);
                    if s > 0 {
                        matrix[(s * b + i) * size + (s - 1) * b + j] = f64::from(lo[q * b * b + i * b + j]);
                    }
                    if s + 1 < n {
                        matrix[(s * b + i) * size + (s + 1) * b + j] = f64::from(up[q * b * b + i * b + j]);
                    }
                }
            }
        }
        let sol = dense_solve(matrix, r, size);
        for s in 0..n {
            let q = base + s * stride;
            for i in 0..b {
                x[q * b + i] = sol[s * b + i] as f32;
            }
        }
    }
    x
}

// =================================================================================================
// G. Stepping machinery on tiny arrays
// =================================================================================================

fn tiny_step(x: &A, y: &A) -> [A; 2] {
    let c = (x * 0.9 + y.shift(1) * 0.1).cache();
    [c.shift(-1) * 0.5 + &c * 0.5, y * 0.95 + &c * 0.05]
}

fn tiny_host_loop(x0: &A, y0: &A, steps: usize) -> [A; 2] {
    let [mut x, mut y] = tiny_step(x0, y0);
    eval_all(&[&x, &y]).unwrap();
    for _ in 1..steps {
        let [nx, ny] = tiny_step(&x, &y);
        eval_all(&[&nx, &ny]).unwrap();
        x = nx;
        y = ny;
    }
    [x, y]
}

fn tiny_until(x0: &A, y0: &A, steps: usize, every: Option<usize>) -> [A; 2] {
    let until = Until::new(steps).record("m", |s: &(A, A)| s.0.max()).stop_when(|h| h["m"].last().lt(0.0));
    let until = match every {
        Some(n) => until.check_every(n),
        None => until,
    };
    let run = iterate_until_all(
        (x0 * 1.0, y0 * 1.0),
        until,
        |s, _| {
            let [a, b] = tiny_step(&s.0, &s.1);
            (a, b)
        },
    )
    .unwrap();
    assert_eq!(run.steps, steps, "iterate_until_all stopped early");
    let (a, b) = run.state;
    eval_all(&[&a, &b]).unwrap();
    [a, b]
}

fn tiny(b: &mut Bench) {
    let steps = if b.quick { 200 } else { 2000 };
    let builds = if b.quick { 50 } else { 200 };
    let size = "64x64";
    let mk = |ctx: &Context| {
        let x = Array::from_slice_on(ctx, &values(64 * 64, 41, 0.5, 1.5), [64, 64]).unwrap();
        let y = Array::from_slice_on(ctx, &values(64 * 64, 42, 0.5, 1.5), [64, 64]).unwrap();
        x.eval().unwrap();
        y.eval().unwrap();
        (x, y)
    };
    let verify_steps = 37;

    b.guarded("tiny/iterate_all_replay", |b| {
        let ctx = b.ctx.clone();
        let (x, y) = mk(&ctx);
        let want = tiny_host_loop(&x, &y, verify_steps);
        let got = iterate_all([&x, &y], verify_steps, |s| tiny_step(&s[0], &s[1]));
        for k in 0..2 {
            check("tiny/iterate_all_replay", &got[k].to_vec().unwrap(), &want[k].to_vec().unwrap(), 1e-4, 1e-5);
        }
        let mut f = || {
            let r = iterate_all([&x, &y], steps, |s| tiny_step(&s[0], &s[1]));
            eval_all(&[&r[0], &r[1]]).unwrap();
        };
        b.time("tiny/iterate_all_replay", size, Tp::PerOp(steps as f64, "us/step"), false, &mut f);
    });

    for every in [Some(16usize), Some(64), None] {
        let name = match every {
            Some(n) => format!("tiny/until_all_every{n}"),
            None => "tiny/until_all_default".to_string(),
        };
        b.guarded(&name, |b| {
            let ctx = b.ctx.clone();
            let (x, y) = mk(&ctx);
            let want = tiny_host_loop(&x, &y, verify_steps);
            let got = tiny_until(&x, &y, verify_steps, every);
            for k in 0..2 {
                check(&name, &got[k].to_vec().unwrap(), &want[k].to_vec().unwrap(), 1e-4, 1e-5);
            }
            let mut f = || {
                black_box(tiny_until(&x, &y, steps, every));
            };
            b.time(&name, size, Tp::PerOp(steps as f64, "us/step"), false, &mut f);
        });
    }

    b.guarded("tiny/host_loop", |b| {
        let ctx = b.ctx.clone();
        let (x, y) = mk(&ctx);
        // Reference: the lazily fused loop through `iterate_all`.
        let want = iterate_all([&x, &y], verify_steps, |s| tiny_step(&s[0], &s[1]));
        let got = tiny_host_loop(&x, &y, verify_steps);
        for k in 0..2 {
            check("tiny/host_loop", &got[k].to_vec().unwrap(), &want[k].to_vec().unwrap(), 1e-4, 1e-5);
        }
        let mut f = || {
            black_box(tiny_host_loop(&x, &y, steps));
        };
        b.time("tiny/host_loop", size, Tp::PerOp(steps as f64, "us/step"), false, &mut f);
    });

    b.guarded("tiny/lazy_build", |b| {
        let ctx = b.ctx.clone();
        // No wall: the ghost assign would evaluate the padded state while building.
        let o = Opts { wall: false, cache: false, muscl: false };
        let h = init_host(64, 64);
        let q = upload(&ctx, &h, 64, 64);
        let want = host_run(&h, 64, 64, 1, o);
        let got = host_of(&step(&q, o));
        for k in 0..4 {
            check("tiny/lazy_build", &got[k], &want[k], 1e-3, 1e-3);
        }
        let mut f = || {
            for _ in 0..builds {
                drop(black_box(step(&q, o)));
            }
        };
        b.time("tiny/lazy_build", size, Tp::PerOp(builds as f64, "us/build"), false, &mut f);
    });
}

// =================================================================================================
// H. Convergence loops on a 2-D heat stencil
// =================================================================================================

fn heat(u: &A) -> A {
    u + (u.shift([1isize, 0]) + u.shift([-1isize, 0]) + u.shift([0isize, 1]) + u.shift([0isize, -1]) - u * 4.0) * 0.2
}

/// `steps` applications of `heat` on the host in f64, circular in both axes.
fn heat_host(h: &[f32], w: usize, ht: usize, steps: usize) -> Vec<f32> {
    let mut u: Vec<f64> = h.iter().map(|&v| f64::from(v)).collect();
    for _ in 0..steps {
        let mut n = u.clone();
        for y in 0..ht {
            for x in 0..w {
                let at = |x: usize, y: usize| u[x + w * y];
                let lap = at((x + 1) % w, y) + at((x + w - 1) % w, y) + at(x, (y + 1) % ht) + at(x, (y + ht - 1) % ht) - 4.0 * at(x, y);
                n[x + w * y] = at(x, y) + 0.2 * lap;
            }
        }
        u = n;
    }
    u.into_iter().map(|v| v as f32).collect()
}

fn converge_readback(u0: &A, steps: usize) -> A {
    let mut u = u0 * 1.0;
    for _ in 0..steps {
        let next = heat(&u);
        let err = (&next - &u).abs().max().item().unwrap();
        u = next;
        black_box(err);
    }
    u
}

fn converge_until(u0: &A, steps: usize) -> A {
    u0.iterate_until(steps, 1e-30, heat)
}

fn converge_until_all(u0: &A, steps: usize) -> A {
    let run = iterate_until_all(
        (u0 * 1.0,),
        Until::new(steps)
            .record("change", |s: &(A,)| (&heat(&s.0) - &s.0).abs().max())
            .stop_when(|h| h["change"].last().lt(-1.0)),
        |s, _| (heat(&s.0),),
    )
    .unwrap();
    assert_eq!(run.steps, steps, "iterate_until_all stopped early");
    run.state.0
}

fn converge(b: &mut Bench) {
    let (sizes, steps): (Vec<usize>, usize) = if b.quick { (vec![64], 50) } else { (vec![256, 1024], 200) };
    type Variant = (&'static str, fn(&A, usize) -> A);
    let variants: [Variant; 4] = [
        ("converge/readback_loop", converge_readback),
        ("converge/iterate_until", converge_until),
        ("converge/until_all", converge_until_all),
        ("converge/iterate", |u, n| u.iterate(n, heat)),
    ];
    for (name, run) in variants {
        for &n in &sizes {
            let label = format!("{name}@{n}");
            b.guarded(&label, |b| {
                let ctx = b.ctx.clone();
                let vn = 32;
                let vsteps = 20;
                let hv = values(vn * vn, 51, 0.0, 1.0);
                let small = Array::from_slice_on(&ctx, &hv, [vn, vn]).unwrap();
                check(name, &run(&small, vsteps).to_vec().unwrap(), &heat_host(&hv, vn, vn, vsteps), 1e-4, 1e-5);
                let u0 = Array::from_slice_on(&ctx, &values(n * n, 52, 0.0, 1.0), [n, n]).unwrap();
                u0.eval().unwrap();
                let mut f = || {
                    let r = run(&u0, steps);
                    black_box(r.to_vec().unwrap());
                };
                b.time(name, &format!("{n}x{n}"), Tp::Mcell((n * n * steps) as f64), true, &mut f);
            });
        }
    }
}

// =================================================================================================

fn print_profiles(b: &Bench) {
    for (name, size, p) in &b.profiles {
        println!("\n== {name} [{size}]: {} launches, {:.3} ms GPU", p.launches(), p.total().as_secs_f64() * 1e3);
        println!("   {:<100} {:>8} {:>10}", "kernel", "launches", "total ms");
        let mut kernels: Vec<_> = p.kernels.iter().collect();
        kernels.sort_by_key(|k| std::cmp::Reverse(k.total));
        for k in kernels.iter().take(12) {
            let label: String = k.label.chars().take(100).collect();
            println!("   {:<100} {:>8} {:>10.3}", label, k.launches, k.total.as_secs_f64() * 1e3);
        }
        if p.kernels.len() > 12 {
            println!("   ... {} more labels", p.kernels.len() - 12);
        }
        let mut notes: BTreeMap<String, usize> = BTreeMap::new();
        for s in &p.splits {
            *notes.entry(s.to_string()).or_default() += 1;
        }
        for (note, count) in notes {
            println!("   {note}  x{count}");
        }
    }
}

fn main() {
    let mut quick = false;
    let mut only = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--quick" => quick = true,
            "--only" => only = args.next(),
            "--bench" => {}
            other => panic!("unknown argument {other}; usage: bench_pde [--quick] [--only <substring>]"),
        }
    }
    std::panic::set_hook(Box::new(|info| {
        *PANIC.lock().unwrap() = info.to_string();
    }));
    let ctx = default_context().unwrap();
    let mut b = Bench { ctx, quick, only, profiles: Vec::new(), skipped: Vec::new() };
    println!("bench_pde ({}), profiling supported: {}", if quick { "quick" } else { "default sizes" }, b.ctx.profiling_supported());
    println!(
        "{:<28} {:>11} {:>10} {:>10} {:>10} {:<11} {:>7} {:>9}",
        "case", "size", "ms", "min ms", "rate", "unit", "kernels", "gpu ms"
    );
    euler(&mut b);
    muscl(&mut b);
    views(&mut b);
    writes(&mut b);
    scalars(&mut b);
    lines(&mut b);
    tiny(&mut b);
    converge(&mut b);
    print_profiles(&b);
    if !b.skipped.is_empty() {
        println!("\n{} case(s) SKIPPED:", b.skipped.len());
        for (n, why) in &b.skipped {
            println!("  {n}: {why}");
        }
    }
}
