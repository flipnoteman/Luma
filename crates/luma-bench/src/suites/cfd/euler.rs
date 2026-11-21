//! The array-API flow solver of the `cfd` suite, generic in the number of fields `N = D + 2`
//! (density, `D` momenta or velocities, energy or pressure): ghost cells by record assigns, a
//! first-order or MUSCL Rusanov residual, and a line-implicit step in the shape of the bowshock
//! solver (block coefficients from the state, cached, block-tridiagonal solves on parts along every
//! axis, defect correction, update). `host.rs` holds the double precision mirror of every formula.

use luma::*;

pub type A = Array<f32>;

pub const GAMMA: f32 = 1.4;
/// Explicit time step (unit cell size; stable for the suite's states in 2-D and 3-D).
pub const DT: f32 = 0.1;

/// Boundary condition of one face of the grid.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bc {
    /// Mirror about the face with the normal velocity negated.
    Wall,
    /// The adjacent interior cell (only assigned for one ghost layer; otherwise the edge pad stays).
    Extrap,
}

/// The shape of `n` (1 to 4 dimensions).
pub fn shape(n: &[usize]) -> Shape {
    match n {
        [a] => Shape::from(*a),
        [a, b] => Shape::from([*a, *b]),
        [a, b, c] => Shape::from([*a, *b, *c]),
        [a, b, c, d] => Shape::from([*a, *b, *c, *d]),
        _ => unreachable!("1 to 4 dimensions"),
    }
}

/// Fields of a record from independent host vectors.
pub fn upload<const N: usize>(ctx: &Context, h: &[Vec<f32>], n: &[usize]) -> [A; N] {
    let q: [A; N] = std::array::from_fn(|k| Array::from_slice_on(ctx, &h[k], shape(n)).unwrap());
    Fields::eval(&q).unwrap();
    q
}

pub fn download<const N: usize>(q: &[A; N]) -> Vec<Vec<f32>> {
    q.iter().map(|a| a.to_vec().unwrap()).collect()
}

/// Conserved initial state: Gaussian density and pressure bumps and smooth velocities; `amp` scales the perturbation
/// (different amplitudes are different "points" of a run), component-major.
pub fn init_host<const N: usize>(n: &[usize], amp: f64) -> Vec<Vec<f32>> {
    let d = N - 2;
    let cells: usize = n.iter().product();
    let sigma = 0.15 * n.iter().copied().max().unwrap() as f64;
    let g = f64::from(GAMMA);
    let mut out: Vec<Vec<f32>> = (0..N).map(|_| Vec::with_capacity(cells)).collect();
    for q in 0..cells {
        let mut x = [0.0f64; 3];
        let mut rest = q;
        for a in 0..d {
            x[a] = (rest % n[a]) as f64;
            rest /= n[a];
        }
        let bump = |frac: [f64; 3]| {
            let r2: f64 = (0..d).map(|a| (x[a] - frac[a] * n[a] as f64).powi(2)).sum();
            (-r2 / (2.0 * sigma * sigma)).exp()
        };
        let rho = 1.0 + 0.2 * amp * bump([0.5, 0.5, 0.5]);
        let p = 1.0 + 0.2 * amp * bump([0.3, 0.6, 0.4]);
        let v: Vec<f64> = (0..d).map(|a| 0.1 * amp * (2.0 * std::f64::consts::PI * x[a] / n[a] as f64 + a as f64).sin()).collect();
        out[0].push(rho as f32);
        for a in 0..d {
            out[1 + a].push((rho * v[a]) as f32);
        }
        out[N - 1].push((p / (g - 1.0) + 0.5 * rho * v.iter().map(|v| v * v).sum::<f64>()) as f32);
    }
    out
}

fn sum_sq<const N: usize>(w: &[A; N]) -> A {
    let mut ke = &w[1] * &w[1];
    for v in &w[2..N - 1] {
        ke = ke + v * v;
    }
    ke
}

/// Primitive variables `[rho, v.., p]` of conserved `q`.
pub fn prim<const N: usize>(q: &[A; N]) -> [A; N] {
    let rho = &q[0];
    let vel: Vec<A> = (1..N - 1).map(|k| &q[k] / rho).collect();
    let mut ke = &vel[0] * &vel[0];
    for v in &vel[1..] {
        ke = ke + v * v;
    }
    let p = (GAMMA - 1.0) * (&q[N - 1] - 0.5 * rho * ke);
    std::array::from_fn(|k| if k == 0 { rho * 1.0 } else if k < N - 1 { vel[k - 1].clone() } else { p.clone() })
}

/// Conserved variables of primitive `w`.
pub fn cons<const N: usize>(w: &[A; N]) -> [A; N] {
    let e = &w[N - 1] / (GAMMA - 1.0) + 0.5 * &w[0] * sum_sq(w);
    std::array::from_fn(|k| if k == 0 { &w[0] * 1.0 } else if k < N - 1 { &w[0] * &w[k] } else { e.clone() })
}

fn flux<const N: usize>(w: &[A; N], dir: usize) -> [A; N] {
    let q = cons(w);
    let un = &w[1 + dir];
    std::array::from_fn(|k| {
        if k == 0 {
            &q[1 + dir] * 1.0
        } else if k < N - 1 {
            let f = &q[k] * un;
            if k == 1 + dir { f + &w[N - 1] } else { f }
        } else {
            (&q[N - 1] + &w[N - 1]) * un
        }
    })
}

fn wave_speed<const N: usize>(w: &[A; N], dir: usize) -> A {
    w[1 + dir].abs() + (GAMMA * &w[N - 1] / &w[0]).sqrt()
}

fn rusanov<const N: usize>(l: &[A; N], r: &[A; N], dir: usize) -> [A; N] {
    let smax = wave_speed(l, dir).maximum(wave_speed(r, dir));
    let (ul, ur) = (cons(l), cons(r));
    let (fl, fr) = (flux(l, dir), flux(r, dir));
    std::array::from_fn(|k| 0.5 * (&fl[k] + &fr[k]) - 0.5 * &smax * (&ur[k] - &ul[k]))
}

fn minmod(a: &A, b: &A) -> A {
    let smaller = select(&a.abs().lt(b.abs()), a, b);
    select(&(a * b).gt(0.0), &smaller, 0.0)
}

fn pad_all<const N: usize>(w: &[A; N], g: usize) -> [A; N] {
    match N {
        4 => Fields::pad(w, [(g, g); 2], PadMode::Edge),
        5 => Fields::pad(w, [(g, g); 3], PadMode::Edge),
        _ => unreachable!("2-D or 3-D"),
    }
}

/// Ghost cells of a primitive record: padded by `g` on every axis (edge values), then every face of `bc` assigned
/// through a record view (`Wall` mirrors the interior with the normal velocity negated). Corners keep the edge pad.
pub fn ghosted<const N: usize>(w: &[A; N], g: usize, bc: &dyn Fn(usize, bool) -> Bc) -> [A; N] {
    let mut w = pad_all(w, g);
    for axis in 0..N - 2 {
        let len = w[0].shape().dims()[axis];
        for high in [false, true] {
            let kind = bc(axis, high);
            if kind == Bc::Extrap && g != 1 {
                continue;
            }
            let (dst, src) = if high { (len - g..len, len - 2 * g..len - g) } else { (0..g, g..2 * g) };
            let from: [A; N] = Fields::flip(&Fields::slice(&w, axis, src), axis);
            let rhs: [A; N] = if kind == Bc::Wall { std::array::from_fn(|k| if k == 1 + axis { -&from[k] } else { from[k].clone() }) } else { from };
            w.slice_mut(axis, dst).assign(rhs).unwrap();
        }
    }
    w
}

/// The record `w` (padded by `g` on every axis) restricted to the interior along every axis but `axis`.
fn core<const N: usize>(w: &[A; N], axis: usize, g: usize, n: &[usize]) -> [A; N] {
    let mut out = w.clone();
    for (b, &len) in n.iter().enumerate() {
        if b != axis {
            out = Fields::slice(&out, b, g..g + len);
        }
    }
    out
}

/// Numerical fluxes at the `n[axis] + 1` faces along `axis` of a record padded by `g` (1: first order, 2: MUSCL).
fn faces<const N: usize>(w: &[A; N], axis: usize, g: usize, n: &[usize]) -> [A; N] {
    let wc = core(w, axis, g, n);
    let m = n[axis];
    let s = |k: usize| Fields::slice(&wc, axis, k..k + m + 1);
    if g == 2 {
        let (a, b, c, d) = (s(0), s(1), s(2), s(3));
        let sl: Vec<A> = (0..N).map(|k| minmod(&(&b[k] - &a[k]), &(&c[k] - &b[k]))).collect();
        let sr: Vec<A> = (0..N).map(|k| minmod(&(&c[k] - &b[k]), &(&d[k] - &c[k]))).collect();
        let l: [A; N] = std::array::from_fn(|k| &b[k] + 0.5 * &sl[k]);
        let r: [A; N] = std::array::from_fn(|k| &c[k] - 0.5 * &sr[k]);
        rusanov(&l, &r, axis)
    } else {
        rusanov(&s(0), &s(1), axis)
    }
}

/// `-sum_axes (F_{i+1/2} - F_{i-1/2})` from the ghosted primitive record `wp` of an interior `n`.
pub fn residual<const N: usize>(wp: &[A; N], g: usize, n: &[usize]) -> [A; N] {
    let mut acc: Option<Vec<A>> = None;
    for (axis, &len) in n.iter().enumerate() {
        let f = faces(wp, axis, g, n);
        let term: Vec<A> = (0..N).map(|k| f[k].slice(axis, 1..) - f[k].slice(axis, ..len)).collect();
        acc = Some(match acc {
            None => term.into_iter().map(|t| -t).collect(),
            Some(a) => a.iter().zip(&term).map(|(a, t)| a - t).collect(),
        });
    }
    let acc = acc.unwrap();
    std::array::from_fn(|k| acc[k].clone())
}

/// One explicit step of the conserved record; `wall0` makes the low side of axis 0 a wall (the others keep the edge values).
pub fn explicit_step<const N: usize>(q: &[A; N], muscl: bool, wall0: bool) -> [A; N] {
    let n: Vec<usize> = q[0].shape().dims()[..N - 2].to_vec();
    let g = if muscl { 2 } else { 1 };
    let wp = ghosted(&prim(q), g, &|axis, high| if wall0 && axis == 0 && !high { Bc::Wall } else { Bc::Extrap });
    let r = residual(&wp, g, &n);
    std::array::from_fn(|k| &q[k] + DT * &r[k])
}

/// The density residual of the explicit step, for a stop rule.
pub fn density_residual<const N: usize>(q: &[A; N], muscl: bool, wall0: bool) -> A {
    let n: Vec<usize> = q[0].shape().dims()[..N - 2].to_vec();
    let g = if muscl { 2 } else { 1 };
    let wp = ghosted(&prim(q), g, &|axis, high| if wall0 && axis == 0 && !high { Bc::Wall } else { Bc::Extrap });
    residual(&wp, g, &n)[0].norm()
}

// ---------------------------------------------------------------------------------------------------------------------
// The line-implicit step
// ---------------------------------------------------------------------------------------------------------------------

/// Boundaries of the implicit step: axis 0 a wall below and extrapolation above, axis 1 walls, other axes extrapolation.
pub fn implicit_bc(axis: usize, high: bool) -> Bc {
    match (axis, high) {
        (0, false) | (1, _) => Bc::Wall,
        _ => Bc::Extrap,
    }
}

/// An entry of a block: structurally zero, a constant or an array.
#[derive(Clone)]
enum Ent {
    Z,
    C(f32),
    A(A),
}

impl Ent {
    fn scale(&self, s: &A) -> Ent {
        match self {
            Ent::Z => Ent::Z,
            Ent::C(c) => Ent::A(s * *c),
            Ent::A(a) => Ent::A(a * s),
        }
    }

    fn add(&self, o: &Ent) -> Ent {
        match (self, o) {
            (Ent::Z, x) | (x, Ent::Z) => x.clone(),
            (Ent::C(a), Ent::C(b)) => Ent::C(a + b),
            (Ent::C(c), Ent::A(a)) | (Ent::A(a), Ent::C(c)) => Ent::A(a + *c),
            (Ent::A(a), Ent::A(b)) => Ent::A(a + b),
        }
    }

    fn mul(&self, o: &Ent) -> Ent {
        match (self, o) {
            (Ent::Z, _) | (_, Ent::Z) => Ent::Z,
            (Ent::C(a), Ent::C(b)) => Ent::C(a * b),
            (Ent::C(c), Ent::A(a)) | (Ent::A(a), Ent::C(c)) => Ent::A(a * *c),
            (Ent::A(a), Ent::A(b)) => Ent::A(a * b),
        }
    }

    fn dense(&self, zero: &A) -> A {
        match self {
            Ent::Z => zero.clone(),
            Ent::C(c) => zero + *c,
            Ent::A(a) => a.clone(),
        }
    }
}

/// The flux Jacobian `dF_dir/dq` of the primitive record `w` (`N x N`, row-major).
fn jacobian<const N: usize>(w: &[A; N], dir: usize) -> Vec<Ent> {
    let d = N - 2;
    let g1 = GAMMA - 1.0;
    let k = 0.5 * g1 * sum_sq(w);
    let h = GAMMA / g1 * &w[N - 1] / &w[0] + 0.5 * sum_sq(w);
    let vn = &w[1 + dir];
    let mut a = vec![Ent::Z; N * N];
    a[1 + dir] = Ent::C(1.0);
    for i in 0..d {
        let vi = &w[1 + i];
        a[(1 + i) * N] = Ent::A(if i == dir { &k - vn * vn } else { -(vi * vn) });
        for j in 0..d {
            let vj = &w[1 + j];
            let mut e = Ent::Z;
            if j == dir {
                e = e.add(&Ent::A(vi.clone()));
            }
            if i == j {
                e = e.add(&Ent::A(vn.clone()));
            }
            if i == dir {
                e = e.add(&Ent::A(-(g1 * vj)));
            }
            a[(1 + i) * N + 1 + j] = e;
        }
        if i == dir {
            a[(1 + i) * N + N - 1] = Ent::C(g1);
        }
    }
    a[(N - 1) * N] = Ent::A(vn * (&k - &h));
    for j in 0..d {
        let t = -(g1 * vn * &w[1 + j]);
        a[(N - 1) * N + 1 + j] = Ent::A(if j == dir { &h + t } else { t });
    }
    a[(N - 1) * N + N - 1] = Ent::A(GAMMA * vn);
    a
}

/// The three blocks of the lines of axis `dir`, `N * N` arrays each, evaluated.
pub struct Factor {
    pub lo: Vec<A>,
    pub diag: Vec<A>,
    pub up: Vec<A>,
}

/// Blocks of the line-implicit operator along `dir` from the ghosted primitives `wp` (padded by 1) and the local time step:
/// `lo = -dt/2 (A_- + s_- I)`, `up = dt/2 (A_+ - s_+ I)`, `diag = (1 + dt s) I + dt/(4 s) A^2` with `A_-`, `A_+` the flux Jacobians
/// of the neighbours below and above and `s` the spectral radius `|v_dir| + c`. All `3 N^2` entries are cached together.
pub fn factor<const N: usize>(wp: &[A; N], dt: &A, dir: usize, n: &[usize], zero: &A) -> Factor {
    let c = core(wp, dir, 1, n);
    let m = n[dir];
    let (wl, wc, wr) = (Fields::slice(&c, dir, 0..m), Fields::slice(&c, dir, 1..m + 1), Fields::slice(&c, dir, 2..m + 2));
    let (al, ac, ar) = (jacobian(&wl, dir), jacobian(&wc, dir), jacobian(&wr, dir));
    let (sl, sc, sr) = (wave_speed(&wl, dir), wave_speed(&wc, dir), wave_speed(&wr, dir));
    let (half, quarter) = (0.5 * dt, 0.25 * dt / &sc);
    let mut lo = Vec::with_capacity(N * N);
    let mut up = Vec::with_capacity(N * N);
    let mut diag = Vec::with_capacity(N * N);
    for i in 0..N {
        for j in 0..N {
            let on_diag = i == j;
            lo.push(al[i * N + j].add(&if on_diag { Ent::A(sl.clone()) } else { Ent::Z }).scale(&-&half));
            up.push(ar[i * N + j].add(&if on_diag { Ent::A(-&sr) } else { Ent::Z }).scale(&half));
            let a2 = (0..N).fold(Ent::Z, |s, k| s.add(&ac[i * N + k].mul(&ac[k * N + j])));
            let own = if on_diag { Ent::A(1.0 + dt * &sc) } else { Ent::Z };
            diag.push(own.add(&a2.scale(&quarter)));
        }
    }
    let dense = |v: &[Ent]| -> Vec<A> { v.iter().map(|e| e.dense(zero)).collect() };
    let f = Factor { lo: dense(&lo), diag: dense(&diag), up: dense(&up) };
    let all: Vec<&A> = f.lo.iter().chain(&f.diag).chain(&f.up).collect();
    eval_all(&all).unwrap();
    f
}

/// Block-tridiagonal solve along `axis` of the grid on parts: `b x b` blocks of `lo`, `diag`, `up` (`b * b` arrays each) and the `b` components of `rhs`.
pub fn solve_parts(axis: usize, b: usize, lo: &[A], diag: &[A], up: &[A], rhs: &[A]) -> Vec<A> {
    macro_rules! go {
        ($b2:literal, $b:literal) => {
            solve_block_tridiagonal_parts(
                axis,
                std::array::from_fn::<_, $b2, _>(|k| &lo[k]),
                std::array::from_fn::<_, $b2, _>(|k| &diag[k]),
                std::array::from_fn::<_, $b2, _>(|k| &up[k]),
                std::array::from_fn::<_, $b, _>(|k| &rhs[k]),
            )
            .to_vec()
        };
    }
    match b {
        1 => solve_tridiagonal_parts(axis, [(&lo[0], &diag[0], &up[0], &rhs[0])]).to_vec(),
        2 => go!(4, 2),
        3 => go!(9, 3),
        4 => go!(16, 4),
        5 => go!(25, 5),
        _ => unreachable!("block sizes 1 to 5"),
    }
}

/// `y = a x` for a dense `N x N` block of arrays (row-major) and a vector of arrays.
fn block_mul<const N: usize>(a: &[A], x: &[A; N]) -> [A; N] {
    std::array::from_fn(|i| {
        let mut s = &a[i * N] * &x[0];
        for j in 1..N {
            s = s + &a[i * N + j] * &x[j];
        }
        s
    })
}

/// `(I - dt J_dir) x`: `diag x + lo x_{s-1} + up x_{s+1}`, the neighbours beyond the ends of a line being zero.
fn apply_factor<const N: usize>(f: &Factor, dir: usize, x: &[A; N]) -> [A; N] {
    let mut lower = [0isize; 3];
    lower[dir] = 1;
    let mut upper = [0isize; 3];
    upper[dir] = -1;
    let shifted = |o: [isize; 3]| -> [A; N] {
        if N == 4 { Fields::shift_fill(x, [o[0], o[1]], 0.0) } else { Fields::shift_fill(x, o, 0.0) }
    };
    let (xm, xp) = (shifted(lower), shifted(upper));
    let (dx, lx, ux) = (block_mul(&f.diag, x), block_mul(&f.lo, &xm), block_mul(&f.up, &xp));
    std::array::from_fn(|k| &dx[k] + &lx[k] + &ux[k])
}

/// The approximate factorisation `(I - dt J_{D-1})^{-1} .. (I - dt J_0)^{-1} rhs`: a solve along every axis in turn.
fn approx_factor<const N: usize>(fs: &[Factor], rhs: &[A; N]) -> [A; N] {
    let mut x: Vec<A> = rhs.to_vec();
    for (axis, f) in fs.iter().enumerate() {
        x = solve_parts(axis, N, &f.lo, &f.diag, &f.up, &x);
    }
    std::array::from_fn(|k| x[k].clone())
}

/// The pieces of an implicit step before the solves, for the cases that time them apart.
pub struct Lhs<const N: usize> {
    /// `dt R` of the first-order residual.
    pub rhs: [A; N],
    pub rn: A,
    pub factors: Vec<Factor>,
}

/// Ghost cells, residual, time step and block coefficients of an implicit step from the conserved record `q`
/// (`cfl` is an array of one element, as the ramp in the loop state of a solver; `zero` an evaluated array of zeros of the grid's shape,
/// made once outside a traced step).
pub fn assemble<const N: usize>(q: &[A; N], cfl: &A, zero: &A) -> Lhs<N> {
    let d = N - 2;
    let n: Vec<usize> = q[0].shape().dims()[..d].to_vec();
    let wp = Fields::cache(&ghosted(&prim(q), 1, &implicit_bc));
    let r0 = residual(&wp, 1, &n);
    let rn = r0[0].norm();
    let wk: [A; N] = (0..d).fold(wp.clone(), |w, a| Fields::slice(&w, a, 1..n[a] + 1));
    let mut sig = wave_speed(&wk, 0);
    for a in 1..d {
        sig = sig + wave_speed(&wk, a);
    }
    let dt = cfl / &sig;
    let rhs: [A; N] = Fields::cache(&std::array::from_fn::<A, N, _>(|k| &r0[k] * &dt));
    let factors = (0..d).map(|a| factor(&wp, &dt, a, &n, zero)).collect();
    Lhs { rhs, rn, factors }
}

/// One implicit iteration of the conserved record `q`: the new state and the density residual of its start.
/// `X = AF^{-1} (dt R)`, then `sub_iters - 1` defect corrections `X += AF^{-1} (b - (sum_a A_a - (D - 1) I) X)`, `q += X`
/// where the density stays positive.
pub fn implicit_step<const N: usize>(q: &[A; N], cfl: &A, zero: &A, sub_iters: usize) -> ([A; N], A) {
    let d = N - 2;
    let l = assemble(q, cfl, zero);
    let mut du = approx_factor(&l.factors, &l.rhs);
    for _ in 1..sub_iters {
        let ax: Vec<[A; N]> = (0..d).map(|a| apply_factor(&l.factors[a], a, &du)).collect();
        let r: [A; N] = std::array::from_fn(|k| {
            let mut sum = &ax[0][k] - (d as f32 - 1.0) * &du[k];
            for a in &ax[1..] {
                sum += &a[k];
            }
            &l.rhs[k] - sum
        });
        let corr = approx_factor(&l.factors, &Fields::cache(&r));
        du = std::array::from_fn(|k| &du[k] + &corr[k]);
    }
    let moved = (&q[0] + &du[0]).gt(0.0);
    let un = std::array::from_fn(|k| select(&moved, &q[k] + &du[k], &q[k]));
    (un, l.rn)
}
