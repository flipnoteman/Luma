//! f64 host references of the `cfd` suite: the Euler residual, the line-implicit step, block
//! Thomas solves and the grid operators (stencils, Poisson smoothers, multigrid transfers). They
//! mirror the array code of `euler.rs` and `cfd.rs` formula by formula, in double precision and
//! with plain loops, so a mismatch is the library's.

pub const GAMMA: f64 = 1.4;

/// The values of one cell: `[rho, momentum or velocity.., energy or pressure]`.
pub type Cell<const N: usize> = [f64; N];

/// Whether the ghost cells below (`high == false`) or above the grid along `axis` are a wall.
pub type Walls<'a> = &'a dyn Fn(usize, bool) -> bool;

/// Grid extents (unused axes are 1), axis 0 fastest like luma's arrays.
#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub n: [usize; 3],
    /// Number of dimensions in use.
    pub d: usize,
}

impl Grid {
    pub fn new(dims: &[usize]) -> Grid {
        let mut n = [1; 3];
        n[..dims.len()].copy_from_slice(dims);
        Grid { n, d: dims.len() }
    }

    pub fn cells(&self) -> usize {
        self.n.iter().product()
    }

    pub fn flat(&self, c: [usize; 3]) -> usize {
        c[0] + self.n[0] * (c[1] + self.n[1] * c[2])
    }

    pub fn coords(&self, q: usize) -> [usize; 3] {
        [q % self.n[0], (q / self.n[0]) % self.n[1], q / (self.n[0] * self.n[1])]
    }

    /// Cells between two neighbours along `axis`.
    pub fn stride(&self, axis: usize) -> usize {
        self.n[..axis].iter().product()
    }

    /// The grid with every axis in use halved.
    pub fn half(&self) -> Grid {
        let mut n = self.n;
        for e in n.iter_mut().take(self.d) {
            *e /= 2;
        }
        Grid { n, d: self.d }
    }
}

/// Component-major host data (one vector per field) as cells.
pub fn to_cells<const N: usize>(h: &[Vec<f32>]) -> Vec<Cell<N>> {
    (0..h[0].len()).map(|q| std::array::from_fn(|k| f64::from(h[k][q]))).collect()
}

pub fn from_cells<const N: usize>(cells: &[Cell<N>]) -> Vec<Vec<f32>> {
    (0..N).map(|k| cells.iter().map(|c| c[k] as f32).collect()).collect()
}

// ---- Euler ----

pub fn h_prim<const N: usize>(q: Cell<N>) -> Cell<N> {
    let mut w = q;
    let mut ke = 0.0;
    for k in 0..N - 2 {
        w[1 + k] = q[1 + k] / q[0];
        ke += 0.5 * q[0] * w[1 + k] * w[1 + k];
    }
    w[N - 1] = (GAMMA - 1.0) * (q[N - 1] - ke);
    w
}

pub fn h_cons<const N: usize>(w: Cell<N>) -> Cell<N> {
    let mut q = w;
    let mut ke = 0.0;
    for k in 0..N - 2 {
        q[1 + k] = w[0] * w[1 + k];
        ke += 0.5 * w[0] * w[1 + k] * w[1 + k];
    }
    q[N - 1] = w[N - 1] / (GAMMA - 1.0) + ke;
    q
}

pub fn h_flux<const N: usize>(w: Cell<N>, dir: usize) -> Cell<N> {
    let q = h_cons(w);
    let un = w[1 + dir];
    let mut f = [0.0; N];
    f[0] = q[1 + dir];
    for k in 0..N - 2 {
        f[1 + k] = q[1 + k] * un + if k == dir { w[N - 1] } else { 0.0 };
    }
    f[N - 1] = (q[N - 1] + w[N - 1]) * un;
    f
}

pub fn speed<const N: usize>(w: &Cell<N>, dir: usize) -> f64 {
    w[1 + dir].abs() + (GAMMA * w[N - 1] / w[0]).sqrt()
}

pub fn h_rus<const N: usize>(l: Cell<N>, r: Cell<N>, dir: usize) -> Cell<N> {
    let smax = speed(&l, dir).max(speed(&r, dir));
    let (ul, ur, fl, fr) = (h_cons(l), h_cons(r), h_flux(l, dir), h_flux(r, dir));
    std::array::from_fn(|k| 0.5 * (fl[k] + fr[k]) - 0.5 * smax * (ur[k] - ul[k]))
}

fn h_minmod(a: f64, b: f64) -> f64 {
    if a * b > 0.0 {
        if a.abs() < b.abs() { a } else { b }
    } else {
        0.0
    }
}

/// Primitive cells with ghost lookup: a wall ghost mirrors the cell across the face with the
/// normal velocity negated, any other ghost holds the edge cell.
pub struct Field<'a, const N: usize> {
    pub grid: Grid,
    pub prim: &'a [Cell<N>],
    pub wall: Walls<'a>,
}

impl<const N: usize> Field<'_, N> {
    pub fn at(&self, c: [isize; 3]) -> Cell<N> {
        let mut cc = [0usize; 3];
        let mut flip = None;
        for a in 0..3 {
            let n = self.grid.n[a] as isize;
            let x = c[a];
            if x < 0 {
                if a < self.grid.d && (self.wall)(a, false) {
                    cc[a] = (-x - 1) as usize;
                    flip = Some(a);
                }
            } else if x >= n {
                if a < self.grid.d && (self.wall)(a, true) {
                    cc[a] = (2 * n - 1 - x).max(0) as usize;
                    flip = Some(a);
                } else {
                    cc[a] = (n - 1) as usize;
                }
            } else {
                cc[a] = x as usize;
            }
        }
        let mut w = self.prim[self.grid.flat(cc)];
        if let Some(a) = flip {
            w[1 + a] = -w[1 + a];
        }
        w
    }
}

/// `-sum_axes (F_{i+1/2} - F_{i-1/2})` of every cell by Rusanov fluxes (MUSCL with minmod slopes when `muscl`).
pub fn h_residual<const N: usize>(f: &Field<N>, muscl: bool) -> Vec<Cell<N>> {
    let g = f.grid;
    let mut out = vec![[0.0; N]; g.cells()];
    for (q, res_cell) in out.iter_mut().enumerate() {
        let co = g.coords(q);
        let ci = [co[0] as isize, co[1] as isize, co[2] as isize];
        let mut res = [0.0; N];
        for a in 0..g.d {
            // The face between cells `lo` and `lo + 1` along `a`.
            let face = |lo: isize| -> Cell<N> {
                let at = |o: isize| {
                    let mut cc = ci;
                    cc[a] = lo + o;
                    f.at(cc)
                };
                let (l, r) = if muscl {
                    let (m1, c0, c1, c2) = (at(-1), at(0), at(1), at(2));
                    let sl: Cell<N> = std::array::from_fn(|k| h_minmod(c0[k] - m1[k], c1[k] - c0[k]));
                    let sr: Cell<N> = std::array::from_fn(|k| h_minmod(c1[k] - c0[k], c2[k] - c1[k]));
                    (std::array::from_fn(|k| c0[k] + 0.5 * sl[k]), std::array::from_fn(|k| c1[k] - 0.5 * sr[k]))
                } else {
                    (at(0), at(1))
                };
                h_rus(l, r, a)
            };
            let (fh, fl) = (face(ci[a]), face(ci[a] - 1));
            for k in 0..N {
                res[k] -= fh[k] - fl[k];
            }
        }
        *res_cell = res;
    }
    out
}

/// `steps` explicit Euler steps `q += dt R` of component-major `h`.
pub fn h_explicit_run<const N: usize>(h: &[Vec<f32>], g: Grid, wall: Walls, muscl: bool, dt: f64, steps: usize) -> Vec<Vec<f32>> {
    let mut q: Vec<Cell<N>> = to_cells(h);
    for _ in 0..steps {
        let prim: Vec<Cell<N>> = q.iter().map(|&c| h_prim(c)).collect();
        let r = h_residual(&Field { grid: g, prim: &prim, wall }, muscl);
        for (c, r) in q.iter_mut().zip(&r) {
            for k in 0..N {
                c[k] += dt * r[k];
            }
        }
    }
    from_cells(&q)
}

// ---- block Thomas solves ----

/// Dense solve of `m x = r` (`n x n`, row-major) with partial pivoting.
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

/// Block Thomas solve of every line of `grid` along `axis`: blocks of `b x b` (`lo`, `diag`, `up`
/// hold `b * b` row-major entries per cell, `rhs` `b`), each pivot solved densely.
pub fn host_tri(b: usize, grid: [usize; 3], axis: usize, lo: &[f64], diag: &[f64], up: &[f64], rhs: &[f64]) -> Vec<f64> {
    let cells: usize = grid.iter().product();
    let n = grid[axis];
    let stride: usize = grid[..axis].iter().product();
    let mut x = vec![0.0f64; cells * b];
    let block = |a: &[f64], q: usize| -> Vec<f64> { a[q * b * b..(q + 1) * b * b].to_vec() };
    for l in 0..cells / n {
        let base = (l % stride) + (l / stride) * stride * n;
        let mut c: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut d: Vec<Vec<f64>> = Vec::with_capacity(n);
        for s in 0..n {
            let q = base + s * stride;
            let mut m = block(diag, q);
            let mut r: Vec<f64> = rhs[q * b..(q + 1) * b].to_vec();
            if s > 0 {
                let lq = block(lo, q);
                for i in 0..b {
                    for k in 0..b {
                        for j in 0..b {
                            m[i * b + j] -= lq[i * b + k] * c[s - 1][k * b + j];
                        }
                        r[i] -= lq[i * b + k] * d[s - 1][k];
                    }
                }
            }
            let uq = block(up, q);
            let mut cs = vec![0.0f64; b * b];
            for j in 0..b {
                let column = dense_solve(m.clone(), (0..b).map(|i| uq[i * b + j]).collect(), b);
                for i in 0..b {
                    cs[i * b + j] = column[i];
                }
            }
            d.push(dense_solve(m, r, b));
            c.push(cs);
        }
        let mut next = vec![0.0f64; b];
        for s in (0..n).rev() {
            let q = base + s * stride;
            let cur: Vec<f64> = (0..b).map(|i| d[s][i] - (0..b).map(|k| c[s][i * b + k] * next[k]).sum::<f64>()).collect();
            x[q * b..(q + 1) * b].copy_from_slice(&cur);
            next = cur;
        }
    }
    x
}

/// The residual of a block-tridiagonal solve in double precision: the largest
/// `|lo x[s-1] + diag x[s] + up x[s+1] - rhs|` over the cells and components of `x` (`f32` data, `b x b`
/// blocks, cell-major like `host_tri`), the first and last cell of a line having no neighbour on that side.
pub fn line_residual(b: usize, grid: [usize; 3], axis: usize, sys: &[Vec<f32>; 4], x: &[f32]) -> f64 {
    let cells: usize = grid.iter().product();
    let n = grid[axis];
    let stride: usize = grid[..axis].iter().product();
    let [lo, diag, up, rhs] = sys;
    let mut worst = 0.0f64;
    for q in 0..cells {
        let s = (q / stride) % n;
        for i in 0..b {
            let mut r = -f64::from(rhs[q * b + i]);
            for j in 0..b {
                r += f64::from(diag[q * b * b + i * b + j]) * f64::from(x[q * b + j]);
                if s > 0 {
                    r += f64::from(lo[q * b * b + i * b + j]) * f64::from(x[(q - stride) * b + j]);
                }
                if s + 1 < n {
                    r += f64::from(up[q * b * b + i * b + j]) * f64::from(x[(q + stride) * b + j]);
                }
            }
            worst = worst.max(r.abs());
        }
    }
    worst
}

// ---- the line-implicit step ----

/// The flux Jacobian `dF_dir/dq` of a primitive cell.
pub fn h_jac<const N: usize>(w: &Cell<N>, dir: usize) -> [[f64; N]; N] {
    let d = N - 2;
    let g1 = GAMMA - 1.0;
    let q2: f64 = (0..d).map(|k| w[1 + k] * w[1 + k]).sum();
    let k = 0.5 * g1 * q2;
    let h = GAMMA / g1 * w[N - 1] / w[0] + 0.5 * q2;
    let vn = w[1 + dir];
    let mut a = [[0.0; N]; N];
    a[0][1 + dir] = 1.0;
    for i in 0..d {
        let vi = w[1 + i];
        a[1 + i][0] = if i == dir { k } else { 0.0 } - vi * vn;
        for j in 0..d {
            let vj = w[1 + j];
            a[1 + i][1 + j] = if j == dir { vi } else { 0.0 } + if i == j { vn } else { 0.0 } - if i == dir { g1 * vj } else { 0.0 };
        }
        a[1 + i][N - 1] = if i == dir { g1 } else { 0.0 };
    }
    a[N - 1][0] = vn * (k - h);
    for j in 0..d {
        a[N - 1][1 + j] = if j == dir { h } else { 0.0 } - g1 * vn * w[1 + j];
    }
    a[N - 1][N - 1] = GAMMA * vn;
    a
}

/// One implicit step of the suite's flux-vector-split line solver (see `euler::implicit_step`): first order Rusanov
/// residual, `dt = cfl / sum_axes(|v| + c)`, approximate factorisation along every axis with the blocks
/// `lo = -dt/2 (A_- + s_- I)`, `up = dt/2 (A_+ - s_+ I)`, `diag = (1 + dt s) I + dt/(4 s) A^2`, and `sub_iters - 1`
/// defect corrections `r = b - (sum_a A_a - (D - 1) I) du`.
pub fn h_implicit_step<const N: usize>(q: &[Cell<N>], g: Grid, wall: Walls, cfl: f64, sub_iters: usize) -> Vec<Cell<N>> {
    let d = g.d;
    let (bb, cells) = (N * N, g.cells());
    let prim: Vec<Cell<N>> = q.iter().map(|&c| h_prim(c)).collect();
    let f = Field { grid: g, prim: &prim, wall };
    let r0 = h_residual(&f, false);
    let sig = |w: &Cell<N>, a: usize| speed(w, a);
    let dt: Vec<f64> = prim.iter().map(|w| cfl / (0..d).map(|a| sig(w, a)).sum::<f64>()).collect();
    let mut facs: Vec<[Vec<f64>; 3]> = Vec::new();
    for a in 0..d {
        let (mut lo, mut dg, mut up) = (vec![0.0; cells * bb], vec![0.0; cells * bb], vec![0.0; cells * bb]);
        for c in 0..cells {
            let co = g.coords(c);
            let ci = [co[0] as isize, co[1] as isize, co[2] as isize];
            let at = |o: isize| {
                let mut cc = ci;
                cc[a] += o;
                f.at(cc)
            };
            let (wl, wr) = (at(-1), at(1));
            let wc = prim[c];
            let (al, ac, ar) = (h_jac(&wl, a), h_jac(&wc, a), h_jac(&wr, a));
            let (sl, sc, sr) = (sig(&wl, a), sig(&wc, a), sig(&wr, a));
            let t = dt[c];
            for i in 0..N {
                for j in 0..N {
                    let delta = if i == j { 1.0 } else { 0.0 };
                    let e = c * bb + i * N + j;
                    lo[e] = -0.5 * t * (al[i][j] + delta * sl);
                    up[e] = 0.5 * t * (ar[i][j] - delta * sr);
                    let a2: f64 = (0..N).map(|k| ac[i][k] * ac[k][j]).sum();
                    dg[e] = delta * (1.0 + t * sc) + 0.25 * t / sc * a2;
                }
            }
        }
        facs.push([lo, dg, up]);
    }
    let approx = |rhs: &[f64]| -> Vec<f64> {
        let mut x = rhs.to_vec();
        for (a, fac) in facs.iter().enumerate() {
            x = host_tri(N, g.n, a, &fac[0], &fac[1], &fac[2], &x);
        }
        x
    };
    let b: Vec<f64> = (0..cells * N).map(|i| r0[i / N][i % N] * dt[i / N]).collect();
    let mut du = approx(&b);
    for _ in 1..sub_iters {
        let mut ad = vec![0.0; cells * N];
        for (a, fac) in facs.iter().enumerate() {
            let st = g.stride(a);
            for c in 0..cells {
                let ca = g.coords(c)[a];
                for i in 0..N {
                    let mut s = 0.0;
                    for j in 0..N {
                        s += fac[1][c * bb + i * N + j] * du[c * N + j];
                        if ca > 0 {
                            s += fac[0][c * bb + i * N + j] * du[(c - st) * N + j];
                        }
                        if ca + 1 < g.n[a] {
                            s += fac[2][c * bb + i * N + j] * du[(c + st) * N + j];
                        }
                    }
                    ad[c * N + i] += s;
                }
            }
        }
        let r: Vec<f64> = (0..cells * N).map(|i| b[i] - (ad[i] - (d as f64 - 1.0) * du[i])).collect();
        let corr = approx(&r);
        for (x, c) in du.iter_mut().zip(&corr) {
            *x += c;
        }
    }
    (0..cells)
        .map(|c| {
            let moved = q[c][0] + du[c * N] > 0.0;
            std::array::from_fn(|k| if moved { q[c][k] + du[c * N + k] } else { q[c][k] })
        })
        .collect()
}

// ---- scalar grid operators ----

/// `steps` explicit heat steps `u += alpha (sum of the neighbours - 2 d u)` with edge-clamped (zero flux) boundaries.
pub fn h_heat(h: &[f32], g: Grid, alpha: f64, steps: usize) -> Vec<f32> {
    let mut u: Vec<f64> = h.iter().map(|&v| f64::from(v)).collect();
    for _ in 0..steps {
        let mut next = u.clone();
        for (q, out) in next.iter_mut().enumerate() {
            let co = g.coords(q);
            let mut lap = -2.0 * g.d as f64 * u[q];
            for a in 0..g.d {
                let st = g.stride(a);
                lap += u[if co[a] > 0 { q - st } else { q }] + u[if co[a] + 1 < g.n[a] { q + st } else { q }];
            }
            *out = u[q] + alpha * lap;
        }
        u = next;
    }
    u.into_iter().map(|v| v as f32).collect()
}

/// `sum_a (u[+a] + u[-a]) - 2 d u` with zero outside the grid (the Dirichlet Laplacian, positive off the diagonal).
pub fn h_lap(u: &[f64], g: Grid) -> Vec<f64> {
    (0..u.len())
        .map(|q| {
            let co = g.coords(q);
            let mut lap = -2.0 * g.d as f64 * u[q];
            for a in 0..g.d {
                let st = g.stride(a);
                if co[a] > 0 {
                    lap += u[q - st];
                }
                if co[a] + 1 < g.n[a] {
                    lap += u[q + st];
                }
            }
            lap
        })
        .collect()
}

/// `steps` delta-form ADI steps of variable-coefficient heat, `(I - r k dxx)` factors solved along every axis (Dirichlet zero):
/// `du = r k lap(u)`, then `du <- (I - r k d_aa)^{-1} du` along each axis, `u += du`.
pub fn h_adi(h: &[f32], kappa: &[f32], g: Grid, r: f64, steps: usize) -> Vec<f32> {
    let mut u: Vec<f64> = h.iter().map(|&v| f64::from(v)).collect();
    let cells = g.cells();
    let k: Vec<f64> = kappa.iter().map(|&v| f64::from(v)).collect();
    let lo: Vec<f64> = k.iter().map(|k| -r * k).collect();
    let dg: Vec<f64> = k.iter().map(|k| 1.0 + 2.0 * r * k).collect();
    for _ in 0..steps {
        let lap = h_lap(&u, g);
        let mut du: Vec<f64> = (0..cells).map(|q| r * k[q] * lap[q]).collect();
        for a in 0..g.d {
            du = host_tri(1, g.n, a, &lo, &dg, &lo, &du);
        }
        for q in 0..cells {
            u[q] += du[q];
        }
    }
    u.into_iter().map(|v| v as f32).collect()
}

/// Weighted Jacobi sweeps for `-lap u = b` (unit spacing), Dirichlet zero.
pub fn h_jacobi(u: &mut [f64], b: &[f64], g: Grid, omega: f64, sweeps: usize) {
    let diag = 2.0 * g.d as f64;
    for _ in 0..sweeps {
        let lap = h_lap(u, g);
        for q in 0..u.len() {
            // (b + neighbours) / diag = u + (b + lap u) / diag
            u[q] += omega * (b[q] + lap[q]) / diag;
        }
    }
}

/// `iters` red-black Gauss-Seidel iterations (red cells, then black) of `-lap u = b`.
pub fn h_redblack(u: &mut [f64], b: &[f64], g: Grid, iters: usize) {
    let diag = 2.0 * g.d as f64;
    for _ in 0..iters {
        for color in 0..2 {
            let lap = h_lap(u, g);
            for q in 0..u.len() {
                let co = g.coords(q);
                if (co[0] + co[1] + co[2]) % 2 == color {
                    u[q] += (b[q] + lap[q]) / diag;
                }
            }
        }
    }
}

/// `iters` iterations of conjugate gradients on `-lap x = b` from zero.
pub fn h_cg(b: &[f64], g: Grid, iters: usize) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rr: f64 = r.iter().map(|v| v * v).sum();
    for _ in 0..iters {
        let ap: Vec<f64> = h_lap(&p, g).into_iter().map(|v| -v).collect();
        let alpha = rr / p.iter().zip(&ap).map(|(a, b)| a * b).sum::<f64>();
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rr_new: f64 = r.iter().map(|v| v * v).sum();
        let beta = rr_new / rr;
        rr = rr_new;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
    }
    x
}

// ---- multigrid transfers ----

/// Average of every 2^d block of cells (`g` has even extents).
pub fn h_restrict(f: &[f64], g: Grid) -> Vec<f64> {
    let c = g.half();
    let weight = 1.0 / (1usize << g.d) as f64;
    (0..c.cells())
        .map(|q| {
            let co = c.coords(q);
            let mut s = 0.0;
            for m in 0..1usize << g.d {
                let mut fc = [0usize; 3];
                for a in 0..3 {
                    fc[a] = if a < g.d { 2 * co[a] + ((m >> a) & 1) } else { 0 };
                }
                s += f[g.flat(fc)];
            }
            s * weight
        })
        .collect()
}

/// Piecewise-constant prolongation onto the grid `g` (even extents) from its half-size coarse field.
pub fn h_prolong_pc(c: &[f64], g: Grid) -> Vec<f64> {
    let cg = g.half();
    (0..g.cells())
        .map(|q| {
            let co = g.coords(q);
            let mut cc = [0usize; 3];
            for a in 0..g.d {
                cc[a] = co[a] / 2;
            }
            c[cg.flat(cc)]
        })
        .collect()
}

/// Bilinear (cell-centred, 9/16 3/16 3/16 1/16) prolongation of a 2-D field, edge values replicated.
pub fn h_prolong_bilinear(c: &[f64], cg: Grid) -> Vec<f64> {
    let (cx, cy) = (cg.n[0], cg.n[1]);
    let mut out = vec![0.0; 4 * cx * cy];
    for y in 0..2 * cy {
        for x in 0..2 * cx {
            let (i, j) = (x / 2, y / 2);
            let ni = if x % 2 == 0 { i.saturating_sub(1) } else { (i + 1).min(cx - 1) };
            let nj = if y % 2 == 0 { j.saturating_sub(1) } else { (j + 1).min(cy - 1) };
            let at = |a: usize, b: usize| c[a + cx * b];
            out[x + 2 * cx * y] = (9.0 * at(i, j) + 3.0 * at(ni, j) + 3.0 * at(i, nj) + at(ni, nj)) / 16.0;
        }
    }
    out
}

/// One V-cycle of weighted Jacobi (`pre` and `post` sweeps, piecewise-constant transfers) for `-lap u = b`,
/// `levels` grids deep; the coarsest grid is relaxed `coarse` sweeps.
#[allow(clippy::too_many_arguments)]
pub fn h_vcycle(u: &mut [f64], b: &[f64], g: Grid, levels: usize, omega: f64, pre: usize, post: usize, coarse: usize) {
    if levels == 1 {
        h_jacobi(u, b, g, omega, coarse);
        return;
    }
    h_jacobi(u, b, g, omega, pre);
    let lap = h_lap(u, g);
    let res: Vec<f64> = (0..u.len()).map(|q| b[q] + lap[q]).collect();
    let cg = g.half();
    // The coarse grid has spacing 2: its unit-spacing stencil is 4 times the operator, so the restricted residual is scaled by 4.
    let rc: Vec<f64> = h_restrict(&res, g).into_iter().map(|v| 4.0 * v).collect();
    let mut e = vec![0.0; rc.len()];
    h_vcycle(&mut e, &rc, cg, levels - 1, omega, pre, post, coarse);
    let ef = h_prolong_pc(&e, g);
    for q in 0..u.len() {
        u[q] += ef[q];
    }
    h_jacobi(u, b, g, omega, post);
}
