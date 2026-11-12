//! The first-order Euler residual of a finite-volume solver written as array expressions: records
//! of fields, a padded primitive state with a mirror-wall ghost assignment, slices as face
//! states, a Rusanov flux and `cache()`, compared with an `f64` host implementation.

mod common;

use common::{assert_close, values};
use luma::{Array, Context, Fields, PadMode};

const GAMMA: f32 = 1.4;
const NI: usize = 16;
const NJ: usize = 12;

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

fn prim(q: &Cons) -> Prim {
    let u = &q.mu / &q.rho;
    let v = &q.mv / &q.rho;
    let p = (GAMMA - 1.0) * (&q.e - 0.5 * &q.rho * (&u * &u + &v * &v));
    Prim { rho: &q.rho * 1.0, u, v, p }
}

fn cons(w: &Prim) -> Cons {
    let ke = 0.5 * &w.rho * (&w.u * &w.u + &w.v * &w.v);
    Cons { rho: &w.rho * 1.0, mu: &w.rho * &w.u, mv: &w.rho * &w.v, e: &w.p / (GAMMA - 1.0) + ke }
}

/// The flux in the first direction of the state `w`.
fn flux(w: &Prim) -> Cons {
    let q = cons(w);
    Cons {
        rho: &q.mu * 1.0,
        mu: &q.mu * &w.u + &w.p,
        mv: &q.mv * &w.u,
        e: (&q.e + &w.p) * &w.u,
    }
}

/// The wall condition of the left boundary: the normal velocity changes sign.
fn mirror(w: Prim) -> Prim {
    Prim { rho: w.rho, u: -&w.u, v: w.v, p: w.p }
}

fn rusanov(l: &Prim, r: &Prim) -> Cons {
    let speed = |w: &Prim| w.u.abs() + (GAMMA * &w.p / &w.rho).sqrt();
    let smax = speed(l).maximum(speed(r));
    let (ul, ur) = (cons(l), cons(r));
    let (fl, fr) = (flux(l), flux(r));
    0.5 * (&fl + &fr) - 0.5 * &smax * (&ur - &ul)
}

// ---- the same in f64 on the host ----

type Cell = [f64; 4];

fn host_prim(q: Cell) -> Cell {
    let g = f64::from(GAMMA);
    let (u, v) = (q[1] / q[0], q[2] / q[0]);
    [q[0], u, v, (g - 1.0) * (q[3] - 0.5 * q[0] * (u * u + v * v))]
}

fn host_cons(w: Cell) -> Cell {
    let g = f64::from(GAMMA);
    [w[0], w[0] * w[1], w[0] * w[2], w[3] / (g - 1.0) + 0.5 * w[0] * (w[1] * w[1] + w[2] * w[2])]
}

fn host_flux(w: Cell) -> Cell {
    let q = host_cons(w);
    [q[1], q[1] * w[1] + w[3], q[2] * w[1], (q[3] + w[3]) * w[1]]
}

fn host_rusanov(l: Cell, r: Cell) -> Cell {
    let g = f64::from(GAMMA);
    let speed = |w: Cell| w[1].abs() + (g * w[3] / w[0]).sqrt();
    let smax = speed(l).max(speed(r));
    let (ul, ur, fl, fr) = (host_cons(l), host_cons(r), host_flux(l), host_flux(r));
    std::array::from_fn(|k| 0.5 * (fl[k] + fr[k]) - 0.5 * smax * (ur[k] - ul[k]))
}

#[test]
fn the_euler_residual_of_records_matches_an_f64_host_implementation() {
    let ctx = Context::new().unwrap();
    let n = NI * NJ;
    let host: [Vec<f32>; 4] = [
        values(n, 1, 0.8, 1.2),
        values(n, 2, -0.3, 0.3),
        values(n, 3, -0.3, 0.3),
        values(n, 4, 2.0, 3.0),
    ];
    let upload = |k: usize| Array::from_slice_on(&ctx, &host[k], [NI, NJ]).unwrap();
    let q = Cons { rho: upload(0), mu: upload(1), mv: upload(2), e: upload(3) };

    // Primitive state with two ghost layers; the left wall mirrors the first two interior columns.
    let w = prim(&q).pad([(2, 2), (2, 2)], PadMode::Edge);
    w.slice(0, ..2).assign(mirror(w.slice(0, 2..4).flip(0))).unwrap();

    // Faces `0..=NI` along the first axis, interior rows only.
    let l = w.slice(0, 1..NI + 2).slice(1, 2..NJ + 2);
    let r = w.slice(0, 2..NI + 3).slice(1, 2..NJ + 2);
    let fi = rusanov(&l, &r).cache();
    let residual = -(fi.slice(0, 1..) - fi.slice(0, ..NI));
    let got = [
        residual.rho.to_vec().unwrap(),
        residual.mu.to_vec().unwrap(),
        residual.mv.to_vec().unwrap(),
        residual.e.to_vec().unwrap(),
    ];

    // Host: the cell `c` of a row, `-2..NI + 2` (ghosts of the left wall mirror, right edge copies).
    let cell = |c: isize, j: usize| -> Cell {
        let at = |i: usize| host_prim(std::array::from_fn(|k| f64::from(host[k][i + NI * j])));
        match c {
            ..=-1 => {
                let mut w = at((-c - 1) as usize);
                w[1] = -w[1];
                w
            }
            c if c as usize >= NI => at(NI - 1),
            c => at(c as usize),
        }
    };
    let mut want: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.0; n]);
    for j in 0..NJ {
        let face = |f: isize| host_rusanov(cell(f - 1, j), cell(f, j));
        for i in 0..NI {
            let (lo, hi) = (face(i as isize), face(i as isize + 1));
            for k in 0..4 {
                want[k][i + NI * j] = (-(hi[k] - lo[k])) as f32;
            }
        }
    }
    for k in 0..4 {
        assert_close(&got[k], &want[k], 2e-4, 2e-4);
    }
}
