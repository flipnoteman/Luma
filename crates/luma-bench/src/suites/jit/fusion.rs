//! Lazy fusion: long chains, many scalars, many inputs, many outputs, DAGs with shared
//! subexpressions, and views (pad, slices, shifts, transposes) inside fused kernels.

use super::common::*;
use crate::harness::*;
use luma::*;

/// One single-node operation of a long chain.
#[derive(Clone, Copy)]
enum U {
    Add(f32),
    Mul(f32),
    Abs,
    MaxC(f32),
    MinC(f32),
    Sin,
    Tanh,
}

fn unary_chain(seed: u64, len: usize) -> Vec<U> {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    (0..len)
        .map(|_| match rng.below(7) {
            0 => U::Add(rng.range(-0.3, 0.3)),
            1 => U::Mul(rng.range(0.7, 1.25)),
            2 => U::Abs,
            3 => U::MaxC(rng.range(-1.0, 0.2)),
            4 => U::MinC(rng.range(0.5, 2.0)),
            5 => U::Sin,
            _ => U::Tanh,
        })
        .collect()
}

fn chain_dev(ops: &[U], x: &A) -> A {
    let mut a = x.clone();
    for op in ops {
        a = match *op {
            U::Add(c) => &a + c,
            U::Mul(c) => &a * c,
            U::Abs => a.abs(),
            U::MaxC(c) => a.maximum(c),
            U::MinC(c) => a.minimum(c),
            U::Sin => a.sin(),
            U::Tanh => a.tanh(),
        };
    }
    a
}

fn chain_host(ops: &[U], x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let mut a = f64::from(v);
            for op in ops {
                a = match *op {
                    U::Add(c) => a + f64::from(c),
                    U::Mul(c) => a * f64::from(c),
                    U::Abs => a.abs(),
                    U::MaxC(c) => a.max(f64::from(c)),
                    U::MinC(c) => a.min(f64::from(c)),
                    U::Sin => a.sin(),
                    U::Tanh => a.tanh(),
                };
            }
            a as f32
        })
        .collect()
}

/// `x`, 64 rounds of `min(|x * c1 + c2|, c3)`: three scalars per round.
fn scalars_dev(x: &A, c: &[f32]) -> A {
    let mut a = x.clone();
    for r in c.chunks(3) {
        a = (&a * r[0] + r[1]).abs().minimum(r[2]);
    }
    a
}

fn scalars_host(x: &[f32], c: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let mut a = f64::from(v);
            for r in c.chunks(3) {
                a = (a * f64::from(r[0]) + f64::from(r[1])).abs().min(f64::from(r[2]));
            }
            a as f32
        })
        .collect()
}

/// The first 48 levels of `t[k] = sin(t[k-1]) * 0.7 + t[k-2] * 0.3`, each level used twice: a DAG
/// of ~200 nodes whose tree is exponentially big.
fn dag_dev(x: &A, y: &A, levels: usize) -> A {
    let (mut prev, mut cur) = (x.clone(), y.clone());
    for _ in 0..levels {
        let next = cur.sin() * 0.7 + &prev * 0.3;
        prev = cur;
        cur = next;
    }
    cur
}

fn dag_host(x: &[f32], y: &[f32], levels: usize) -> Vec<f32> {
    x.iter()
        .zip(y)
        .map(|(&x, &y)| {
            let (mut prev, mut cur) = (f64::from(x), f64::from(y));
            for _ in 0..levels {
                let next = cur.sin() * 0.7 + prev * 0.3;
                prev = cur;
                cur = next;
            }
            cur as f32
        })
        .collect()
}

/// `n` complex numbers with components in `lo..hi`.
fn complex_values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<C32> {
    let (re, im) = (values(n, seed, lo, hi), values(n, seed + 1000, lo, hi));
    re.into_iter().zip(im).map(|(re, im)| C32::new(re, im)).collect()
}

pub fn run(h: &mut Harness) {
    h.guarded("fusion", |h| {
        let ctx = h.ctx.clone();
        let n = size(h, 1 << 18, 1 << 20, 1 << 20);
        let xh = values(n, 6, -1.5, 1.5);
        let yh = values(n, 7, 0.5, 1.5);
        let x = A::from_slice_on(&ctx, &xh, n).unwrap();
        let y = A::from_slice_on(&ctx, &yh, n).unwrap();
        let bytes = 8.0 * n as f64;

        // Long chains: one fused kernel regardless of length, until the node or depth limit cuts it.
        for len in [8usize, 32, 128, 256, 512] {
            let ops = unary_chain(len as u64, len);
            check(&format!("fusion/chain{len}"), &chain_dev(&ops, &x).to_vec().unwrap(), &chain_host(&ops, &xh), 2e-3, 2e-3);
            h.case(&format!("fusion/chain{len}"), &format!("{len}op n={n}"), Work::Bytes(bytes), &mut || {
                force(&chain_dev(&ops, &x));
            });
        }
        // The same chain with a kernel per operation: what fusion saves (bytes are those of the fused kernel).
        {
            let ops = unary_chain(32, 32);
            let eager = fusion_limit(1, || chain_dev(&ops, &x));
            check("fusion/chain32_eager", &eager.to_vec().unwrap(), &chain_host(&ops, &xh), 2e-3, 2e-3);
            h.case("fusion/chain32_eager", &format!("32op n={n}"), Work::Bytes(bytes), &mut || {
                force(&fusion_limit(1, || chain_dev(&ops, &x)));
            });
        }

        // The 24-operation expression in Df32 arithmetic (each operation is ~10-20 float operations).
        {
            let want: Vec<f32> = xh.iter().map(|&v| heavy_host(f64::from(v).abs()) as f32).collect();
            let xa = x.abs();
            check("fusion/heavy_f32", &heavy(&xa).to_vec().unwrap(), &want, 1e-4, 1e-5);
            h.case("fusion/heavy_f32", &format!("24op f32 n={n}"), Work::Bytes(bytes), &mut || {
                force(&heavy(&xa));
            });
            check("fusion/df32_heavy", &heavy_df32(&xa).to_vec().unwrap(), &want, 1e-5, 1e-6);
            h.case("fusion/df32_heavy", &format!("24op df32 n={n}"), Work::Bytes(bytes), &mut || {
                force(&heavy_df32(&xa));
            });
        }

        // A fused chain of complex arithmetic on `C32` arrays: products, a conjugate, a quotient and a modulus in
        // one kernel that reads two complex arrays and writes one real one.
        {
            let nc = size(h, 1 << 20, 1 << 22, 1 << 22);
            let (zh, wh) = (complex_values(nc, 8, -1.5, 1.5), complex_values(nc, 9, 0.5, 1.5));
            let z = Array::<C32>::from_slice_on(&ctx, &zh, nc).unwrap();
            let w = Array::<C32>::from_slice_on(&ctx, &wh, nc).unwrap();
            let chain = |z: &Array<C32>, w: &Array<C32>| ((z * w + z).conj() / (w + 1.0f32)).abs();
            let want: Vec<f32> = zh
                .iter()
                .zip(&wh)
                .map(|(z, w)| {
                    let (zr, zi, wr, wi) = (f64::from(z.re), f64::from(z.im), f64::from(w.re), f64::from(w.im));
                    let (pr, pi) = (zr * wr - zi * wi + zr, zr * wi + zi * wr + zi);
                    let (nr, ni) = (pr, -pi);
                    let (dr, di) = (wr + 1.0, wi);
                    let d = dr * dr + di * di;
                    (((nr * dr + ni * di) / d).hypot((ni * dr - nr * di) / d)) as f32
                })
                .collect();
            check("fusion/complex_chain", &chain(&z, &w).to_vec().unwrap(), &want, 2e-5, 1e-6);
            h.case("fusion/complex_chain", &format!("c32 z*w+z conj / (w+1) abs n={nc}"), Work::Bytes(20.0 * nc as f64), &mut || {
                force(&chain(&z, &w));
            });
        }

        // Many scalars (a uniform block of 192 values) that change from call to call: one kernel.
        let mut round = 0u32;
        let scalars = |round: u32| -> Vec<f32> {
            let mut rng = Rng(u64::from(round) * 7919 + 17);
            (0..64).flat_map(|_| [rng.range(0.9, 1.1), rng.range(-0.1, 0.1), rng.range(2.0, 4.0)]).collect()
        };
        {
            let c = scalars(0);
            check("fusion/wide_scalars", &scalars_dev(&x, &c).to_vec().unwrap(), &scalars_host(&xh, &c), 2e-3, 2e-3);
        }
        h.case("fusion/wide_scalars", &format!("192 scalars n={n}"), Work::Bytes(bytes), &mut || {
            round += 1;
            let c = scalars(round);
            force(&scalars_dev(&x, &c));
        });

        // Many inputs: more distinct buffers than a kernel can bind.
        let inputs = 24;
        let ins: Vec<A> = (0..inputs).map(|k| A::from_slice_on(&ctx, &values(n, 20 + k as u64, 0.0, 1.0), n).unwrap()).collect();
        let wide = |ins: &[A]| -> A {
            let mut acc = &ins[0] * 0.01;
            for (k, a) in ins.iter().enumerate().skip(1) {
                acc = acc + a * (0.01 * (k + 1) as f32);
            }
            acc
        };
        {
            let hosts: Vec<Vec<f32>> = ins.iter().map(|a| a.to_vec().unwrap()).collect();
            let want: Vec<f32> = (0..n).map(|i| hosts.iter().enumerate().map(|(k, v)| f64::from(v[i]) * 0.01 * (k + 1) as f64).sum::<f64>() as f32).collect();
            check("fusion/wide_inputs", &wide(&ins).to_vec().unwrap(), &want, 1e-4, 1e-5);
        }
        h.case("fusion/wide_inputs", &format!("{inputs} inputs n={n}"), Work::Bytes(4.0 * n as f64 * (inputs + 1) as f64), &mut || {
            force(&wide(&ins));
        });
        drop(ins);

        // Many outputs sharing inputs and a common subexpression: eval_all computes them together.
        let nm = n / 4;
        let (xm, ym) = (x.slice(0, ..nm), y.slice(0, ..nm));
        for outs in [16usize, 64] {
            let build = |xm: &A, ym: &A| -> Vec<A> {
                let base = xm * ym;
                (0..outs).map(|k| &base * (0.5 + 0.01 * k as f32) + xm * 0.1).collect()
            };
            {
                let arrays = build(&xm, &ym);
                eval_all(&arrays.iter().collect::<Vec<_>>()).unwrap();
                for k in [0, outs / 2, outs - 1] {
                    let want: Vec<f32> = (0..nm).map(|i| (f64::from(xh[i]) * f64::from(yh[i]) * (0.5 + 0.01 * k as f64) + f64::from(xh[i]) * 0.1) as f32).collect();
                    check(&format!("fusion/outputs{outs}"), &arrays[k].to_vec().unwrap(), &want, 1e-4, 1e-4);
                }
            }
            h.case(&format!("fusion/outputs{outs}"), &format!("{outs} outs n={nm}"), Work::Bytes(4.0 * nm as f64 * (2 + outs) as f64), &mut || {
                let arrays = build(&xm, &ym);
                eval_all(&arrays.iter().collect::<Vec<_>>()).unwrap();
                ctx.sync().unwrap();
            });
        }

        // A DAG: shared subexpressions must be computed once and emitted once.
        let levels = 48;
        check("fusion/dag", &dag_dev(&x, &y, levels).to_vec().unwrap(), &dag_host(&xh, &yh, levels), 2e-3, 2e-3);
        h.case("fusion/dag", &format!("{levels} levels n={n}"), Work::Bytes(12.0 * n as f64), &mut || {
            force(&dag_dev(&x, &y, levels));
        });

        // Views inside a fused kernel.
        let side = size(h, 512, 1024, 1024);
        let uh = values(side * side, 8, 0.2, 1.0);
        let u = A::from_slice_on(&ctx, &uh, [side, side]).unwrap();
        let cells = (side * side) as f64;
        let pad_step = |u: &A| -> A {
            let e = heavy(u);
            u + lap_edge(&e, side, side) * 0.05
        };
        {
            let e: Vec<f64> = uh.iter().map(|&v| heavy_host(f64::from(v))).collect();
            let lap = lap_edge_host(&e, side, side);
            let want: Vec<f32> = uh.iter().zip(&lap).map(|(&u, l)| (f64::from(u) + 0.05 * l) as f32).collect();
            check("fusion/stencil_pad2d", &pad_step(&u).to_vec().unwrap(), &want, 2e-3, 2e-3);
        }
        h.case("fusion/stencil_pad2d", &format!("24op+pad {side}x{side}"), Work::Bytes(8.0 * cells), &mut || {
            force(&pad_step(&u));
        });
        let shift_step = |u: &A| -> A { u + lap_wrap(u) * 0.1 };
        {
            let lap = lap_wrap_host(&to_f64(&uh), side, side);
            let want: Vec<f32> = uh.iter().zip(&lap).map(|(&u, l)| (f64::from(u) + 0.1 * l) as f32).collect();
            check("fusion/stencil_shift2d", &shift_step(&u).to_vec().unwrap(), &want, 2e-3, 2e-3);
        }
        h.case("fusion/stencil_shift2d", &format!("shift {side}x{side}"), Work::Bytes(8.0 * cells), &mut || {
            force(&shift_step(&u));
        });

        let m = size(h, 1024, 1024, 2048);
        let (ah, bh) = (values(m * m, 9, 0.0, 1.0), values(m * m, 10, 0.0, 1.0));
        let (a, b) = (A::from_slice_on(&ctx, &ah, [m, m]).unwrap(), A::from_slice_on(&ctx, &bh, [m, m]).unwrap());
        {
            let want: Vec<f32> = (0..m * m).map(|i| ((f64::from(ah[(i % m) * m + i / m]) + f64::from(bh[i])) * 0.5) as f32).collect();
            check("fusion/transpose", &((a.transpose() + &b) * 0.5).to_vec().unwrap(), &want, 1e-5, 1e-5);
        }
        h.case("fusion/transpose", &format!("(a^T+b)*0.5 {m}x{m}"), Work::Bytes(12.0 * (m * m) as f64), &mut || {
            force(&((a.transpose() + &b) * 0.5));
        });
        drop((a, b));

        // Selects: branches on a value computed in the same kernel.
        let sel = |x: &A| -> A {
            let mut a = x.clone();
            for k in 0..8 {
                let thr = -0.8 + 0.2 * k as f32;
                a = select(&a.gt(thr), &a * 0.9, &a + 0.1);
            }
            a
        };
        {
            let want: Vec<f32> = xh
                .iter()
                .map(|&v| {
                    let mut a = f64::from(v);
                    for k in 0..8 {
                        let thr = f64::from(-0.8 + 0.2 * k as f32);
                        a = if a > thr { a * 0.9 } else { a + 0.1 };
                    }
                    a as f32
                })
                .collect();
            check("fusion/select8", &sel(&x).to_vec().unwrap(), &want, 1e-3, 1e-3);
        }
        h.case("fusion/select8", &format!("8 nested n={n}"), Work::Bytes(bytes), &mut || {
            force(&sel(&x));
        });
    });
}
