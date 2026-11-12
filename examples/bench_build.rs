//! Host cost of building lazy expressions (nothing is evaluated): nanoseconds per operation of a mixed chain over four
//! 64x64 views, the form of a ghost-state expression in a solver.
//!
//! `cargo run --release --example bench_build [-- reps]`

use luma::Array;
use std::time::Instant;

type A = Array<f32>;

/// A chain of `ops` mixed steps over `xs`; returns the array and the number of operations created.
fn chain(xs: &[&A], seed: usize, steps: usize) -> (A, usize) {
    let mut acc = xs[seed % xs.len()] * 1.0;
    let mut made = 1;
    for k in 0..steps {
        let y = xs[(seed + k + 1) % xs.len()];
        acc = match k % 4 {
            0 => {
                made += 2;
                &acc * y + 0.1
            }
            1 => {
                made += 3;
                (&acc - y).abs() + 0.2
            }
            2 => {
                made += 2;
                &acc / (y + 1.5)
            }
            _ => {
                made += 2;
                &acc + y * 0.5
            }
        };
    }
    (acc, made)
}

fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let ctx = luma::default_context().unwrap();
    let data = |k: usize| Array::from_slice_on(&ctx, &(0..200 * 200).map(|i| 1.0 + 0.1 * ((i + k) as f32 * 0.37).sin()).collect::<Vec<_>>(), [200, 200]).unwrap();
    let xs: Vec<A> = (0..4).map(|k| data(k).slice(0, ..64).slice(1, ..64)).collect();
    let refs: Vec<&A> = xs.iter().collect();
    println!("{:>6} {:>8} {:>12} {:>10} {:>10}", "steps", "ops", "us per build", "ns per op", "drop ns/op");
    let sizes: Vec<usize> = std::env::args().skip(2).filter_map(|s| s.parse().ok()).collect();
    let sizes = if sizes.is_empty() { vec![50, 200, 450] } else { sizes };
    for steps in sizes {
        let (mut best, mut best_drop) = (f64::MAX, f64::MAX);
        let mut ops = 0;
        for _ in 0..3 {
            let mut dropping = 0.0;
            let t0 = Instant::now();
            for r in 0..reps {
                let (a, n) = chain(&refs, r, steps);
                ops = n;
                let t1 = Instant::now();
                drop(std::hint::black_box(a));
                dropping += t1.elapsed().as_secs_f64();
            }
            best = best.min((t0.elapsed().as_secs_f64() - dropping) / reps as f64);
            best_drop = best_drop.min(dropping / reps as f64);
        }
        println!("{steps:>6} {ops:>8} {:>12.1} {:>10.0} {:>10.0}", best * 1e6, best * 1e9 / ops as f64, best_drop * 1e9 / ops as f64);
    }
}
