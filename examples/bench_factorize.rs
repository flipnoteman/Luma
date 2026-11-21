//! `factorize` of u32 keys: counting path (narrow range) against the radix sort (wide range).
//! Columns: n, distinct values, key range, median microseconds per call (evaluated, synchronized).

use luma::*;
use std::time::Instant;

fn keys(n: usize, distinct: u32, stride: u32) -> Vec<u32> {
    let mut s = 0x9E3779B97F4A7C15u64 ^ n as u64 ^ distinct as u64;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as u32 % distinct).wrapping_mul(stride)
        })
        .collect()
}

fn main() {
    let quick = std::env::args().any(|a| a == "--quick");
    println!("{:>10} {:>9} {:>12} {:>10}", "n", "distinct", "stride", "us");
    for n in [10_000usize, 1_000_000, 10_000_000] {
        if quick && n > 1_000_000 {
            continue;
        }
        for (distinct, stride) in [(10u32, 1u32), (1000, 1), (65536, 1), (1000, 1 << 20), (10, 7)] {
            let x = Array::from_slice(&keys(n, distinct, stride), n).unwrap();
            x.eval().unwrap();
            let run = || {
                let (u, c) = x.factorize();
                u.eval().unwrap();
                c.eval().unwrap();
            };
            for _ in 0..3 {
                run();
            }
            let reps = if n >= 10_000_000 { 15 } else { 60 };
            let mut t: Vec<f64> = (0..reps)
                .map(|_| {
                    let s = Instant::now();
                    run();
                    s.elapsed().as_secs_f64() * 1e6
                })
                .collect();
            t.sort_by(f64::total_cmp);
            println!("{n:>10} {distinct:>9} {stride:>12} {:>10.1}", t[reps / 2]);
        }
    }
}
