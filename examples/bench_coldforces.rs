//! First-call cost of a force monitor on an empty kernel cache: `3 * masks` sums of the components of a wall pressure
//! (a gathered pending expression) joined by `concat_slice` and read back once, as a 3-D solver's `wall_forces`. The sums are
//! recorded in groups of `Tuning::reduce_multi` per kernel, and the groups are different programs, so each is compiled
//! the first time. Run with an empty `LUMA_CACHE_DIR` (and `CUDA_CACHE_DISABLE=1`) for a cold kernel cache.
//!
//! `cargo run --release --example bench_coldforces [-- masks]`

use luma::{concat_slice, Array};
use std::time::Instant;

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 40) as f32) / (1u64 << 24) as f32
        })
        .collect()
}

fn main() {
    let masks_n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(14);
    let ctx = luma::default_context().unwrap();
    let faces = 200_000usize;
    let cells = faces * 4;
    let state: Vec<Array<f32>> = (0..5).map(|k| Array::from_slice_on(&ctx, &values(cells, 500 + k), cells).unwrap()).collect();
    let ia: Vec<u32> = (0..faces).map(|i| ((i * 7) % cells) as u32).collect();
    let ib: Vec<u32> = (0..faces).map(|i| ((i * 7 + 1) % cells) as u32).collect();
    let (ia, ib) = (Array::from_slice_on(&ctx, &ia, faces).unwrap(), Array::from_slice_on(&ctx, &ib, faces).unwrap());
    let k = Array::from_slice_on(&ctx, &values(faces, 510), faces).unwrap();
    let weights: Vec<Array<f32>> = (0..3).map(|c| Array::from_slice_on(&ctx, &values(faces, 520 + c), faces).unwrap()).collect();
    let masks: Vec<Array<bool>> = (0..masks_n)
        .map(|c| Array::from_slice_on(&ctx, &(0..faces).map(|i| i % masks_n == c).collect::<Vec<_>>(), faces).unwrap())
        .collect();
    ctx.sync().unwrap();
    let run = || {
        let speed = &state[1] * &state[1] + &state[2] * &state[2] + &state[3] * &state[3];
        let pall = (&state[4] - &speed * &state[0] * 0.5f32) * 0.4f32;
        let (pa, pb) = (pall.take(&ia), pall.take(&ib));
        let pw = &pa + &k * (&pa - &pb);
        let d = pw - 0.7f32;
        let comps: Vec<Array<f32>> = weights.iter().map(|w| &d * w).collect();
        let mut sums = Vec::new();
        for mask in &masks {
            for comp in &comps {
                sums.push(comp.sum_where(mask).reshape([1]));
            }
        }
        let t0 = Instant::now();
        let out = concat_slice(0, &sums).to_vec().unwrap();
        (t0.elapsed(), out)
    };
    // Calls 1 to 12 back to back, then two more once the merged kernels have had time to compile in the background.
    let mut times = vec![];
    let mut reference: Vec<f32> = vec![];
    for call in 0..14 {
        let (t, out) = run();
        if call == 0 {
            reference = out;
        } else {
            assert!(out.iter().zip(&reference).all(|(x, y)| x.to_bits() == y.to_bits()), "unstable");
        }
        times.push(t.as_secs_f64() * 1e3);
        if call == 11 {
            std::thread::sleep(std::time::Duration::from_secs(4));
        }
    }
    let fmt = |ts: &[f64]| ts.iter().map(|t| format!("{t:.1}")).collect::<Vec<_>>().join(" ");
    println!("forces of {} sums, calls 1-12 and 13-14 after 4 s (ms): {}", reference.len(), fmt(&times));
}
