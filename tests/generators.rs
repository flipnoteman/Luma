//! Generator arrays (`uniform`, `normal`, `random_bits`, `range`, `linspace`) match a CPU reference
//! bit for bit wherever they are computed: alone, fused, inside reductions, chunked or shifted.

mod common;

use common::assert_close;
use luma::*;

const SEED: u64 = 0x1234_5678_9abc_def0;

fn threefry([c0, c1]: [u32; 2], seed: u64) -> [u32; 2] {
    const ROTATIONS: [u32; 8] = [13, 15, 26, 6, 17, 29, 16, 24];
    let (k0, k1) = (seed as u32, (seed >> 32) as u32);
    let keys = [k0, k1, 0x1BD1_1BDA ^ k0 ^ k1];
    let (mut x0, mut x1) = (c0.wrapping_add(k0), c1.wrapping_add(k1));
    for round in 0..20 {
        x0 = x0.wrapping_add(x1);
        x1 = x1.rotate_left(ROTATIONS[round % 8]) ^ x0;
        if round % 4 == 3 {
            let s = round / 4 + 1;
            x0 = x0.wrapping_add(keys[s % 3]);
            x1 = x1.wrapping_add(keys[(s + 1) % 3]).wrapping_add(s as u32);
        }
    }
    [x0, x1]
}

fn uniform_cpu(n: usize, seed: u64) -> Vec<f32> {
    (0..n).map(|i| (threefry([i as u32, 0], seed)[0] >> 8) as f32 * (1.0 / 16_777_216.0)).collect()
}

#[test]
fn the_reference_matches_random123() {
    assert_eq!(threefry([0, 0], 0), [0x6b20_0159, 0x99ba_4efe]);
    assert_eq!(threefry([u32::MAX; 2], u64::MAX), [0x1cb9_96fc, 0xbb00_2be7]);
    assert_eq!(threefry([0x243f_6a88, 0x85a3_08d3], 0x0370_7344_1319_8a2e), [0xc492_3a9c, 0x483d_f7a0]);
}

#[test]
fn random_bits_and_uniform_match_the_cpu_exactly() {
    let n = 10_000;
    let bits = Array::<u32>::random_bits(n, SEED).unwrap().to_vec().unwrap();
    let want: Vec<u32> = (0..n).map(|i| threefry([i as u32, 0], SEED)[0]).collect();
    assert_eq!(bits, want);
    assert_eq!(Array::<f32>::uniform(n, SEED).unwrap().to_vec().unwrap(), uniform_cpu(n, SEED));
    assert_ne!(Array::<f32>::uniform(n, SEED + 1).unwrap().to_vec().unwrap(), uniform_cpu(n, SEED));
}

#[test]
fn uniform_and_normal_have_the_expected_moments() {
    let n = 1 << 20;
    let u = Array::<f32>::uniform(n, 7).unwrap();
    let mean = u.mean().item().unwrap();
    let var = (&u * &u).mean().item().unwrap() - mean * mean;
    assert!((mean - 0.5).abs() < 2e-3, "uniform mean {mean}");
    assert!((var - 1.0 / 12.0).abs() < 2e-3, "uniform variance {var}");

    let z = Array::<f32>::normal(n, 7).unwrap();
    let mean = z.mean().item().unwrap();
    let var = (&z * &z).mean().item().unwrap() - mean * mean;
    assert!(mean.abs() < 5e-3, "normal mean {mean}");
    assert!((var - 1.0).abs() < 1e-2, "normal variance {var}");
    assert!(z.to_vec().unwrap().iter().all(|v| v.is_finite()));
}

#[test]
fn a_reduction_of_generators_launches_no_store_kernel() {
    let ctx = Context::new().unwrap();
    let n = 1 << 20;
    let (pi, profile) = ctx
        .profile(|| {
            let u = Array::<f32>::uniform_on(&ctx, n, 1).unwrap();
            let v = Array::<f32>::uniform_on(&ctx, n, 2).unwrap();
            let inside = (&u * &u + &v * &v).le(1.0);
            select(&inside, 4.0, 0.0).sum().item().unwrap() / n as f32
        })
        .unwrap();
    assert!((pi - std::f32::consts::PI).abs() < 1e-2, "pi estimate {pi}");
    for kernel in &profile.kernels {
        assert!(!kernel.label.starts_with("Fused(") && !kernel.label.starts_with("Fused["), "{}", kernel.label);
    }
}

#[test]
fn chunked_arrays_see_the_same_values() {
    let ctx = Context::with_options(ContextOptions::default().chunk_bytes(1 << 12)).unwrap();
    let n = 10_000;
    let u = Array::<f32>::uniform_on(&ctx, n, SEED).unwrap();
    let want = uniform_cpu(n, SEED);
    assert_eq!((&u * 1.0).to_vec().unwrap(), want);
    let sum: f64 = want.iter().map(|&v| f64::from(v)).sum();
    assert!((f64::from(u.sum().item().unwrap()) - sum).abs() < sum * 1e-5);
    let ramp = Array::<f32>::range_on(&ctx, n, 0).unwrap() * 1.0;
    assert_eq!(ramp.to_vec().unwrap(), (0..n).map(|i| i as f32).collect::<Vec<_>>());
}

#[test]
fn shifted_generators_read_the_shifted_index() {
    let n = 1000;
    let want = uniform_cpu(n, SEED);
    let shifted = Array::<f32>::uniform(n, SEED).unwrap().shift([3]) + 0.0;
    let rotated: Vec<f32> = (0..n).map(|i| want[(i + n - 3) % n]).collect();
    assert_eq!(shifted.to_vec().unwrap(), rotated);
}

#[test]
fn range_and_linspace() {
    let shape = [5, 3];
    let x = Array::<f32>::range(shape, 0).unwrap().to_vec().unwrap();
    let y = Array::<u32>::range(shape, 1).unwrap().to_vec().unwrap();
    for i in 0..15 {
        assert_eq!(x[i], (i % 5) as f32);
        assert_eq!(y[i], (i / 5) as u32);
    }
    assert!(Array::<f32>::range(shape, 4).is_err());
    let line = Array::<f32>::linspace(0.0, 1.0, 5).unwrap().to_vec().unwrap();
    assert_close(&line, &[0.0, 0.25, 0.5, 0.75, 1.0], 1e-7, 1e-7);
    assert_eq!(Array::<f32>::linspace(2.0, 3.0, 1).unwrap().to_vec().unwrap(), vec![2.0]);
}

#[test]
fn generators_fuse_into_iterate_all() {
    let n = 4096;
    let noise = Array::<f32>::uniform(n, 3).unwrap();
    let zero = Array::<f32>::zeros(n).unwrap();
    let [acc] = iterate_all([&zero], 4, |[a]| [a + &noise]);
    let want: Vec<f32> = uniform_cpu(n, 3).iter().map(|v| v + v + v + v).collect();
    assert_close(&acc.to_vec().unwrap(), &want, 1e-6, 1e-6);
}
