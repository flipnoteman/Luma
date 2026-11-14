//! `put_add` / `put_min` / `put_max`, `bincount` and `histogram` against host computations, on the
//! privatised (workgroup memory) and the global path.

use luma::{bincount, Array, Context};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        self.next() as f32 / (1u32 << 31) as f32
    }
}

fn codes(rng: &mut Rng, n: usize, bins: usize, stray: bool) -> Vec<u32> {
    // With `stray`, one in 16 codes is out of range (and must be skipped).
    (0..n).map(|_| if stray && rng.next() % 16 == 0 { (bins as u32).wrapping_add(rng.next() % 1000) } else { rng.next() % bins as u32 }).collect()
}

fn host_count(codes: &[u32], bins: usize) -> Vec<u32> {
    let mut out = vec![0u32; bins];
    for &c in codes {
        if (c as usize) < bins {
            out[c as usize] += 1;
        }
    }
    out
}

#[test]
fn bincount_matches_the_host_count() {
    let ctx = Context::new().unwrap();
    let mut rng = Rng(1);
    for bins in [16usize, 1000, 4096, 1 << 20] {
        let host = codes(&mut rng, 1 << 20, bins, true);
        let indices = Array::from_slice_on(&ctx, &host, host.len()).unwrap();
        assert_eq!(bincount(&indices, bins).to_vec().unwrap(), host_count(&host, bins), "bins {bins}");
    }
}

#[test]
fn bincount_of_a_pending_expression() {
    let ctx = Context::new().unwrap();
    let host: Vec<u32> = (0..100_000).collect();
    let a = Array::from_slice_on(&ctx, &host, host.len()).unwrap();
    let bins = 50;
    let got = bincount(&(&a % 50u32), bins).to_vec().unwrap();
    assert_eq!(got, vec![2000u32; 50]);
}

#[test]
fn put_add_f32_matches_f64_sums() {
    let ctx = Context::new().unwrap();
    let mut rng = Rng(2);
    for bins in [1usize << 16, 64] {
        let n = 1 << 20;
        let host = codes(&mut rng, n, bins, true);
        let vals: Vec<f32> = (0..n).map(|_| rng.unit() - 0.5).collect();
        let mut sums = vec![0f64; bins];
        let mut mags = vec![0f64; bins];
        for (&c, &v) in host.iter().zip(&vals) {
            if (c as usize) < bins {
                sums[c as usize] += f64::from(v);
                mags[c as usize] += f64::from(v.abs());
            }
        }
        let indices = Array::from_slice_on(&ctx, &host, n).unwrap();
        let values = Array::from_slice_on(&ctx, &vals, n).unwrap();
        let mut target = Array::<f32>::zeros_on(&ctx, bins).unwrap();
        target.put_add(&indices, &values).unwrap();
        for (k, got) in target.to_vec().unwrap().into_iter().enumerate() {
            assert!((f64::from(got) - sums[k]).abs() <= 1e-5 * mags[k].max(1e-30), "bin {k} of {bins}: {got} vs {}", sums[k]);
        }
    }
}

#[test]
fn duplicates_count_their_multiplicity() {
    let ctx = Context::new().unwrap();
    let mut rng = Rng(3);
    let n = 1 << 18;
    let host = codes(&mut rng, n, 1000, false);
    let indices = Array::from_slice_on(&ctx, &host, n).unwrap();
    let mut target = Array::<u32>::zeros_on(&ctx, 1000).unwrap();
    target.put_add(&indices, 1u32).unwrap();
    assert_eq!(target.to_vec().unwrap(), host_count(&host, 1000));
}

#[test]
fn put_add_i32_with_negative_values() {
    let ctx = Context::new().unwrap();
    let mut rng = Rng(4);
    for bins in [8usize, 1 << 18] {
        let n = 1 << 17;
        let host = codes(&mut rng, n, bins, true);
        let vals: Vec<i32> = (0..n).map(|_| rng.next() as i32 % 1000).collect();
        let mut want = vec![0i32; bins];
        for (&c, &v) in host.iter().zip(&vals) {
            if (c as usize) < bins {
                want[c as usize] = want[c as usize].wrapping_add(v);
            }
        }
        let indices = Array::from_slice_on(&ctx, &host, n).unwrap();
        let values = Array::from_slice_on(&ctx, &vals, n).unwrap();
        let mut target = Array::<i32>::zeros_on(&ctx, bins).unwrap();
        target.put_add(&indices, &values).unwrap();
        assert_eq!(target.to_vec().unwrap(), want, "bins {bins}");
    }
}

macro_rules! min_max_test {
    ($name:ident, $t:ty, $make:expr) => {
        #[test]
        fn $name() {
            let ctx = Context::new().unwrap();
            let mut rng = Rng(5);
            let make: fn(&mut Rng) -> $t = $make;
            for bins in [4usize, 100, 1 << 17] {
                let n = 1 << 17;
                let host = codes(&mut rng, n, bins, true);
                let vals: Vec<$t> = (0..n).map(|_| make(&mut rng)).collect();
                let start: Vec<$t> = (0..bins).map(|_| make(&mut rng)).collect();
                let (mut lo, mut hi) = (start.clone(), start.clone());
                for (&c, &v) in host.iter().zip(&vals) {
                    if (c as usize) < bins {
                        lo[c as usize] = lo[c as usize].min(v);
                        hi[c as usize] = hi[c as usize].max(v);
                    }
                }
                let indices = Array::from_slice_on(&ctx, &host, n).unwrap();
                let values = Array::from_slice_on(&ctx, &vals, n).unwrap();
                let mut min = Array::from_slice_on(&ctx, &start, bins).unwrap();
                let mut max = Array::from_slice_on(&ctx, &start, bins).unwrap();
                min.put_min(&indices, &values).unwrap();
                max.put_max(&indices, &values).unwrap();
                assert_eq!(min.to_vec().unwrap(), lo, "min, bins {bins}");
                assert_eq!(max.to_vec().unwrap(), hi, "max, bins {bins}");
            }
        }
    };
}

min_max_test!(put_min_max_u32, u32, |rng| rng.next());
min_max_test!(put_min_max_i32, i32, |rng| rng.next() as i32 - (1 << 30));
min_max_test!(put_min_max_f32, f32, |rng| rng.unit() * 200.0 - 100.0);

#[test]
fn private_and_global_paths_agree_around_the_threshold() {
    let ctx = Context::new().unwrap();
    let mut rng = Rng(6);
    let bins = 4096;
    for n in [16 * bins - 1, 16 * bins, 16 * bins + 1] {
        let host = codes(&mut rng, n, bins, false);
        let indices = Array::from_slice_on(&ctx, &host, n).unwrap();
        assert_eq!(bincount(&indices, bins).to_vec().unwrap(), host_count(&host, bins), "count {n}");
    }
}

#[test]
fn put_add_adds_to_existing_contents() {
    let ctx = Context::new().unwrap();
    for bins in [16usize, 1 << 16] {
        let n = 1 << 20;
        let host: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(2654435761) % bins as u32).collect();
        let start: Vec<u32> = (0..bins as u32).map(|i| i * 3).collect();
        let mut want = start.clone();
        for &c in &host {
            want[c as usize] += 7;
        }
        let indices = Array::from_slice_on(&ctx, &host, n).unwrap();
        let mut target = Array::from_slice_on(&ctx, &start, bins).unwrap();
        target.put_add(&indices, 7u32).unwrap();
        assert_eq!(target.to_vec().unwrap(), want, "bins {bins}");
    }
}

#[test]
fn put_add_reading_its_own_target() {
    let ctx = Context::new().unwrap();
    let start: Vec<u32> = (1..=8).collect();
    let mut target = Array::from_slice_on(&ctx, &start, 8).unwrap();
    let indices = Array::from_slice_on(&ctx, &[0u32, 0, 7, 3], 4).unwrap();
    let values = target.slice(0, 0..4);
    let values = &values * 10u32;
    target.put_add(&indices, &values).unwrap();
    assert_eq!(target.to_vec().unwrap(), vec![1 + 10 + 20, 2, 3, 4 + 40, 5, 6, 7, 8 + 30]);
}

#[test]
fn histogram_matches_the_host() {
    let ctx = Context::new().unwrap();
    let mut rng = Rng(7);
    let n = 1 << 20;
    let mut vals: Vec<f32> = (0..n).map(|_| rng.unit() * 14.0 - 2.0).collect();
    vals[0] = f32::NAN;
    vals[1] = -1e9;
    vals[2] = 1e9;
    vals[3] = 10.0;
    let x = Array::from_slice_on(&ctx, &vals, n).unwrap();
    for bins in [256usize, 4096, 7] {
        let (lo, hi) = (0.0f32, 10.0f32);
        let scale = bins as f32 / (hi - lo);
        let mut want = vec![0u32; bins];
        for &v in &vals {
            let k = if v.is_nan() { 0 } else { (((v - lo) * scale).max(0.0) as u32).min(bins as u32 - 1) };
            want[k as usize] += 1;
        }
        assert_eq!(x.histogram(bins, lo, hi).to_vec().unwrap(), want, "bins {bins}");
    }
    assert!(x.histogram(0, 0.0, 1.0).to_vec().is_err());
    assert!(x.histogram(4, 1.0, 1.0).to_vec().is_err());
}

#[test]
fn histogram_is_one_scatter_kernel_and_a_fill() {
    let ctx = Context::new().unwrap();
    if !ctx.profiling_supported() {
        return;
    }
    let mut rng = Rng(8);
    let vals: Vec<f32> = (0..1 << 20).map(|_| rng.unit()).collect();
    let x = Array::from_slice_on(&ctx, &vals, vals.len()).unwrap();
    let (counts, profile) = ctx.profile(|| x.histogram(256, 0.0, 1.0).to_vec().unwrap()).unwrap();
    assert_eq!(counts.iter().sum::<u32>() as usize, vals.len());
    assert!(profile.launches() <= 2, "{} launches", profile.launches());
}
