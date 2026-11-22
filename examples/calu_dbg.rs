use luma::*;
fn main() {
    let ctx = Context::new().unwrap();
    let (m, w) = (std::env::args().nth(1).map_or(100, |s| s.parse().unwrap()), std::env::args().nth(2).map_or(32, |s| s.parse().unwrap()));
    let mut s = 12345u64;
    let a: Vec<f32> = (0..m * w).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 }).collect();
    let arr = Array::from_slice_on(&ctx, &a, [w, m]).unwrap();
    let f = arr.lu();
    let lu = f.lu.to_vec().unwrap();
    let perm = f.perm.to_vec().unwrap();
    println!("perm {:?}", &perm[..m.min(80)]);
    let mut seen = vec![false; m];
    for &p in &perm { assert!(!std::mem::replace(&mut seen[p as usize], true), "dup"); }
    // residual
    let mut err = 0.0f64; let mut nrm = 0.0f64; let mut lmax = 0.0f64;
    for i in 0..m { for j in 0..w {
        let mut acc = 0.0f64;
        for k in 0..=i.min(j) {
            let l = if k == i { 1.0 } else { lu[i * w + k] as f64 };
            if k < i { lmax = lmax.max(l.abs()); }
            let u = if k <= j { lu[k * w + j] as f64 } else { 0.0 };
            if k < w { acc += l * u; }
        }
        let pa = a[perm[i] as usize * w + j] as f64;
        err += (acc - pa).powi(2); nrm += pa * pa;
    } }
    println!("m {m} w {w}: rel {:e} lmax {lmax}", (err / nrm).sqrt());
}
