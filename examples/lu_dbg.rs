use luma::*;
fn main() {
    let ctx = Context::new().unwrap();
    for &n in &[64usize, 100, 300, 700, 1100] {
        let mut s = 777u64;
        let a: Vec<f32> = (0..n * n).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 }).collect();
        let arr = Array::from_slice_on(&ctx, &a, [n, n]).unwrap();
        let f = arr.lu();
        let lu = f.lu.to_vec();
        let perm = f.perm.to_vec();
        match (lu, perm) {
            (Ok(lu), Ok(perm)) => {
                let mut err = 0.0f64; let mut nrm = 0.0f64;
                for i in 0..n { for j in 0..n {
                    let mut acc = 0.0f64;
                    for k in 0..=i.min(j) {
                        let l = if k == i { 1.0 } else { lu[i * n + k] as f64 };
                        acc += l * lu[k * n + j] as f64;
                    }
                    let pa = a[perm[i] as usize * n + j] as f64;
                    err += (acc - pa).powi(2); nrm += pa * pa;
                } }
                println!("n {n}: rel {:e}", (err / nrm).sqrt());
            }
            (a, b) => println!("n {n}: error {:?} {:?}", a.err(), b.err()),
        }
    }
}
