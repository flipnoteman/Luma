use luma::*;
fn main() {
    let ctx = Context::new().unwrap();
    let w: usize = std::env::var("W").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    for &m in &[256usize, 512, 1024, 2048, 4096] {
        let mut s = 12345u64;
        let a: Vec<f32> = (0..m * w).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 }).collect();
        let arr = Array::from_slice_on(&ctx, &a, [w, m]).unwrap();
        for _ in 0..3 { let f = arr.lu(); f.lu.to_vec().unwrap(); }
        let reps = 20;
        let (_, p) = ctx.profile(|| { for _ in 0..reps { let f = arr.lu(); f.lu.cache(); let _ = f.perm.cache(); } ctx.sync().unwrap(); }).unwrap();
        print!("m={m:5} w={w}:");
        for k in &p.kernels { print!("  {} {:.1}us", k.label, k.total.as_secs_f64() * 1e6 / k.launches as f64); }
        println!();
    }
}
