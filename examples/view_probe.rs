use luma::*;
fn blk(a: &Array<f32>, r: std::ops::Range<usize>, c: std::ops::Range<usize>) -> Array<f32> { a.slice(1, r).slice(0, c) }
fn main() {
    let ctx = Context::new().unwrap();
    let n = 2048usize;
    let x = Array::<f32>::from_slice_on(&ctx, &vec![0.5f32; n * n], [n, n]).unwrap();
    x.eval().unwrap();
    ctx.sync().unwrap();
    let show = |name: &str, f: &dyn Fn() -> Array<f32>| {
        let (_, p) = ctx.profile(|| { let r = f(); r.eval().unwrap(); ctx.sync().unwrap(); }).unwrap();
        println!("{name}:");
        for k in &p.kernels { println!("    {:7.1} us x{} {}", k.total.as_secs_f64() * 1e6, k.launches, k.label.chars().take(44).collect::<String>()); }
    };
    show("matmul strided cols view x strided view", &|| blk(&x, 1024..2048, 0..512).matmul_f32(&blk(&x, 0..512, 1024..2048)));
    show("matmul row-slice (contiguous) x row-slice", &|| blk(&x, 1024..2048, 0..2048).matmul_f32(&blk(&x, 0..2048, 0..1024)));
    show("matmul full compact", &|| x.matmul_f32(&x));
    show("matmul 32-wide strided", &|| blk(&x, 1024..2048, 0..32).matmul_f32(&blk(&x, 0..32, 1024..2048)));
    show("matmul 512 strided L x compact", &|| { let c = blk(&x, 0..512, 1024..2048).cache(); blk(&x, 1024..2048, 0..512).matmul_f32(&c) });
}
