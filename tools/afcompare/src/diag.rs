//! `afcompare diag`: luma only. Profiles every product of the MLP step in isolation with the
//! operand kinds the step gives it (stored, lazy expression, transposed view), then the whole step.

use crate::cases::L;
use crate::lumax;
use crate::util::uniform;

/// Keeps the GPU busy for 400 ms so that the first products are not timed at idle clocks.
fn wake(ctx: &luma::Context) {
    let x = luma::Array::<f32>::linspace_on(ctx, 0.0, 1.0, 1 << 24).unwrap();
    let start = std::time::Instant::now();
    while start.elapsed() < std::time::Duration::from_millis(400) {
        let mut z = x.clone();
        for _ in 0..6 {
            z = (z * 1.0001f32 + 0.1f32).tan().cos();
        }
        z.eval().unwrap();
        ctx.sync().unwrap();
    }
}

fn show(ctx: &luma::Context, name: &str, mut f: impl FnMut()) {
    for _ in 0..5 {
        f();
        ctx.sync().unwrap();
    }
    let (_, p) = ctx.profile(|| {
        f();
        ctx.sync().unwrap();
    }).unwrap();
    println!("{name}: {} launches, {:.1} us GPU", p.launches(), p.total().as_secs_f64() * 1e6);
    for k in &p.kernels {
        let label = k.label.to_string();
        let tail: String = label.chars().rev().take(110).collect::<Vec<_>>().into_iter().rev().collect();
        let head: String = label.chars().take(24).collect();
        println!("    {:>7.1} us x{:<2} {head} ... {tail}", k.total.as_secs_f64() * 1e6, k.launches);
    }
}

pub fn run() {
    let ctx = lumax::open();
    wake(&ctx);
    let (b, d) = (256usize, [784usize, 512, 512, 10]);
    let arr = |n: usize, shape: [usize; 2], seed| lumax::arr(&ctx, &uniform(n, seed, -1.0, 1.0), shape);
    let x = arr(b * d[0], [d[0], b], 1);
    let w1 = arr(d[0] * d[1], [d[1], d[0]], 2);
    let w2 = arr(d[1] * d[2], [d[2], d[1]], 3);
    let w3 = arr(d[2] * d[3], [d[3], d[2]], 4);
    let z1 = arr(b * d[1], [d[1], b], 5);
    let z2 = arr(b * d[2], [d[2], b], 6);
    let dl = arr(b * d[3], [d[3], b], 7);
    let dz2 = arr(b * d[2], [d[2], b], 8);
    let dz1 = arr(b * d[1], [d[1], b], 9);
    for tf32 in [false, true] {
        let mm: fn(&L, &L) -> L = if tf32 { luma::matmul_tf32 } else { luma::matmul_f32 };
        println!("---- {}", if tf32 { "TF32" } else { "f32" });
        let ev = |r: L| r.eval().unwrap();
        show(&ctx, "fwd1  x.w1        stored/stored      M256 K784 N512", || ev(mm(&x, &w1)));
        show(&ctx, "fwd2  relu(z1).w2 lazy/stored        M256 K512 N512", || ev(mm(&z1.maximum(0.0f32), &w2)));
        show(&ctx, "fwd3  relu(z2).w3 lazy/stored        M256 K512 N10 ", || ev(mm(&z2.maximum(0.0f32), &w3)));
        show(&ctx, "dW3   relu(z2)T.dl lazyT/stored      M512 K256 N10 ", || ev(mm(&z2.maximum(0.0f32).transpose(), &dl)));
        show(&ctx, "dz2   dl.w3T      stored/T           M256 K10  N512", || ev(mm(&dl, &w3.transpose())));
        show(&ctx, "dW2   relu(z1)T.dz2 lazyT/stored     M512 K256 N512", || ev(mm(&z1.maximum(0.0f32).transpose(), &dz2)));
        show(&ctx, "dz1   dz2.w2T     stored/T           M256 K512 N512", || ev(mm(&dz2, &w2.transpose())));
        show(&ctx, "dW1   xT.dz1      T/stored           M784 K256 N512", || ev(mm(&x.transpose(), &dz1)));
    }
    println!("---- reductions / elementwise");
    let prod = arr(b * d[1], [d[1], b], 10);
    let logits = arr(b * d[3], [d[3], b], 11);
    let y = arr(b * d[3], [d[3], b], 12);
    show(&ctx, "sum_axis(1) of plain [512,256]", || { let r = prod.sum_axis(1); r.eval().unwrap(); });
    show(&ctx, "sum_axis(1) of select(z1>0, prod, 0)", || { let r = luma::select(&z1.gt(0.0f32), &prod, 0.0f32).sum_axis(1); r.eval().unwrap(); });
    show(&ctx, "sum_axis(1) of plain [10,256]", || { let r = dl.sum_axis(1); r.eval().unwrap(); });
    show(&ctx, "max_axis(0) of [10,256]", || { let r = logits.max_axis(0); r.eval().unwrap(); });
    show(&ctx, "sum_axis(0) of exp [10,256]", || { let r = logits.exp().sum_axis(0); r.eval().unwrap(); });
    show(&ctx, "sum() of logits*y", || { let r = (&logits * &y).sum(); r.eval().unwrap(); });
    show(&ctx, "loss-ish (m+ln s).sum()", || { let m = logits.max_axis(0); let s = logits.exp().sum_axis(0); let r = (&m + &s.ln()).sum(); r.eval().unwrap(); });
    show(&ctx, "update p - g*lr [512,784]", || { let r = &w1 - &(&w1 * 0.5f32); r.eval().unwrap(); });
    println!("---- softmax pieces");
    let mmo = arr(b * d[3], [d[3], b], 13);
    let b3 = arr(d[3], [d[3], 1], 14);
    show(&ctx, "s lazy: logits=mm+b3; m=max; s=sum exp(l-m)", || { let l = &mmo + &b3; let m = l.max_axis(0); let s = (&l - &m).exp().sum_axis(0); s.eval().unwrap(); });
    show(&ctx, "s with logits.cache()", || { let l = (&mmo + &b3).cache(); let m = l.max_axis(0); let s = (&l - &m).exp().sum_axis(0); s.eval().unwrap(); });
    show(&ctx, "s with m evaluated", || { let l = &mmo + &b3; let m = l.max_axis(0); m.eval().unwrap(); let s = (&l - &m).exp().sum_axis(0); s.eval().unwrap(); });
    show(&ctx, "s with logits and m evaluated", || { let l = (&mmo + &b3).cache(); let m = l.max_axis(0); m.eval().unwrap(); let s = (&l - &m).exp().sum_axis(0); s.eval().unwrap(); });
    show(&ctx, "loss+dl whole softmax part (lazy)", || {
        let l = &mmo + &b3;
        let m = l.max_axis(0);
        let e = (&l - &m).exp();
        let s = e.sum_axis(0);
        let loss = &((&m + &s.ln()).sum() - (&l * &y).sum()) * 0.0039f32;
        let dlv = &(&(&e / &s) - &y) * 0.0039f32;
        luma::eval_all(&[&loss, &dlv]).unwrap();
    });
    show(&ctx, "loss+dl with logits.cache()", || {
        let l = (&mmo + &b3).cache();
        let m = l.max_axis(0);
        let e = (&l - &m).exp();
        let s = e.sum_axis(0);
        let loss = &((&m + &s.ln()).sum() - (&l * &y).sum()) * 0.0039f32;
        let dlv = &(&(&e / &s) - &y) * 0.0039f32;
        luma::eval_all(&[&loss, &dlv]).unwrap();
    });
    show(&ctx, "select(z1>0, prod, 0) eval", || { let r = luma::select(&z1.gt(0.0f32), &prod, 0.0f32); r.eval().unwrap(); });
}

/// `afcompare sweep`: one line per product shape with the GPU time of `matmul_f32` and `matmul_tf32`
/// (stored operands, plus the transposed-operand variants of the MLP), for comparing planner settings.
pub fn sweep() {
    let ctx = lumax::open();
    wake(&ctx);
    let shapes: [(&str, usize, usize, usize); 14] = [
        ("M256 K1024 N1024", 256, 1024, 1024),
        ("M2048 K512 N128", 2048, 512, 128),
        ("M256 K784 N512", 256, 784, 512),
        ("M256 K512 N512", 256, 512, 512),
        ("M256 K512 N12", 256, 512, 12),
        ("M512 K256 N12", 512, 256, 12),
        ("M512 K256 N512", 512, 256, 512),
        ("M784 K256 N512", 784, 256, 512),
        ("M2048 K2048 N64", 2048, 2048, 64),
        ("M2048 K64 N2048", 2048, 64, 2048),
        ("M1024 K1024 N1024", 1024, 1024, 1024),
        ("M4096 K4096 N4096", 4096, 4096, 4096),
        ("M128 K4096 N4096", 128, 4096, 4096),
        ("M64 K8192 N64", 64, 8192, 64),
    ];
    let gpu = |f: &mut dyn FnMut()| {
        for _ in 0..8 {
            f();
            ctx.sync().unwrap();
        }
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let (_, p) = ctx.profile(|| {
                f();
                ctx.sync().unwrap();
            }).unwrap();
            best = best.min(p.total().as_secs_f64() * 1e6);
        }
        best
    };
    for (name, m, k, n) in shapes {
        let a = lumax::arr(&ctx, &uniform(m * k, 1, -1.0, 1.0), [k, m]);
        let b = lumax::arr(&ctx, &uniform(k * n, 2, -1.0, 1.0), [n, k]);
        let at = lumax::arr(&ctx, &uniform(m * k, 3, -1.0, 1.0), [m, k]);
        let f = gpu(&mut || luma::matmul_f32(&a, &b).eval().unwrap());
        let t = gpu(&mut || luma::matmul_tf32(&a, &b).eval().unwrap());
        let ft = gpu(&mut || luma::matmul_f32(&at.transpose(), &b).eval().unwrap());
        println!("{name:<18} f32 {f:>8.1} us   tf32 {t:>8.1} us   f32 with lhs^T {ft:>8.1} us");
    }
}
