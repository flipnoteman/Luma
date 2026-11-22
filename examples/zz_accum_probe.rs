use luma::*;

fn show(ctx: &Context, title: &str, f: &mut dyn FnMut()) {
    f();
    ctx.sync().unwrap();
    let ((), p) = ctx.profile(&mut *f).unwrap();
    println!("-- {title}: {} launches, {:.1} us", p.launches(), p.total().as_secs_f64() * 1e6);
    for k in &p.kernels {
        println!("     {:>8.1} us x{:<3} {}", k.total.as_secs_f64() * 1e6, k.launches, k.label.chars().take(90).collect::<String>());
    }
}

fn main() {
    let ctx = default_context().unwrap();
    let data: Vec<f32> = (0..4096 * 256).map(|i| (i % 977) as f32 * 0.01).collect();
    let x = Array::from_slice_on(&ctx, &data, [4096, 256]).unwrap();
    let k = Array::<f32>::full_on(&ctx, [4096, 1], 4.0).unwrap();
    k.eval().unwrap();
    let mk = || {
        let d = &x - &k;
        let [s1, s2] = d.reduce_axis(1, [Reduction::Sum, Reduction::SumOfSquares]);
        let d2 = &d * &d;
        let s3 = (&d2 * &d).sum_axis(1);
        let [s4] = d2.reduce_axis(1, [Reduction::SumOfSquares]);
        (s1, s2, s3, s4)
    };
    show(&ctx, "4 reductions, eval_all f32", &mut || {
        let (a, b, c, d) = mk();
        eval_all(&[&a, &b, &c, &d]).unwrap();
    });
    show(&ctx, "4 reductions, Df32 casts + eval_all", &mut || {
        let (a, b, c, d) = mk();
        let w: Vec<Array<Df32>> = [a, b, c, d].iter().map(|r| r.to_df32()).collect();
        let sum = &(&w[0] + &w[1]) + &(&w[2] + &w[3]);
        sum.eval().unwrap();
    });
    show(&ctx, "2 reductions only (sum+sumsq)", &mut || {
        let d = &x - &k;
        let [s1, s2] = d.reduce_axis(1, [Reduction::Sum, Reduction::SumOfSquares]);
        eval_all(&[&s1, &s2]).unwrap();
    });
    show(&ctx, "min+max axis reduce", &mut || {
        let [lo, hi] = x.reduce_axis(1, [Reduction::Min, Reduction::Max]);
        eval_all(&[&lo, &hi]).unwrap();
    });
    show(&ctx, "sum,sumsq + min,max", &mut || {
        let d = &x - &k;
        let [s1, s2] = d.reduce_axis(1, [Reduction::Sum, Reduction::SumOfSquares]);
        let [lo, hi] = x.reduce_axis(1, [Reduction::Min, Reduction::Max]);
        eval_all(&[&s1, &s2, &lo, &hi]).unwrap();
    });
}
