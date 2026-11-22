use luma::*;
use std::time::Instant;

fn bench(ctx: &Context, title: &str, f: &mut dyn FnMut()) {
    for _ in 0..3 {
        f();
        ctx.sync().unwrap();
    }
    let mut times = Vec::new();
    for _ in 0..15 {
        let t = Instant::now();
        f();
        ctx.sync().unwrap();
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(f64::total_cmp);
    println!("{title:<52} median {:.3} ms  min {:.3} ms", times[7], times[0]);
}

fn main() {
    let ctx = default_context().unwrap();
    let (f, r, b) = (4096usize, 256usize, 64usize);
    let data: Vec<f32> = (0..f * r).map(|i| ((i * 7919) % 1000) as f32 * 0.001 + (i % f) as f32).collect();
    let batches: Vec<Array<f32>> = (0..8).map(|i| Array::from_slice_on(&ctx, &data.iter().map(|v| v + i as f32).collect::<Vec<_>>(), [f, r]).unwrap()).collect();
    let kdata: Vec<f32> = (0..f).map(|i| i as f32 + 0.25).collect();
    let k = Array::from_slice_on(&ctx, &kdata, [f, 1]).unwrap();
    let k = (&k * 1.0f32 + 0.5f32).round();
    k.eval().unwrap();
    let ones = Array::<f32>::full_on(&ctx, 1usize, 1.0).unwrap();
    ones.eval().unwrap();
    let run = |ctx: &Context, title: &str, build: &dyn Fn(&Array<f32>) -> Vec<Array<f32>>| {
        bench(ctx, title, &mut || {
            for i in 0..b {
                let outs = build(&batches[i % 8]);
                let refs: Vec<&Array<f32>> = outs.iter().collect();
                eval_all(&refs).unwrap();
            }
        });
    };
    run(&ctx, "scalar pivot: sum+sumsq", &|x| {
        let d = x - 3.0f32;
        d.reduce_axis(1, [Reduction::Sum, Reduction::SumOfSquares]).into()
    });
    run(&ctx, "A: d sum+sumsq", &|x| {
        let d = x - &k;
        d.reduce_axis(1, [Reduction::Sum, Reduction::SumOfSquares]).into()
    });
    run(&ctx, "A + S3", &|x| {
        let d = x - &k;
        let mut v: Vec<Array<f32>> = d.reduce_axis(1, [Reduction::Sum, Reduction::SumOfSquares]).into();
        v.push((&(&d * &d) * &d).sum_axis(1));
        v
    });
    run(&ctx, "A + S3 + S4", &|x| {
        let d = x - &k;
        let d2 = &d * &d;
        let mut v: Vec<Array<f32>> = d.reduce_axis(1, [Reduction::Sum, Reduction::SumOfSquares]).into();
        v.push((&d2 * &d).sum_axis(1));
        v.extend(d2.reduce_axis(1, [Reduction::SumOfSquares]));
        v
    });
    run(&ctx, "min,max of x (stored)", &|x| x.reduce_axis(1, [Reduction::Min, Reduction::Max]).into());
    run(&ctx, "A + min,max of x (stored)", &|x| {
        let d = x - &k;
        let mut v: Vec<Array<f32>> = d.reduce_axis(1, [Reduction::Sum, Reduction::SumOfSquares]).into();
        v.extend(x.reduce_axis(1, [Reduction::Min, Reduction::Max]));
        v
    });
    run(&ctx, "A + min,max of (x - 0.0)", &|x| {
        let d = x - &k;
        let mut v: Vec<Array<f32>> = d.reduce_axis(1, [Reduction::Sum, Reduction::SumOfSquares]).into();
        v.extend((x - 0.0f32).reduce_axis(1, [Reduction::Min, Reduction::Max]));
        v
    });
    run(&ctx, "A + min,max of (x * 1.0)", &|x| {
        let d = x - &k;
        let mut v: Vec<Array<f32>> = d.reduce_axis(1, [Reduction::Sum, Reduction::SumOfSquares]).into();
        v.extend((x * 1.0f32).reduce_axis(1, [Reduction::Min, Reduction::Max]));
        v
    });
    run(&ctx, "A + min,max of (x * ones[1])", &|x| {
        let d = x - &k;
        let mut v: Vec<Array<f32>> = d.reduce_axis(1, [Reduction::Sum, Reduction::SumOfSquares]).into();
        v.extend((x * &ones).reduce_axis(1, [Reduction::Min, Reduction::Max]));
        v
    });
    run(&ctx, "A + min,max of d", &|x| {
        let d = x - &k;
        let mut v: Vec<Array<f32>> = d.reduce_axis(1, [Reduction::Sum, Reduction::SumOfSquares]).into();
        v.extend(d.reduce_axis(1, [Reduction::Min, Reduction::Max]));
        v
    });
    run(&ctx, "A + S3 + S4 + min,max of x", &|x| {
        let d = x - &k;
        let d2 = &d * &d;
        let mut v: Vec<Array<f32>> = d.reduce_axis(1, [Reduction::Sum, Reduction::SumOfSquares]).into();
        v.push((&d2 * &d).sum_axis(1));
        v.extend(d2.reduce_axis(1, [Reduction::SumOfSquares]));
        v.extend(x.reduce_axis(1, [Reduction::Min, Reduction::Max]));
        v
    });
    run(&ctx, "one call: d sum,sumsq,min,max", &|x| {
        let d = x - &k;
        d.reduce_axis(1, [Reduction::Sum, Reduction::SumOfSquares, Reduction::Min, Reduction::Max]).into()
    });
}
