//! Luma side of the Luma vs ArrayFire benchmark.

use common::*;
use luma::*;

const BACKEND: &str = "vulkan";

fn heat_step(u: &Array<f32>) -> Array<f32> {
    let lap = u.shift([1isize, 0]) + u.shift([-1isize, 0]) + u.shift([0isize, 1]) + u.shift([0isize, -1]) - u * 4.0;
    u + lap * ALPHA
}

fn per_step(g0: &Array<f32>, steps: usize, step: impl Fn(&Array<f32>) -> Array<f32>) -> Array<f32> {
    let mut u = g0 * 1.0;
    for _ in 0..steps {
        u = step(&u);
    }
    u
}

fn life_step(u: &Array<f32>) -> Array<f32> {
    let mut n = u.shift([1isize, 1]) + u.shift([1isize, 0]) + u.shift([1isize, -1]);
    n = n + u.shift([0isize, 1]) + u.shift([0isize, -1]);
    n = n + u.shift([-1isize, 1]) + u.shift([-1isize, 0]) + u.shift([-1isize, -1]);
    let survive = select(&n.eq(2.0), u, 0.0);
    select(&n.eq(3.0), 1.0, &survive)
}

fn cnd(d: &Array<f32>) -> Array<f32> {
    let k = 1.0 / (d.abs() * 0.2316419 + 1.0);
    let poly = &k * (A[0] + &k * (A[1] + &k * (A[2] + &k * (A[3] + &k * A[4]))));
    let tail = (d * d * -0.5).exp() * INV_SQRT_2PI * poly;
    select(&d.gt(0.0), &(1.0 - &tail), &tail)
}

fn black_scholes(s: &Array<f32>, k: &Array<f32>, t: &Array<f32>) -> Array<f32> {
    let vol_sqrt_t = t.sqrt() * VOL;
    let d1 = ((s / k).ln() + t * (RATE + 0.5 * VOL * VOL)) / &vol_sqrt_t;
    let d2 = &d1 - &vol_sqrt_t;
    s * cnd(&d1) - k * (t * -RATE).exp() * cnd(&d2)
}

const MANDELBROT: &str = r#"
struct Params { x0: f32, y0: f32, dx: f32, dy: f32, max_iter: u32 }
@group(1) @binding(0) var<storage, read_write> out: array<u32>;
@group(1) @binding(1) var<uniform> p: Params;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= luma.size.x || g.y >= luma.size.y) { return; }
    let cx = p.x0 + f32(g.x) * p.dx;
    let cy = p.y0 + f32(g.y) * p.dy;
    var x = 0.0;
    var y = 0.0;
    var i = 0u;
    loop {
        if (i >= p.max_iter || x * x + y * y > 4.0) { break; }
        let xt = x * x - y * y + cx;
        y = 2.0 * x * y + cy;
        x = xt;
        i = i + 1u;
    }
    out[luma_idx2(g.x, g.y)] = i;
}
"#;

#[derive(encase::ShaderType)]
struct MandelbrotParams {
    x0: f32,
    y0: f32,
    dx: f32,
    dy: f32,
    max_iter: u32,
}

fn check(name: &str, err: f32, tol: f32) {
    assert!(err <= tol, "{name}: error {err:e} exceeds tolerance {tol:e}");
}

fn report(row: Row, mut f: impl FnMut()) {
    eprintln!("  {} {} {}: err {:e}", row.workload, row.size, row.variant, row.err);
    let ctx = default_context().unwrap();
    let t = time(|| {
        f();
        ctx.sync().unwrap();
    });
    print_row(&row, &t);
}

fn heat(sz: &Sizes) {
    let steps = sz.heat_steps;
    for &w in &sz.heat {
        let u0 = values(w * w, 10, 0.0, 1.0);
        let want = heat_ref(w, w, &u0, steps);
        let g = Array::from_slice(&u0, [w, w]).unwrap();
        let mk = |variant: &'static str, run: &dyn Fn() -> Array<f32>| {
            let err = rel_err(&run().to_vec().unwrap(), &want);
            check(&format!("heat {w} {variant}"), err, 1e-4);
            Row {
                workload: "heat",
                size: format!("{w}x{w}"),
                variant,
                library: "luma",
                backend: BACKEND,
                work: (w * w * steps) as f64,
                unit: "Gcell/s",
                err,
            }
        };
        let it = || g.iterate(steps, heat_step);
        report(mk("iterate", &it), || it().eval().unwrap());
        let ps = || per_step(&g, steps, heat_step);
        report(mk("per_step", &ps), || ps().eval().unwrap());
    }
}

fn life(sz: &Sizes) {
    let steps = sz.life_steps;
    for &w in &sz.life {
        let g0 = life_grid(w, w);
        let want = life_ref(w, w, &g0, steps);
        let g = Array::from_slice(&g0, [w, w]).unwrap();
        let mk = |variant: &'static str, run: &dyn Fn() -> Array<f32>| {
            let err = mismatch_frac(&run().to_vec().unwrap(), &want);
            check(&format!("life {w} {variant}"), err, 0.0);
            Row {
                workload: "life",
                size: format!("{w}x{w}"),
                variant,
                library: "luma",
                backend: BACKEND,
                work: (w * w * steps) as f64,
                unit: "Gcell/s",
                err,
            }
        };
        let it = || g.iterate(steps, life_step);
        report(mk("iterate", &it), || it().eval().unwrap());
        let ps = || per_step(&g, steps, life_step);
        report(mk("per_step", &ps), || ps().eval().unwrap());
    }
}

fn blackscholes(sz: &Sizes) {
    for &n in &sz.bs {
        let s = values(n, 1, 10.0, 100.0);
        let k = values(n, 2, 10.0, 100.0);
        let t = values(n, 3, 0.1, 5.0);
        let want = bs_ref(&s, &k, &t);
        let want_sum: f64 = want.iter().map(|&v| v as f64).sum();
        let (gs, gk, gt) = (
            Array::from_slice(&s, n).unwrap(),
            Array::from_slice(&k, n).unwrap(),
            Array::from_slice(&t, n).unwrap(),
        );
        let row = |variant: &'static str, err: f32| Row {
            workload: "blackscholes",
            size: n.to_string(),
            variant,
            library: "luma",
            backend: BACKEND,
            work: n as f64,
            unit: "Gopt/s",
            err,
        };
        let err = rel_err(&black_scholes(&gs, &gk, &gt).to_vec().unwrap(), &want);
        check(&format!("blackscholes {n} price"), err, 1e-4);
        report(row("price", err), || black_scholes(&gs, &gk, &gt).eval().unwrap());

        let got_sum = black_scholes(&gs, &gk, &gt).sum().unwrap() as f64;
        let err = ((got_sum - want_sum).abs() / want_sum.abs()) as f32;
        check(&format!("blackscholes {n} price_sum"), err, 1e-3);
        report(row("price_sum", err), || {
            std::hint::black_box(black_scholes(&gs, &gk, &gt).sum().unwrap());
        });
    }
}

fn mandelbrot(sz: &Sizes) {
    let iters = sz.mandel_iters;
    for &w in &sz.mandel {
        let want = mandel_ref(w, w, iters);
        let (cx, cy) = mandel_c(w, w);
        let (cx, cy) = (Array::from_slice(&cx, [w, w]).unwrap(), Array::from_slice(&cy, [w, w]).unwrap());
        let zero = Array::<f32>::full([w, w], 0.0).unwrap();
        let run = |every: usize| -> Array<f32> {
            let (mut zx, mut zy, mut cnt) = (&zero * 1.0, &zero * 1.0, &zero * 1.0);
            for i in 1..=iters {
                let inside = (&zx * &zx + &zy * &zy).le(4.0);
                let nzx = &zx * &zx - &zy * &zy + &cx;
                let nzy = &zx * &zy * 2.0 + &cy;
                zx = select(&inside, &nzx, &zx);
                zy = select(&inside, &nzy, &zy);
                cnt = &cnt + select(&inside, 1.0, 0.0);
                if every > 0 && i % every == 0 {
                    eval_all(&[&zx, &zy, &cnt]).unwrap();
                }
            }
            eval_all(&[&zx, &zy, &cnt]).unwrap();
            cnt
        };
        let row = |variant: &'static str, err: f32| Row {
            workload: "mandelbrot",
            size: format!("{w}x{w}"),
            variant,
            library: "luma",
            backend: BACKEND,
            work: (w * w * iters) as f64,
            unit: "Gpix-iter/s",
            err,
        };
        for (variant, every) in [("eval_every_1", 1usize), ("eval_every_8", 8), ("eval_end_only", 0)] {
            let err = mismatch_frac(&run(every).to_vec().unwrap(), &want);
            check(&format!("mandelbrot {w} {variant}"), err, 0.01);
            report(row(variant, err), || {
                run(every);
            });
        }

        let looped = || -> Array<f32> {
            let [_, _, cnt] = iterate_all([&zero, &zero, &zero], iters, |[zx, zy, cnt]| {
                let inside = (zx * zx + zy * zy).le(4.0);
                let nzx = zx * zx - zy * zy + &cx;
                let nzy = zx * zy * 2.0 + &cy;
                [select(&inside, &nzx, zx), select(&inside, &nzy, zy), cnt + select(&inside, 1.0, 0.0)]
            });
            cnt
        };
        let err = mismatch_frac(&looped().to_vec().unwrap(), &want);
        check(&format!("mandelbrot {w} iterate_all"), err, 0.01);
        report(row("iterate_all", err), || looped().eval().unwrap());

        let kernel = Kernel::new(MANDELBROT).unwrap();
        let ctx = default_context().unwrap();
        let p = MandelbrotParams { x0: -2.2, y0: -1.4, dx: 3.2 / w as f32, dy: 2.8 / w as f32, max_iter: iters as u32 };
        let mut img = Array::<u32>::zeros([w, w]).unwrap();
        let launch = |img: &mut Array<u32>| {
            kernel.launch([w as u32, w as u32], args![&mut *img, uniform(&p)]).unwrap();
            ctx.sync().unwrap();
        };
        launch(&mut img);
        let got: Vec<f32> = img.to_vec().unwrap().into_iter().map(|v| v as f32).collect();
        let err = mismatch_frac(&got, &want);
        check(&format!("mandelbrot {w} custom_kernel"), err, 0.01);
        report(row("custom_kernel", err), || launch(&mut img));
    }
}

fn fir(sz: &Sizes) {
    let n = sz.fir_n;
    let x = values(n, 11, -1.0, 1.0);
    let gx = Array::from_slice(&x, n).unwrap();
    for &taps in &sz.fir_taps {
        let h = values(taps, 12, -0.1, 0.1);
        let idx = fir_sample_indices(n + taps - 1);
        let want = fir_ref_sample(&x, &h, &idx);
        let gh = Array::from_slice(&h, taps).unwrap();
        let sample = |y: &Array<f32>| -> Vec<f32> {
            assert_eq!(y.len(), n + taps - 1, "fir output length");
            let all = y.to_vec().unwrap();
            idx.iter().map(|&i| all[i]).collect()
        };
        let row = |variant: &'static str, err: f32| Row {
            workload: "fir",
            size: format!("n={n},taps={taps}"),
            variant,
            library: "luma",
            backend: BACKEND,
            work: n as f64,
            unit: "Gsample/s",
            err,
        };
        let err = rel_err(&sample(&gx.conv1d_fft(&gh)), &want);
        check(&format!("fir {taps} fft"), err, if taps > 4095 { 1e-3 } else { 1e-4 });
        report(row("fft", err), || gx.conv1d_fft(&gh).eval().unwrap());

        if taps <= sz.fir_direct_max_taps {
            let hr: Vec<f32> = h.iter().rev().copied().collect();
            let ghr = Array::from_slice(&hr, taps).unwrap();
            let err = rel_err(&sample(&gx.conv1d(&ghr, 1, taps - 1)), &want);
            check(&format!("fir {taps} direct"), err, 1e-4);
            report(row("direct", err), || gx.conv1d(&ghr, 1, taps - 1).eval().unwrap());
        }
    }
}

fn main() {
    let args = parse_args();
    let sz = sizes(args.quick);
    print_header();
    let workloads: [(&str, fn(&Sizes)); 5] =
        [("heat", heat), ("life", life), ("blackscholes", blackscholes), ("mandelbrot", mandelbrot), ("fir", fir)];
    for (name, f) in workloads {
        if wants(&args, name) {
            eprintln!("luma: {name}");
            f(&sz);
        }
    }
}
