use arrayfire as af;
use arrayfire::{Array, Dim4};
use common::*;

fn k(v: f32, d: Dim4) -> Array<f32> {
    af::constant(v, d)
}

fn host(a: &Array<f32>) -> Vec<f32> {
    let mut v = vec![0f32; a.elements()];
    a.host(&mut v);
    v
}

fn sh(a: &Array<f32>, x: i32, y: i32) -> Array<f32> {
    af::shift(a, &[x, y, 0, 0])
}

fn heat_step(u: &Array<f32>, c4: &Array<f32>, alpha: &Array<f32>) -> Array<f32> {
    let lap = sh(u, 1, 0) + sh(u, -1, 0) + sh(u, 0, 1) + sh(u, 0, -1) - u * c4;
    u + lap * alpha
}

fn life_step(
    u: &Array<f32>,
    ones: &Array<f32>,
    zeros: &Array<f32>,
    two: &Array<f32>,
    three: &Array<f32>,
) -> Array<f32> {
    let mut n = sh(u, 1, 0);
    for (dx, dy) in [(-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
        n = n + sh(u, dx, dy);
    }
    let keep = af::select(u, &af::eq(&n, two, false), zeros);
    af::select(ones, &af::eq(&n, three, false), &keep)
}

fn cnd(d: &Array<f32>, c: &Consts) -> Array<f32> {
    let kk = &c.one / &(af::abs(d) * &c.c1 + &c.one);
    let mut poly = c.a[4].clone();
    for i in [3, 2, 1, 0] {
        poly = &c.a[i] + &kk * &poly;
    }
    let poly = &kk * poly;
    let tail = af::exp(&(d * d * &c.neg_half)) * &c.inv * poly;
    af::select(&(&c.one - &tail), &af::gt(d, &c.zero, false), &tail)
}

struct Consts {
    one: Array<f32>,
    zero: Array<f32>,
    neg_half: Array<f32>,
    c1: Array<f32>,
    inv: Array<f32>,
    a: Vec<Array<f32>>,
}

fn bs_price(s: &Array<f32>, kx: &Array<f32>, t: &Array<f32>, c: &Consts, d: Dim4) -> Array<f32> {
    let vst = af::sqrt(t) * &k(VOL, d);
    let d1 = (af::log(&(s / kx)) + t * &k(RATE + 0.5 * VOL * VOL, d)) / &vst;
    let d2 = &d1 - &vst;
    s * cnd(&d1, c) - kx * af::exp(&(t * &k(-RATE, d))) * cnd(&d2, c)
}

fn sync() {
    af::sync(0);
}

fn emit(workload: &str, size: String, variant: &str, backend: &str, work: f64, unit: &str, err: f32, f: impl FnMut()) {
    eprintln!("[af-bench] {workload} {size} {variant} err={err:e}");
    let t = time(f);
    print_row(&Row { workload, size, variant, library: "arrayfire", backend, work, unit, err }, &t);
}

fn heat(sz: &Sizes, be: &str) {
    for &w in &sz.heat {
        let d = Dim4::new(&[w as u64, w as u64, 1, 1]);
        let h0 = values(w * w, 10, 0.0, 1.0);
        let u0 = Array::new(&h0, d);
        let (c4, al) = (k(4.0, d), k(ALPHA, d));
        let run = || {
            let mut u = u0.copy();
            for _ in 0..sz.heat_steps {
                u = heat_step(&u, &c4, &al);
                u.eval();
            }
            sync();
            u
        };
        let want = heat_ref(w, w, &h0, sz.heat_steps);
        let err = rel_err(&host(&run()), &want);
        assert!(err <= 1e-4, "heat {w}: rel_err {err} > 1e-4");
        emit("heat", format!("{w}x{w}"), "per_step", be, (w * w * sz.heat_steps) as f64, "Gcell/s", err, || {
            run();
        });
    }
}

fn life(sz: &Sizes, be: &str) {
    for &w in &sz.life {
        let d = Dim4::new(&[w as u64, w as u64, 1, 1]);
        let g0 = life_grid(w, w);
        let u0 = Array::new(&g0, d);
        let (ones, zeros, two, three) = (k(1.0, d), k(0.0, d), k(2.0, d), k(3.0, d));
        let run = || {
            let mut u = u0.copy();
            for _ in 0..sz.life_steps {
                u = life_step(&u, &ones, &zeros, &two, &three);
                u.eval();
            }
            sync();
            u
        };
        let want = life_ref(w, w, &g0, sz.life_steps);
        let err = mismatch_frac(&host(&run()), &want);
        assert!(err == 0.0, "life {w}: mismatch_frac {err} != 0");
        emit("life", format!("{w}x{w}"), "per_step", be, (w * w * sz.life_steps) as f64, "Gcell/s", err, || {
            run();
        });
    }
}

fn blackscholes(sz: &Sizes, be: &str) {
    for &n in &sz.bs {
        let d = Dim4::new(&[n as u64, 1, 1, 1]);
        let (hs, hk, ht) = (values(n, 1, 10.0, 100.0), values(n, 2, 10.0, 100.0), values(n, 3, 0.1, 5.0));
        let (s, kx, t) = (Array::new(&hs, d), Array::new(&hk, d), Array::new(&ht, d));
        let c = Consts {
            one: k(1.0, d),
            zero: k(0.0, d),
            neg_half: k(-0.5, d),
            c1: k(0.2316419, d),
            inv: k(INV_SQRT_2PI, d),
            a: A.iter().map(|&v| k(v, d)).collect(),
        };
        let want = bs_ref(&hs, &hk, &ht);
        let want_sum: f64 = want.iter().map(|&v| v as f64).sum();
        let price = || {
            let p = bs_price(&s, &kx, &t, &c, d);
            p.eval();
            sync();
            p
        };
        let err = rel_err(&host(&price()), &want);
        assert!(err <= 1e-4, "bs {n}: rel_err {err} > 1e-4");
        emit("blackscholes", n.to_string(), "price", be, n as f64, "Gopt/s", err, || {
            price();
        });
        let sum = || {
            let p = bs_price(&s, &kx, &t, &c, d);
            let (r, _) = af::sum_all(&p);
            r as f64
        };
        let err = ((sum() - want_sum).abs() / want_sum.abs()) as f32;
        assert!(err <= 1e-3, "bs {n}: sum err {err} > 1e-3");
        emit("blackscholes", n.to_string(), "price_sum", be, n as f64, "Gopt/s", err, || {
            std::hint::black_box(sum());
        });
    }
}

fn mandelbrot(sz: &Sizes, be: &str) {
    for &w in &sz.mandel {
        let d = Dim4::new(&[w as u64, w as u64, 1, 1]);
        let (hx, hy) = mandel_c(w, w);
        let (cx, cy) = (Array::new(&hx, d), Array::new(&hy, d));
        let (ones, zeros, two, four) = (k(1.0, d), k(0.0, d), k(2.0, d), k(4.0, d));
        let want = mandel_ref(w, w, sz.mandel_iters);
        for every in [1usize, 8, 0] {
            let variant = if every == 0 { "eval_end_only".to_string() } else { format!("eval_every_{every}") };
            let run = || {
                let mut zx = k(0.0, d);
                let mut zy = k(0.0, d);
                let mut cnt = k(0.0, d);
                for i in 1..=sz.mandel_iters {
                    let (xx, yy) = (&zx * &zx, &zy * &zy);
                    let inside = af::le(&(&xx + &yy), &four, false);
                    let nzx = &xx - &yy + &cx;
                    let nzy = &two * &zx * &zy + &cy;
                    zx = af::select(&nzx, &inside, &zx);
                    zy = af::select(&nzy, &inside, &zy);
                    cnt = &cnt + af::select(&ones, &inside, &zeros);
                    if every > 0 && i % every == 0 {
                        af::eval_multiple(vec![&zx, &zy, &cnt]);
                    }
                }
                af::eval_multiple(vec![&zx, &zy, &cnt]);
                sync();
                cnt
            };
            let err = mismatch_frac(&host(&run()), &want);
            assert!(err < 0.01, "mandelbrot {w} every {every}: mismatch_frac {err} >= 0.01");
            emit(
                "mandelbrot",
                format!("{w}x{w}"),
                &variant,
                be,
                (w * w * sz.mandel_iters) as f64,
                "Gpix-iter/s",
                err,
                || {
                    run();
                },
            );
        }
    }
}

fn fir(sz: &Sizes, be: &str) {
    let n = sz.fir_n;
    let hx = values(n, 11, -1.0, 1.0);
    let x = Array::new(&hx, Dim4::new(&[n as u64, 1, 1, 1]));
    for &taps in &sz.fir_taps {
        let hh = values(taps, 12, -0.1, 0.1);
        let h = Array::new(&hh, Dim4::new(&[taps as u64, 1, 1, 1]));
        let out_len = n + taps - 1;
        let idx = fir_sample_indices(out_len);
        let want = fir_ref_sample(&hx, &hh, &idx);
        let size = format!("n={n},taps={taps}");
        let tol = if taps >= 65535 { 1e-3 } else { 1e-4 };
        type Conv = Box<dyn Fn() -> Array<f32>>;
        let mut variants: Vec<(&str, Conv)> = vec![];
        {
            let (x, h) = (x.clone(), h.clone());
            variants.push((
                "fft",
                Box::new(move || {
                    let y = af::fft_convolve1(&x, &h, af::ConvMode::EXPAND);
                    y.eval();
                    sync();
                    y
                }),
            ));
        }
        let direct_supported = taps <= sz.fir_direct_max_taps && {
            // ArrayFire's spatial 1-D convolve rejects long filters (observed limit: between 129 and 199 taps).
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                af::convolve1(&x, &h, af::ConvMode::EXPAND, af::ConvDomain::SPATIAL).eval();
                sync();
            }))
            .is_ok();
            std::panic::set_hook(prev);
            if !ok {
                eprintln!("[af-bench] fir direct {size}: SKIPPED, ArrayFire spatial convolve1 unsupported for {taps} taps");
            }
            ok
        };
        if direct_supported {
            let (x, h) = (x.clone(), h.clone());
            variants.push((
                "direct",
                Box::new(move || {
                    let y = af::convolve1(&x, &h, af::ConvMode::EXPAND, af::ConvDomain::SPATIAL);
                    y.eval();
                    sync();
                    y
                }),
            ));
        }
        for (name, f) in variants {
            let y = f();
            assert_eq!(y.elements(), out_len, "fir {name} {size}: output length");
            let yh = host(&y);
            let got: Vec<f32> = idx.iter().map(|&i| yh[i]).collect();
            let err = rel_err(&got, &want);
            assert!(err <= tol, "fir {name} {size}: rel_err {err} > {tol}");
            drop(y);
            emit("fir", size.clone(), name, be, n as f64, "Gsample/s", err, || {
                f();
            });
        }
    }
}

fn main() {
    let args = parse_args();
    let be = args.backend.clone().unwrap_or_else(|| "cuda".into());
    let backend = match be.as_str() {
        "cuda" => af::Backend::CUDA,
        "opencl" => af::Backend::OPENCL,
        "cpu" => af::Backend::CPU,
        o => panic!("unknown backend {o}"),
    };
    af::set_backend(backend);
    af::set_device(0);
    eprintln!("{}", af::info_string(true));
    let sz = sizes(args.quick);
    print_header();
    type W = fn(&Sizes, &str);
    let list: [(&str, W); 5] =
        [("heat", heat), ("life", life), ("blackscholes", blackscholes), ("mandelbrot", mandelbrot), ("fir", fir)];
    for (name, f) in list {
        if wants(&args, name) {
            f(&sz, &be);
            af::device_gc();
        }
    }
}
