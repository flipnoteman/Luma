//! Many reductions over the same data: a force monitor (masked sums of one expression) and a
//! `describe` (mean, variance, minimum, maximum of every column).
//!
//! `cargo run --release --example bench_multireduce -- [--quick] [--only <substring>]`
//!
//! Every case is checked once against a host reference, then timed (median ms of the call, host
//! wall time) and profiled once (`kernels` launched, `gpu ms`). `LUMA_MULTI_REDUCE=off` runs every
//! reduction in a kernel of its own.

use luma::*;
use std::hint::black_box;
use std::time::Instant;

fn values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + ((state >> 40) as f32) / (1u64 << 24) as f32 * (hi - lo)
        })
        .collect()
}

fn time(ctx: &Context, name: &str, size: &str, f: &mut dyn FnMut()) {
    for _ in 0..3 {
        f();
    }
    let mut samples: Vec<f64> = (0..15)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    let (kernels, gpu) = match ctx.profile(f) {
        Ok(((), p)) => {
            if std::env::var_os("BENCH_KERNELS").is_some() {
                for k in &p.kernels {
                    println!("    {:<90} {:>4} {:>9.3} ms", k.label.chars().take(90).collect::<String>(), k.launches, k.total.as_secs_f64() * 1e3);
                }
            }
            (p.launches().to_string(), format!("{:.3}", p.total().as_secs_f64() * 1e3))
        }
        Err(e) => ("err".into(), e.to_string()),
    };
    println!("{name:<30} {size:>12} {:>10.3} {:>10.3} {kernels:>8} {gpu:>9}", samples[samples.len() / 2], samples[0]);
}

fn main() {
    let mut quick = false;
    let mut only = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--quick" => quick = true,
            "--only" => only = args.next(),
            "--bench" => {}
            other => panic!("unknown argument {other}"),
        }
    }
    let want = |name: &str| only.as_deref().is_none_or(|f| name.contains(f));
    let ctx = default_context().unwrap();
    println!("bench_multireduce, profiling supported: {}", ctx.profiling_supported());
    println!("{:<30} {:>12} {:>10} {:>10} {:>8} {:>9}", "case", "size", "ms", "min ms", "kernels", "gpu ms");

    // Monitor: `k` masked sums of the pressure expression `p`, each read back.
    let sizes: &[usize] = if quick { &[200_000] } else { &[200_000, 1_000_000] };
    for &n in sizes {
        let a = Array::from_slice_on(&ctx, &values(n, 1, 0.5, 2.0), n).unwrap();
        let b = Array::from_slice_on(&ctx, &values(n, 2, 0.5, 2.0), n).unwrap();
        let host_a = values(n, 1, 0.5, 2.0);
        let host_b = values(n, 2, 0.5, 2.0);
        for (k, lazy) in [(12usize, false), (24, false), (24, true)] {
            let name = format!("monitor/masked_sums_{k}{}", if lazy { "_lazy_masks" } else { "" });
            if !want(&name) {
                continue;
            }
            // `k` disjoint regions of the cells (a surface class each).
            let class: Vec<u32> = (0..n).map(|i| ((i * 7919) % (k + 3)) as u32).collect();
            let class = Array::from_slice_on(&ctx, &class, n).unwrap();
            // Evaluated masks are one binding each; lazy ones are one comparison of the class array each.
            let masks: Vec<Array<bool>> = (0..k).map(|c| class.eq(c as u32)).collect();
            if !lazy {
                for m in &masks {
                    m.eval().unwrap();
                }
            }
            let mut out = vec![0f32; k];
            let run = |out: &mut Vec<f32>| {
                // The pressure is an expression shared by every sum.
                let p = &a * &b + 0.25;
                let sums: Vec<Array<f32>> = masks.iter().map(|m| p.sum_where(m)).collect();
                for (o, s) in out.iter_mut().zip(&sums) {
                    *o = s.item().unwrap();
                }
            };
            run(&mut out);
            let mut want_sums = vec![0f64; k];
            for i in 0..n {
                let c = (i * 7919) % (k + 3);
                if c < k {
                    want_sums[c] += f64::from(host_a[i] * host_b[i] + 0.25);
                }
            }
            for (c, (g, w)) in out.iter().zip(&want_sums).enumerate() {
                assert!((f64::from(*g) - w).abs() <= 1e-3 * w.abs().max(1.0), "{name}: class {c}: {g} vs {w}");
            }
            time(&ctx, &name, &n.to_string(), &mut || {
                run(&mut out);
                black_box(&out);
            });
        }
    }

    // Describe: mean, variance, minimum and maximum of `cols` columns.
    let shapes: &[(usize, usize)] = if quick { &[(8, 100_000)] } else { &[(8, 100_000), (8, 1_000_000)] };
    for &(cols, n) in shapes {
        let name = "describe/mean_var_min_max";
        if !want(name) {
            continue;
        }
        let hosts: Vec<Vec<f32>> = (0..cols).map(|c| values(n, 100 + c as u64, -10.0, 30.0)).collect();
        let columns: Vec<Array<f32>> = hosts.iter().map(|h| Array::from_slice_on(&ctx, h, n).unwrap()).collect();
        let run = |out: &mut Vec<f32>| {
            let stats: Vec<_> = columns
                .iter()
                .map(|c| {
                    let mean = c.mean();
                    let centered = c - &mean;
                    let var = (&centered * &centered).mean();
                    (mean, var, c.min(), c.max())
                })
                .collect();
            out.clear();
            for (mean, var, min, max) in &stats {
                out.extend([mean.item().unwrap(), var.item().unwrap(), min.item().unwrap(), max.item().unwrap()]);
            }
        };
        let mut out = Vec::new();
        run(&mut out);
        for (c, h) in hosts.iter().enumerate() {
            let mean = h.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
            let var = h.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / n as f64;
            assert!((f64::from(out[4 * c]) - mean).abs() < 1e-3 * mean.abs().max(1.0), "mean {c}");
            assert!((f64::from(out[4 * c + 1]) - var).abs() < 1e-3 * var, "var {c}");
            assert_eq!(out[4 * c + 2], h.iter().copied().fold(f32::INFINITY, f32::min));
            assert_eq!(out[4 * c + 3], h.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        }
        time(&ctx, name, &format!("{cols}x{n}"), &mut || {
            run(&mut out);
            black_box(&out);
        });
    }

    // The same statistics, every column's mean first (so that the means are pending together), then the variances.
    for &(cols, n) in shapes {
        let name = "describe/phased_mean_var_min_max";
        if !want(name) {
            continue;
        }
        let columns: Vec<Array<f32>> = (0..cols).map(|c| Array::from_slice_on(&ctx, &values(n, 100 + c as u64, -10.0, 30.0), n).unwrap()).collect();
        let run = |out: &mut Vec<f32>| {
            let means: Vec<Array<f32>> = columns.iter().map(|c| c.mean()).collect();
            let mins: Vec<Array<f32>> = columns.iter().map(|c| c.min()).collect();
            let maxs: Vec<Array<f32>> = columns.iter().map(|c| c.max()).collect();
            let vars: Vec<Array<f32>> = columns
                .iter()
                .zip(&means)
                .map(|(c, mean)| {
                    let centered = c - mean;
                    (&centered * &centered).mean()
                })
                .collect();
            out.clear();
            for c in 0..cols {
                out.extend([means[c].item().unwrap(), vars[c].item().unwrap(), mins[c].item().unwrap(), maxs[c].item().unwrap()]);
            }
        };
        let mut out = Vec::new();
        run(&mut out);
        time(&ctx, name, &format!("{cols}x{n}"), &mut || {
            run(&mut out);
            black_box(&out);
        });
    }

    // A force monitor: `3 * classes` masked sums of components of one gathered expression, each reshaped to one
    // element, joined by `concat_slice` and read back once (`plain` reads every sum on its own).
    for &n in sizes {
        for variant in ["reshape_concat", "plain"] {
            let name = format!("forces/{variant}");
            if !want(&name) {
                continue;
            }
            let p = Array::from_slice_on(&ctx, &values(n, 300, 0.0, 1.0), n).unwrap();
            let m = n / 4;
            let idx: Vec<u32> = (0..m).map(|i| ((i * 3) % n) as u32).collect();
            let idx = Array::from_slice_on(&ctx, &idx, m).unwrap();
            let cx = Array::<f32>::full_on(&ctx, m, 0.5).unwrap();
            let masks: Vec<Array<bool>> = (0..8u32)
                .map(|k| Array::from_slice_on(&ctx, &(0..m).map(|i| (i % 8) as u32 == k).collect::<Vec<_>>(), m).unwrap())
                .collect();
            let run = |out: &mut Vec<f32>| {
                let d = &p.take(&idx) - 1.0f32;
                let comps = [&d * &cx, &d * 2.0f32, &d * &d];
                let mut sums = Vec::new();
                for mask in &masks {
                    for comp in &comps {
                        let s = comp.sum_where(mask);
                        sums.push(if variant == "plain" { s } else { s.reshape([1]) });
                    }
                }
                out.clear();
                if variant == "plain" {
                    out.extend(sums.iter().map(|s| s.item().unwrap()));
                } else {
                    out.extend(concat_slice(0, &sums).to_vec().unwrap());
                }
            };
            let (mut out, mut again) = (Vec::new(), Vec::new());
            run(&mut out);
            run(&mut again);
            assert_eq!(out.len(), 24);
            assert!(out.iter().zip(&again).all(|(a, b)| a.to_bits() == b.to_bits()), "{name}: unstable");
            time(&ctx, &name, &n.to_string(), &mut || {
                run(&mut out);
                black_box(&out);
            });
        }
    }

    // `wall_forces` of a 3-D body: the wall pressure is a pending expression of the whole state, read through two gathers,
    // and `3 * masks` sums of its components are joined and read back once.
    {
        let (masks_n, mut faces) = if quick { (16usize, 20_000usize) } else { (64, 200_000) };
        if let Some(n) = std::env::var("BENCH_FACES").ok().and_then(|v| v.parse().ok()) {
            faces = n;
        }
        let depth: usize = std::env::var("BENCH_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let name = "forces/wall_forces";
        if want(name) {
            let cells = faces * 4;
            let state: Vec<Array<f32>> = (0..5).map(|k| Array::from_slice_on(&ctx, &values(cells, 500 + k, 0.5, 1.5), cells).unwrap()).collect();
            let ia: Vec<u32> = (0..faces).map(|i| ((i * 7) % cells) as u32).collect();
            let ib: Vec<u32> = (0..faces).map(|i| ((i * 7 + 1) % cells) as u32).collect();
            let (ia, ib) = (Array::from_slice_on(&ctx, &ia, faces).unwrap(), Array::from_slice_on(&ctx, &ib, faces).unwrap());
            let k = Array::from_slice_on(&ctx, &values(faces, 510, 0.0, 1.0), faces).unwrap();
            let weights: Vec<Array<f32>> = (0..3).map(|c| Array::from_slice_on(&ctx, &values(faces, 520 + c, -1.0, 1.0), faces).unwrap()).collect();
            let masks: Vec<Array<bool>> = (0..masks_n)
                .map(|c| Array::from_slice_on(&ctx, &(0..faces).map(|i| i % masks_n == c).collect::<Vec<_>>(), faces).unwrap())
                .collect();
            let run = |out: &mut Vec<f32>| {
                let speed = &state[1] * &state[1] + &state[2] * &state[2] + &state[3] * &state[3];
                let mut pall = (&state[4] - &speed * &state[0] * 0.5f32) * 0.4f32;
                // A deep state expression (the padded state of a 3-D solver with its ghost layers), `BENCH_DEPTH` steps of it.
                for step in 0..depth {
                    pall = &pall * 0.999f32 + &state[step % 4] * 0.001f32;
                }
                let (pa, pb) = (pall.take(&ia), pall.take(&ib));
                let pw = &pa + &k * (&pa - &pb);
                let d = pw - 0.7f32;
                let comps: Vec<Array<f32>> = weights.iter().map(|w| &d * w).collect();
                let mut sums = Vec::new();
                for mask in &masks {
                    for comp in &comps {
                        sums.push(comp.sum_where(mask).reshape([1]));
                    }
                }
                out.clear();
                out.extend(concat_slice(0, &sums).to_vec().unwrap());
            };
            let (mut out, mut again) = (Vec::new(), Vec::new());
            run(&mut out);
            run(&mut again);
            assert_eq!(out.len(), masks_n * 3);
            assert!(out.iter().zip(&again).all(|(a, b)| a.to_bits() == b.to_bits()), "{name}: unstable");
            time(&ctx, name, &format!("{masks_n}x{faces}"), &mut || {
                run(&mut out);
                black_box(&out);
            });
        }
    }

    // The stop rule's forces at the scale of a 3-D sector: pressure and shear sums of 3 components over `classes` masks
    // of `faces` gathered faces (about 200 reductions of two expressions), joined and read back once.
    {
        let (classes, mut faces) = if quick { (8usize, 100_000usize) } else { (32, 1_000_000) };
        if let Some(n) = std::env::var("BENCH_FACES").ok().and_then(|v| v.parse().ok()) {
            faces = n;
        }
        let name = "forces/scale_pressure_shear";
        if want(name) {
            let cells = faces * 2;
            let p = Array::from_slice_on(&ctx, &values(cells, 400, 0.0, 1.0), cells).unwrap();
            let tau = Array::from_slice_on(&ctx, &values(cells, 401, 0.0, 1.0), cells).unwrap();
            let idx: Vec<u32> = (0..faces).map(|i| ((i * 7) % cells) as u32).collect();
            let idx = Array::from_slice_on(&ctx, &idx, faces).unwrap();
            let normals: Vec<Array<f32>> = (0..3).map(|k| Array::<f32>::full_on(&ctx, faces, 0.3 + k as f32).unwrap()).collect();
            let masks: Vec<Array<bool>> = (0..classes)
                .map(|c| Array::from_slice_on(&ctx, &(0..faces).map(|i| i % classes == c).collect::<Vec<_>>(), faces).unwrap())
                .collect();
            let run = |out: &mut Vec<f32>| {
                let pf = &p.take(&idx) - 1.0f32;
                let tf = &tau.take(&idx) * 0.5f32;
                let mut sums = Vec::new();
                for mask in &masks {
                    for n in &normals {
                        sums.push((&pf * n).sum_where(mask).reshape([1]));
                        sums.push((&tf * n).sum_where(mask).reshape([1]));
                    }
                }
                out.clear();
                out.extend(concat_slice(0, &sums).to_vec().unwrap());
            };
            let (mut out, mut again) = (Vec::new(), Vec::new());
            run(&mut out);
            run(&mut again);
            assert_eq!(out.len(), classes * 6);
            assert!(out.iter().zip(&again).all(|(a, b)| a.to_bits() == b.to_bits()), "{name}: unstable");
            time(&ctx, name, &format!("{classes}x{faces}"), &mut || {
                run(&mut out);
                black_box(&out);
            });
        }
    }

    // Norms of several fields and dot products that share an operand.
    for &n in sizes {
        let name = "norms/6_fields_and_3_dots";
        if !want(name) {
            continue;
        }
        let fields: Vec<Array<f32>> = (0..6).map(|c| Array::from_slice_on(&ctx, &values(n, 200 + c, -1.0, 1.0), n).unwrap()).collect();
        let run = |out: &mut Vec<f32>| {
            let mut pending: Vec<Array<f32>> = fields.iter().map(|f| f.norm()).collect();
            for pair in fields.chunks(2) {
                pending.push((&pair[0] * &pair[1]).sum());
            }
            out.clear();
            out.extend(pending.iter().map(|p| p.item().unwrap()));
        };
        let mut out = Vec::new();
        run(&mut out);
        time(&ctx, name, &n.to_string(), &mut || {
            run(&mut out);
            black_box(&out);
        });
    }
}
