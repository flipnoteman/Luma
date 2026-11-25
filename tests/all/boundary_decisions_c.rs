//! Spectral and Sample (light ports), Conv and Window (full ports) and the declared `Opaque` ops record a Decision at every
//! selection site (`Context::record_decisions`); the ports compute what they computed before; `LUMA_STRATEGY` refuses the light
//! ports and forces the strategies of the full ones where they fit.

use luma::*;

fn logged(ctx: &Context, f: impl FnOnce()) -> Vec<Decision> {
    ctx.record_decisions(true);
    f();
    ctx.sync().unwrap();
    let decisions = ctx.take_decisions();
    ctx.record_decisions(false);
    decisions
}

fn labels(decisions: &[Decision], site: DecisionSite) -> Vec<String> {
    decisions.iter().filter(|d| d.site == site).map(|d| d.chosen.to_string()).collect()
}

fn ramp(ctx: &Context, n: usize) -> Array<f32> {
    Array::<f32>::from_slice_on(ctx, &(0..n).map(|i| ((i * 7919) % 101) as f32 / 101.0).collect::<Vec<_>>(), n).unwrap()
}

fn image(ctx: &Context, w: usize, h: usize) -> Array<f32> {
    Array::<f32>::from_slice_on(ctx, &(0..w * h).map(|i| ((i * 31 + (i / w) * 17) % 255) as f32 / 255.0).collect::<Vec<_>>(), [w, h]).unwrap()
}

#[test]
fn spectral_logs_the_labels_of_its_branches() {
    let ctx = Context::new().unwrap();
    let complex = |n: usize| Array::<C32>::from_slice_on(&ctx, &vec![C32::new(1.0, 0.5); n], n).unwrap();
    let d = logged(&ctx, || {
        complex(64).fft().eval().unwrap();
        complex(37).fft().eval().unwrap();
    });
    assert_eq!(labels(&d, DecisionSite::SpectralPlan), ["spectral::stockham", "spectral::bluestein"], "{d:?}");
    assert!(d.iter().filter(|d| d.site == DecisionSite::SpectralPlan).all(|d| d.op == "fft" && d.location.file().contains("boundary_decisions_c")), "{d:?}");

    let d = logged(&ctx, || {
        ramp(&ctx, 64).rfft().eval().unwrap();
        ramp(&ctx, 37).rfft().eval().unwrap();
        ramp(&ctx, 6000).rfft().eval().unwrap();
    });
    let labels_of = labels(&d, DecisionSite::SpectralPlan);
    assert!(["spectral::real_fwd_fused", "spectral::real_fwd_staged"].contains(&labels_of[0].as_str()), "{labels_of:?}");
    assert!(labels_of.iter().any(|l| l == "spectral::real_fwd_direct"), "{labels_of:?}");
    assert!(labels_of.iter().any(|l| l == "spectral::bluestein"), "{labels_of:?}");

    let half = ramp(&ctx, 64).rfft();
    let d = logged(&ctx, || {
        half.irfft(64).eval().unwrap();
    });
    assert!(labels(&d, DecisionSite::SpectralPlan).iter().any(|l| l == "spectral::real_inv_fused"), "{d:?}");

    let d = logged(&ctx, || {
        ramp(&ctx, 100).fft_convolve1(&ramp(&ctx, 9), ConvMode::Full).eval().unwrap();
    });
    let spectral = labels(&d, DecisionSite::SpectralPlan);
    assert_eq!(spectral.iter().filter(|l| *l == "spectral::convolve").count(), 1, "{spectral:?}");
    assert!(spectral.iter().any(|l| l.starts_with("spectral::real_inv")), "{spectral:?}");
}

#[test]
fn sample_logs_the_read() {
    let ctx = Context::new().unwrap();
    let src = ramp(&ctx, 256);
    let pos = Array::<f32>::from_slice_on(&ctx, &[0.5, 3.25, 100.75], 3).unwrap();
    let d = logged(&ctx, || {
        src.approx1(&pos, Interp::Linear, 0.0).eval().unwrap();
    });
    assert_eq!(labels(&d, DecisionSite::SamplePlan), ["sample::buffer"], "{d:?}");
    assert!(d.iter().filter(|d| d.site == DecisionSite::SamplePlan).all(|d| d.op == "approx1"));
    let img = image(&ctx, 32, 24);
    let d = logged(&ctx, || {
        img.resize(16, 12, Interp::Linear).eval().unwrap();
    });
    let sample = labels(&d, DecisionSite::SamplePlan);
    assert!(sample.len() == 1 && ["sample::buffer", "sample::texture_gather"].contains(&sample[0].as_str()), "{sample:?}");
    let d = logged(&ctx, || {
        img.resize(16, 12, Interp::LinearHardware).eval().unwrap();
    });
    let sample = labels(&d, DecisionSite::SamplePlan);
    assert!(sample.len() == 1 && ["sample::buffer", "sample::texture_linear"].contains(&sample[0].as_str()), "{sample:?}");
}

#[test]
fn conv_and_window_log_their_strategies() {
    let ctx = Context::new().unwrap();
    let x = ramp(&ctx, 5000);
    let k = |n: usize| ramp(&ctx, n);
    let d = logged(&ctx, || {
        x.conv1d(&k(9), 1, 0).eval().unwrap();
        x.conv1d(&k(9), 5, 0).eval().unwrap();
        x.fir(&k(100)).eval().unwrap();
        x.fir(&k(300)).eval().unwrap();
        x.fir(&k(400)).eval().unwrap();
    });
    let conv = labels(&d, DecisionSite::ConvPlan);
    // The long FIR forms compose other convolutions: blocks of taps are `fir`s of 256 taps; the FFT form is a `conv1d_fft`.
    assert_eq!(conv[..5], ["direct_tiled", "direct", "fir_direct", "fir_blocks", "fir_direct"], "{conv:?}");
    assert!(conv.iter().any(|l| l == "fir_fft") && conv.iter().any(|l| l.starts_with("fft_")), "{conv:?}");
    assert!(d.iter().filter(|d| d.site == DecisionSite::ConvPlan).all(|d| d.location.file().contains("boundary_decisions_c") || d.location.file().ends_with("signal.rs")), "{d:?}");

    let d = logged(&ctx, || {
        x.conv1d_fft(&k(9)).eval().unwrap();
    });
    assert!(labels(&d, DecisionSite::ConvPlan).iter().all(|l| l == "fft_single" || l == "fft_blocked"), "{d:?}");

    let img = image(&ctx, 40, 30);
    let d = logged(&ctx, || {
        img.medfilt2([3, 3], Border::Edge).eval().unwrap();
        img.medfilt2([5, 10], Border::Edge).eval().unwrap();
        img.minfilt([5, 5], Border::Edge).eval().unwrap();
        img.minfilt([3, 3], Border::Edge).eval().unwrap();
        img.conv2(&image(&ctx, 3, 3), ConvSpec::new()).eval().unwrap();
        img.sobel_magnitude().eval().unwrap();
    });
    let window = labels(&d, DecisionSite::WindowPlan);
    assert_eq!(window, ["median_registers", "median_bitwise", "separable_extremum", "tiled", "tiled", "tiled", "tiled", "tiled"], "{window:?}");
}

fn product_of(ctx: &Context, dense: &[f32], xs: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<String>) {
    let a = sparse(&Array::<f32>::from_slice_on(ctx, dense, [cols, rows]).unwrap(), SparseFormat::Csr);
    let x = Array::<f32>::from_slice_on(ctx, xs, cols).unwrap();
    let mut y = Vec::new();
    let d = logged(ctx, || y = a.matmul(&x).to_vec().unwrap());
    (y, labels(&d, DecisionSite::OpaquePlan))
}

fn host_fir(x: &[f32], b: &[f32]) -> Vec<f32> {
    (0..x.len()).map(|i| (0..b.len().min(i + 1)).map(|j| f64::from(b[j]) * f64::from(x[i - j])).sum::<f64>() as f32).collect()
}

#[test]
fn the_fir_forms_agree() {
    let ctx = Context::new().unwrap();
    let n = 3000;
    let xs: Vec<f32> = (0..n).map(|i| ((i * 7919) % 101) as f32 / 101.0 - 0.5).collect();
    let bs: Vec<f32> = (0..400).map(|i| ((i * 31) % 17) as f32 / 170.0 - 0.05).collect();
    let want = host_fir(&xs, &bs);
    let x = Array::<f32>::from_slice_on(&ctx, &xs, n).unwrap();
    let b = Array::<f32>::from_slice_on(&ctx, &bs, bs.len()).unwrap();
    // One signal: the FFT.
    let fft = x.fir(&b).to_vec().unwrap();
    // Sixteen: blocks of taps.
    let batch: Vec<f32> = (0..16).flat_map(|_| xs.iter().copied()).collect();
    let xb = Array::<f32>::from_slice_on(&ctx, &batch, [n, 16]).unwrap();
    let d = logged(&ctx, || {
        let _ = xb.fir(&b).eval();
    });
    assert!(labels(&d, DecisionSite::ConvPlan).iter().any(|l| l == "fir_blocks"));
    let blocks = xb.fir(&b).to_vec().unwrap();
    for i in 0..n {
        assert!((fft[i] - want[i]).abs() < 2e-3, "fft {i}: {} {}", fft[i], want[i]);
        assert!((blocks[i] - want[i]).abs() < 1e-3, "blocks {i}: {} {}", blocks[i], want[i]);
        assert!((blocks[15 * n + i] - blocks[i]).abs() == 0.0);
    }
}

#[test]
fn strategies_of_the_new_families_are_forced_by_name() {
    let open = |text: &'static str| Context::with_options(ContextOptions::from_env().calibrate(false).strategy(text));
    // Light ports and unknown names are refused.
    for text in ["spectral:stockham", "sample:buffer"] {
        let error = open(text).map(drop).expect_err("not ported");
        assert!(matches!(&error, LumaError::InvalidOption(message) if message.contains("family not ported")), "{error}");
    }
    for text in ["conv:nonsense", "window:nonsense", "opaque:nonsense"] {
        assert!(matches!(open(text).map(drop), Err(LumaError::InvalidOption(_))), "{text}");
    }
    // A window kernel forced to `tiled` is the kernel the median selection runs.
    let default = Context::with_options(ContextOptions::from_env().calibrate(false).strategy("")).unwrap();
    let forced = open("window:tiled").unwrap();
    let want = image(&default, 40, 30).medfilt2([3, 3], Border::Edge).to_vec().unwrap();
    let img = image(&forced, 40, 30);
    let d = logged(&forced, || {
        assert_eq!(img.medfilt2([3, 3], Border::Edge).to_vec().unwrap(), want);
    });
    assert_eq!(labels(&d, DecisionSite::WindowPlan), ["tiled"]);
    // The sparse row kernels forced over a matrix whose long rows would take the merge kernel: the same product.
    let (rows, cols) = (40usize, 200usize);
    let dense: Vec<f32> = (0..rows * cols).map(|i| if (i % cols) < 2 + 3 * (i / cols) % 150 { ((i % 13) as f32 - 6.0) / 8.0 } else { 0.0 }).collect();
    let xs: Vec<f32> = (0..cols).map(|i| ((i % 7) as f32 - 3.0) / 4.0).collect();
    let product = |ctx: &Context| product_of(ctx, &dense, &xs, rows, cols);
    let (want, default_labels) = product(&default);
    assert!(default_labels.iter().any(|l| l == "opaque::sparse_spmv_merge"), "{default_labels:?}");
    let (got, forced_labels) = product(&open("opaque:sparse_rows").unwrap());
    assert!(forced_labels.iter().any(|l| l == "opaque::sparse_rows") && !forced_labels.iter().any(|l| l == "opaque::sparse_spmv_merge"), "{forced_labels:?}");
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() < 1e-3, "{a} {b}");
    }
}

#[test]
fn sparse_products_log_their_strategy_and_the_row_plan_once() {
    let ctx = Context::new().unwrap();
    // Rows of 64 / 65 entries: the merge kernel starts above SHORT_ROWS; 256 / 257: the long rows' pass above LONG_ROW.
    let build = |len: usize| {
        let (rows, cols) = (8usize, 300usize);
        let mut dense = vec![0.0f32; rows * cols];
        dense[..len].fill(1.0);
        for r in 1..rows {
            dense[r * cols] = 2.0;
        }
        sparse(&Array::<f32>::from_slice_on(&ctx, &dense, [cols, rows]).unwrap(), SparseFormat::Csr)
    };
    let x = Array::<f32>::from_slice_on(&ctx, &vec![1.0; 300], 300).unwrap();
    for (len, merge) in [(64, false), (65, true), (256, true), (257, true)] {
        let a = build(len);
        let d = logged(&ctx, || {
            assert_eq!(a.matmul(&x).to_vec().unwrap()[0], len as f32);
            assert_eq!(a.matmul(&x).to_vec().unwrap()[1], 2.0);
        });
        let opaque = labels(&d, DecisionSite::OpaquePlan);
        let want = if merge { "opaque::sparse_spmv_merge" } else { "opaque::sparse_rows" };
        assert_eq!(opaque.iter().filter(|l| l.contains("sparse_row_plan")).count(), 1, "len {len}: {opaque:?}");
        assert_eq!(opaque.iter().filter(|l| *l == want).count(), 2, "len {len}: {opaque:?}");
    }
    // Columns take the row kernels whatever the rows.
    let a = build(257);
    let xs = Array::<f32>::from_slice_on(&ctx, &vec![1.0; 600], [2, 300]).unwrap();
    let d = logged(&ctx, || {
        let _ = a.matmul(&xs).to_vec().unwrap();
    });
    assert!(labels(&d, DecisionSite::OpaquePlan).iter().any(|l| l == "opaque::sparse_rows"));
    // CSC `A x` is not native: the first product scatters, the second converts and selects again.
    let csc = a.to_format(SparseFormat::Csc);
    let d = logged(&ctx, || {
        let _ = csc.matmul_t(&Array::<f32>::from_slice_on(&ctx, &[1.0; 8], 8).unwrap()).to_vec().unwrap();
    });
    assert!(!labels(&d, DecisionSite::OpaquePlan).iter().any(|l| l == "opaque::sparse_scatter"), "CSC A^T x is native");
    let d = logged(&ctx, || {
        let _ = csc.matmul(&x).to_vec().unwrap();
        let _ = csc.matmul(&x).to_vec().unwrap();
    });
    let opaque = labels(&d, DecisionSite::OpaquePlan);
    assert_eq!(opaque.iter().filter(|l| *l == "opaque::sparse_scatter").count(), 1, "{opaque:?}");
    assert!(opaque.iter().any(|l| l == "opaque::sparse_rows" || l == "opaque::sparse_spmv_merge"), "{opaque:?}");
    // An empty matrix is the guard.
    let empty = Sparse::empty_on(&ctx, 4, 4, SparseFormat::Csr);
    let d = logged(&ctx, || {
        let _ = empty.matmul(&Array::<f32>::from_slice_on(&ctx, &[1.0; 4], 4).unwrap()).to_vec().unwrap();
    });
    assert_eq!(labels(&d, DecisionSite::OpaquePlan), ["opaque::sparse_empty"]);
}

/// Every declared op records its label when it runs, and none records anything while nobody asks.
#[test]
fn opaque_ops_are_all_declared() {
    let ctx = Context::new().unwrap();
    let img = image(&ctx, 32, 32);
    let rgb = Array::<f32>::from_slice_on(&ctx, &vec![0.5; 32 * 32 * 3], [32, 32, 3]).unwrap();
    let psf = Array::<f32>::from_slice_on(&ctx, &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [3, 3]).unwrap();
    let words = |n: usize| Array::<u32>::from_slice_on(&ctx, &(0..n as u32 * 8).collect::<Vec<_>>(), [8, n]).unwrap();
    let hist = bincount(&(&img * 255.0f32).to_u32(), 256);
    let pts = |f: fn(usize) -> f32| Array::<f32>::from_slice_on(&ctx, &(0..12).map(f).collect::<Vec<_>>(), 12).unwrap();
    let seeds = Array::<u32>::from_slice_on(&ctx, &[3], 1).unwrap();
    let dense = Array::<f32>::from_slice_on(&ctx, &[1.0, 0.0, 0.0, 2.0], [2, 2]).unwrap();
    let run = || {
        let _ = img.hist_equal(&hist).eval();
        let _ = img.sat().eval();
        let _ = img.moments().eval();
        let _ = img.unwrap([3, 3], [1, 1], [0, 0], true).eval();
        let _ = img.wrap([32, 32], [3, 3], [1, 1], [0, 0], true).eval();
        let _ = rgb.rgb2gray().eval();
        let _ = ramp(&ctx, 64).iir(&[1.0], &[1.0, -0.5]).eval();
        let _ = ramp(&ctx, 64).iir_sos(&[[1.0, 0.0, 0.0, 1.0, -0.5, 0.0]]).eval();
        let _ = img.fast(20.0, 9, true, 0.1, 3);
        let _ = img.harris(10, 0.0, 1.0, 0, 0.04);
        let _ = img.susan(3, 32.0, 10.0, 0.5, 3);
        let _ = img.dog(1, 2).eval();
        let _ = img.orb(20.0, 100, 1.5, 2, false);
        let _ = img.sift(3, 0.04, 10.0, 1.6, false, 1.0 / 255.0, 0.05);
        let _ = img.gloh(3, 0.04, 10.0, 1.6, false, 1.0 / 255.0, 0.05);
        let _ = img.canny(CannyThreshold::Manual, 0.1, 0.3, 3, false).eval();
        let _ = img.anisotropic_diffusion(0.1, 1.0, 2, FluxFunction::Exponential, DiffusionEq::Gradient).eval();
        let _ = rgb.mean_shift(2.0, 10.0, 2, true).eval();
        let _ = img.match_template(&psf, MatchType::Sad).eval();
        let _ = nearest_neighbour(&img, &img, 0, 1, DistType::Ssd);
        let _ = hamming_matcher(&words(4), &words(4), 1, 1);
        let _ = img.gt(0.5f32).regions(Connectivity::Four).eval();
        let _ = img.confidence_cc(&seeds, &seeds, 1, 1, 1, 1.0).eval();
        let _ = img.iterative_deconv(&psf, 2, 0.5, IterativeDeconvAlgo::Landweber).eval();
        let _ = homography(&pts(|i| i as f32), &pts(|i| (i * i % 7) as f32), &pts(|i| i as f32 + 1.0), &pts(|i| (i * i % 7) as f32), HomographyMethod::Ransac, 1.0, 10);
        let a = sparse(&dense, SparseFormat::Csr);
        let _ = a.to_format(SparseFormat::Csc);
        let _ = a.mul_dense(&dense);
    };
    // Nobody records: nothing is built.
    ctx.record_decisions(true);
    ctx.take_decisions();
    ctx.record_decisions(false);
    run();
    ctx.sync().unwrap();
    ctx.record_decisions(true);
    assert!(ctx.take_decisions().iter().all(|d| d.site != DecisionSite::OpaquePlan), "decisions were recorded while nobody asked");
    ctx.record_decisions(false);

    let d = logged(&ctx, run);
    let seen = labels(&d, DecisionSite::OpaquePlan);
    for label in [
        "image_pointwise", "image_sat", "image_moments", "image_unwrap", "image_wrap", "image_hist_equal", "signal_iir", "signal_iir_sos", "vision_fast", "vision_harris",
        "vision_susan", "vision_dog", "vision_orb", "vision_sift", "vision_gloh", "vision_canny", "vision_diffusion", "vision_mean_shift", "vision_match_template",
        "vision_nearest_neighbour", "vision_hamming", "vision_regions", "vision_confidence_cc", "vision_deconv", "vision_homography", "sparse_convert", "sparse_elementwise",
    ] {
        let label = format!("opaque::{label}");
        assert!(seen.contains(&label), "{label} not recorded: {seen:?}");
    }
}
