//! `cumsum` of a pending expression computes it in the scan's own kernels: no elementwise kernel writes a mask or a cast that
//! only the scan reads (decision W2-U3a S5), and the result is bit for bit that of the scan of the evaluated array.

use luma::*;

use crate::common::values;

/// Every kernel launched while `run` runs, by label.
fn kernels(ctx: &Context, run: impl FnOnce()) -> Vec<String> {
    let ((), profile) = ctx.profile(run).unwrap();
    profile.kernels.iter().map(|k| k.label.to_string()).collect()
}

fn scan_labels(ctx: &Context, run: impl FnOnce()) -> Vec<String> {
    ctx.record_decisions(true);
    run();
    ctx.sync().unwrap();
    let decisions = ctx.take_decisions();
    ctx.record_decisions(false);
    decisions.iter().filter(|d| d.site == DecisionSite::ScanPlan).map(|d| d.chosen.to_string()).collect()
}

/// (shape, axis): a short line, one chunk, several chunks, long lines (a warp or look-back where the device has them), and strided
/// lines with and without blocks.
const SHAPES: [([usize; 2], usize); 7] = [([100, 3], 0), ([700, 1], 0), ([5000, 1], 0), ([4096, 130], 0), ([1 << 20, 1], 0), ([20, 200], 1), ([4, 1500], 1)];

fn bits<T: Copy>(values: &[T], to_bits: &impl Fn(T) -> u32) -> Vec<u32> {
    values.iter().map(|&v| to_bits(v)).collect()
}

#[test]
fn scan_of_a_pending_expression_does_not_materialise_it() {
    let ctx = Context::new().unwrap();
    for (dims, axis) in SHAPES {
        let n = dims[0] * dims[1];
        let host = values(n, 7, -1.0, 1.0);
        let f = Array::<f32>::from_slice_on(&ctx, &host, dims).unwrap();
        let i = Array::<i32>::from_slice_on(&ctx, &host.iter().map(|&v| (v * 1000.0) as i32).collect::<Vec<_>>(), dims).unwrap();
        let u = Array::<u32>::from_slice_on(&ctx, &host.iter().map(|&v| ((v + 1.0) * 1000.0) as u32).collect::<Vec<_>>(), dims).unwrap();
        // The same data in other buffers, so that the evaluated reference does not share expressions with the pending operand.
        let f2 = Array::<f32>::from_slice_on(&ctx, &host, dims).unwrap();
        let i2 = Array::<i32>::from_slice_on(&ctx, &host.iter().map(|&v| (v * 1000.0) as i32).collect::<Vec<_>>(), dims).unwrap();
        let u2 = Array::<u32>::from_slice_on(&ctx, &host.iter().map(|&v| ((v + 1.0) * 1000.0) as u32).collect::<Vec<_>>(), dims).unwrap();
        for array in [&f, &f2] {
            array.eval().unwrap();
        }
        for array in [&i, &i2] {
            array.eval().unwrap();
        }
        for array in [&u, &u2] {
            array.eval().unwrap();
        }
        ctx.sync().unwrap();

        // The operands: a scaled float (fixed grouping: the sums round as they do through the evaluated array), a signed integer
        // expression, and the cast of a mask, which is what feeds the cumsum of a factorize.
        let want = (&f2 * 2.0f32).cumsum(axis).to_vec().unwrap();
        check(&ctx, "f32", dims, axis, || &f * 2.0f32, &want, f32::to_bits);
        let want = (&i2 * 3 + 1).cumsum(axis).to_vec().unwrap();
        check(&ctx, "i32", dims, axis, || &i * 3 + 1, &want, |v| v as u32);
        let want = u2.gt(1000u32).to_u32().cumsum(axis).to_vec().unwrap();
        check(&ctx, "u32 mask", dims, axis, || u.gt(1000u32).to_u32(), &want, |v| v);

        let again = u.gt(1000u32).to_u32();
        let labels = scan_labels(&ctx, || {
            again.cumsum(axis).eval().unwrap();
        });
        assert!(labels.iter().any(|label| label.ends_with("prologue=fused")), "{dims:?} axis {axis}: {labels:?}");
        // An evaluated operand is read as it is.
        let evaluated = u2.gt(1000u32).to_u32();
        evaluated.eval().unwrap();
        let labels = scan_labels(&ctx, || {
            evaluated.cumsum(axis).eval().unwrap();
        });
        assert!(!labels.is_empty() && labels.iter().all(|label| !label.contains("prologue")), "{labels:?}");
    }
}

/// The cumsum of the pending operand `pending()` is `want`, and the scan's kernels are the only ones that run.
fn check<T: Numeric + Copy + std::fmt::Debug>(ctx: &Context, name: &str, dims: [usize; 2], axis: usize, pending: impl FnOnce() -> Array<T>, want: &[T], to_bits: impl Fn(T) -> u32) {
    ctx.sync().unwrap();
    let mut got = Vec::new();
    let launched = kernels(ctx, || {
        let operand = pending();
        got = operand.cumsum(axis).to_vec().unwrap();
    });
    assert_eq!(bits(&got, &to_bits), bits(want, &to_bits), "{name} {dims:?} axis {axis}");
    assert!(launched.iter().all(|label| label.starts_with("Scan") || label.starts_with("FusedAxisReduce") || label == "Fill"), "{name} {dims:?} axis {axis}: {launched:?}");
    assert!(launched.iter().any(|label| label.starts_with("FusedAxisReduce")), "{name} {dims:?} axis {axis}: {launched:?}");
}

/// A write into the operand after the scan was made does not change the scan; one before it does.
#[test]
fn scan_prologue_respects_readers() {
    let ctx = Context::new().unwrap();
    let n = 5000;
    let host: Vec<u32> = (0..n as u32).map(|i| i % 7).collect();
    let mut x = Array::<u32>::from_slice_on(&ctx, &host, n).unwrap();
    let before = x.gt(2u32).to_u32().cumsum(0);
    x.write(&vec![0u32; n]).unwrap();
    let after = x.gt(2u32).to_u32().cumsum(0);
    let mut want = 0u32;
    let want: Vec<u32> = host
        .iter()
        .map(|&v| {
            want += u32::from(v > 2);
            want
        })
        .collect();
    assert_eq!(before.to_vec().unwrap(), want);
    assert!(after.to_vec().unwrap().iter().all(|&v| v == 0));
}

/// The sum of a pending operand is the only fused scan: another operator reads a buffer.
#[test]
fn only_sums_of_pending_operands_are_fused() {
    let ctx = Context::new().unwrap();
    let host = values(3000, 3, 0.0, 1.0);
    let x = Array::<f32>::from_slice_on(&ctx, &host, 3000).unwrap();
    x.eval().unwrap();
    let labels = scan_labels(&ctx, || {
        let got = (&x * 2.0f32).scan(0, |a, b| a.maximum(b)).to_vec().unwrap();
        assert!(got.iter().zip(&host).all(|(&g, &h)| g >= 2.0 * h));
    });
    assert!(labels.iter().all(|label| !label.contains("prologue")), "{labels:?}");
}
