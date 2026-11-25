//! Launch counts of the sorts and scans (CHANGELOG 0.5.x: "6 launches instead of 16 for a sort"): recorded on the code before the
//! strategy registry took over their selection and asserted after every port. A port that changes a launch count fails here.

use luma::*;

fn launches(ctx: &Context, run: impl FnOnce()) -> Option<u32> {
    if !ctx.profiling_supported() {
        return None;
    }
    let ((), profile) = ctx.profile(run).unwrap();
    Some(profile.launches())
}

fn random(n: usize) -> Vec<u32> {
    let mut state = 0x9e37_79b9_u32;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        })
        .collect()
}

/// A flat sort and an argsort of 10M `f32`.
#[test]
fn sort_launch_counts() {
    let ctx = Context::new().unwrap();
    let keys: Vec<f32> = random(10_000_000).into_iter().map(|bits| (bits >> 8) as f32).collect();
    let x = Array::from_slice_on(&ctx, &keys, keys.len()).unwrap();
    x.eval().unwrap();
    ctx.sync().unwrap();
    let flat = launches(&ctx, || {
        x.sort().eval().unwrap();
        ctx.sync().unwrap();
    });
    let arg = launches(&ctx, || {
        x.argsort().eval().unwrap();
        ctx.sync().unwrap();
    });
    // Recorded on the code before the registry port (RTX 3090 / CPU backend, `Profile::launches`).
    let want = match ctx.backend() {
        BackendKind::Cuda => (6, 6),
        BackendKind::Wgpu => (20, 20),
        BackendKind::Cpu => (28, 28),
        _ => return,
    };
    assert_eq!((flat, arg), (Some(want.0), Some(want.1)), "{:?}", ctx.backend());
}

#[test]
fn scan_launch_counts() {
    let ctx = Context::new().unwrap();
    let x = Array::<u32>::from_slice_on(&ctx, &random(1 << 20).into_iter().map(|v| v & 0xff).collect::<Vec<_>>(), 1 << 20).unwrap();
    x.eval().unwrap();
    ctx.sync().unwrap();
    let flat = launches(&ctx, || {
        x.cumsum(0).eval().unwrap();
        ctx.sync().unwrap();
    });
    let strided = Array::<u32>::from_slice_on(&ctx, &random(4096 * 4096).into_iter().map(|v| v & 0xff).collect::<Vec<_>>(), [4096, 4096]).unwrap();
    strided.eval().unwrap();
    ctx.sync().unwrap();
    let along_1 = launches(&ctx, || {
        strided.cumsum(1).eval().unwrap();
        ctx.sync().unwrap();
    });
    // `cumsum` of 1 << 20 `u32`: the zeroing launch and the look-back scan on CUDA, three on the others; strided: fold, the scan of
    // the block totals, walk.
    let want = match ctx.backend() {
        BackendKind::Cuda => (2, 3),
        BackendKind::Wgpu | BackendKind::Cpu => (3, 3),
        _ => return,
    };
    assert_eq!((flat, along_1), (Some(want.0), Some(want.1)), "{:?}", ctx.backend());
}
