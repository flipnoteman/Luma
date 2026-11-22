//! A profile of more launches than the live query sets of the profiler hold (`LIVE_SETS` sets of 256 launches) is read back in
//! pieces while it is recorded, without losing a timestamp.

use luma::*;

#[test]
fn profile_counts_every_launch_beyond_the_live_query_sets() {
    let ctx = Context::new().unwrap();
    if !ctx.profiling_supported() {
        return;
    }
    let launches = 9000;
    let x = Array::from_slice_on(&ctx, &[1.0f32; 64], 64).unwrap();
    let (y, profile) = ctx
        .profile(|| {
            let mut y = x.clone();
            for _ in 0..launches {
                y = (&y + 1.0).cache();
            }
            y
        })
        .unwrap();
    assert_eq!(profile.launches(), launches, "{profile}");
    assert!(profile.kernels.iter().all(|k| k.launches > 0 && k.min <= k.max && k.total >= k.max), "{profile}");
    assert_eq!(y.to_vec().unwrap(), vec![1.0 + launches as f32; 64]);
}
