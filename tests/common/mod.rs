#![allow(dead_code)]

/// Deterministic pseudo-random values in `lo..hi` (a small LCG; no dependencies).
pub fn values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / (1u64 << 24) as f32;
            lo + unit * (hi - lo)
        })
        .collect()
}

/// Asserts `got` is within `rel` (relative) or `abs` (absolute) of `want`, elementwise.
pub fn assert_close(got: &[f32], want: &[f32], rel: f32, abs: f32) {
    assert_eq!(got.len(), want.len(), "length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tolerance = abs.max(rel * w.abs());
        assert!(
            (g - w).abs() <= tolerance,
            "index {i}: got {g}, want {w} (tolerance {tolerance})"
        );
    }
}
