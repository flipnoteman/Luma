//! The reference verification harness (W3-R7 §5) against the real kernels: every variant of the baseline units passes (a harness that
//! rejects a good kernel would disable good variants), a broken run fails, a verdict persists across a reopen, and a cold and a
//! warm verifier give the same answer.

use luma_core::hal::verify::UnitVerifier;
use luma_core::{Backend, GemmLanes, GemmSpec, GemmUnit};
use luma_cuda::CudaBackend;

#[test]
fn every_baseline_variant_passes_and_a_broken_one_fails() {
    let dir = std::env::temp_dir().join(format!("luma-unit-verify-cuda-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // SAFETY: set before any thread of this process reads it; this binary has one test.
    unsafe { std::env::set_var("LUMA_CACHE_DIR", &dir) };
    let backend = match CudaBackend::new() {
        Ok(backend) => backend,
        Err(error) => {
            println!("SKIPPED: no CUDA device ({error})");
            return;
        }
    };
    if !backend.gemm_ragged() {
        println!("SKIPPED: the device has no ragged matrix products");
        return;
    }
    let cold = UnitVerifier::new();
    let mut specs = Vec::new();
    for unit in [GemmUnit::Fma, GemmUnit::Tf32] {
        for (index, &(rows, cols)) in backend.gemm_tiles(unit).iter().enumerate() {
            for lanes in [GemmLanes::Vec4, GemmLanes::Vec4Tails, GemmLanes::Scalar] {
                for (lhs_t, rhs_t) in [(false, false), (true, false), (false, true), (true, true)] {
                    // Every orientation of every tile for the first two tiles; the other tiles on the transposed pair of lanes.
                    if index >= 2 && lhs_t != rhs_t {
                        continue;
                    }
                    specs.push(GemmSpec::new(rows, cols, lhs_t, rhs_t, unit).with_lanes(lanes));
                }
            }
        }
    }
    for spec in &specs {
        let (verdict, waited) = cold.verdict(&backend, spec);
        assert!(verdict.ok, "{spec:?} failed its verification: {verdict:?}");
        assert!(waited.is_some_and(|time| time.as_millis() < 5_000), "{spec:?}: the first call verifies");
        assert!(verdict.max_rel_err < 16.0 / 16_777_216.0 || spec.unit == GemmUnit::Tf32, "{spec:?}: {verdict:?}");
        // The same verifier knows it now, and so does a new one reading the disk.
        assert_eq!(cold.verdict(&backend, spec), (verdict, None));
        assert_eq!(UnitVerifier::new().known(&backend, spec), Some(verdict), "{spec:?}: stored on disk");
    }
    // A cold verifier with an empty cache and a warm one agree.
    let _ = std::fs::remove_dir_all(&dir);
    let spec = specs[0];
    assert_eq!(UnitVerifier::new().verdict(&backend, &spec).0, cold.verdict(&backend, &spec).0);

    // A kernel with a flipped fragment index: columns swapped, an output wrong.
    let broken = UnitVerifier::new();
    broken.set_tamper(Some(std::sync::Arc::new(|_, out: &mut [f32]| out.swap(400, 401))));
    let spec = GemmSpec::new(64, 64, true, true, GemmUnit::Tf32);
    assert!(!broken.verdict(&backend, &spec).0.ok);
    assert_eq!(UnitVerifier::new().known(&backend, &spec), None, "a tampered verdict is not kept");
    let _ = std::fs::remove_dir_all(&dir);
}
