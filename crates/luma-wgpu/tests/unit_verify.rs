//! The cooperative-matrix `f16` unit (W3-R7) against the reference verification harness: every tile, orientation, lane and output
//! passes (a layout the hardware reads wrongly is what xla#51016 hid), and a broken run fails.

use luma_core::hal::verify::UnitVerifier;
use luma_core::{Backend, DeviceKind, GemmLanes, GemmOut, GemmSpec, GemmUnit};
use luma_wgpu::backend::WgpuBackend;

#[test]
fn every_coop_variant_passes_its_verification() {
    let dir = std::env::temp_dir().join(format!("luma-unit-verify-coop-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // SAFETY: set before the backend opens and before any other thread of this binary reads it; this binary has one test.
    unsafe { std::env::set_var("LUMA_CACHE_DIR", &dir) };
    let backend = match WgpuBackend::new(DeviceKind::Gpu) {
        Ok(backend) if !backend.gemm_tiles(GemmUnit::CoopF16).is_empty() => backend,
        Ok(_) => {
            println!("SKIPPED: the device has no cooperative-matrix f16 unit");
            return;
        }
        Err(error) => {
            println!("SKIPPED: no GPU ({error})");
            return;
        }
    };
    let verifier = UnitVerifier::new();
    let mut count = 0;
    let mut failures = Vec::new();
    for &(rows, cols) in backend.gemm_tiles(GemmUnit::CoopF16) {
        for lanes in [GemmLanes::Vec4, GemmLanes::Vec4Tails] {
            for (lhs_t, rhs_t) in [(false, false), (true, false), (false, true), (true, true)] {
                for out in [GemmOut::F32, GemmOut::F16] {
                    let spec = GemmSpec::new(rows, cols, lhs_t, rhs_t, GemmUnit::CoopF16).with_lanes(lanes).with_out(out);
                    if out == GemmOut::F16 && lanes == GemmLanes::Vec4Tails {
                        assert!(backend.compile_gemm(&spec, None).is_none(), "a half store of a ragged product is not taken");
                        continue;
                    }
                    let (verdict, waited) = verifier.verdict(&backend, &spec);
                    count += 1;
                    if !verdict.ok || waited.is_none() {
                        failures.push(format!("{spec:?}: {verdict:?}"));
                    }
                    assert_eq!(UnitVerifier::new().known(&backend, &spec), Some(verdict), "{spec:?}: stored on disk");
                }
            }
        }
    }
    println!("{count} variants verified");
    // The harness is not vacuous: a swapped pair of outputs fails.
    let _ = std::fs::remove_dir_all(&dir);
    let broken = UnitVerifier::new();
    broken.set_tamper(Some(std::sync::Arc::new(|_, out: &mut [f32]| out.swap(400, 401))));
    assert!(!broken.verdict(&backend, &GemmSpec::new(64, 64, false, true, GemmUnit::CoopF16).with_out(GemmOut::F16)).0.ok);
    // A kernel that is right to the last ulp but not the same in two launches (a race inside a workgroup) fails: the tamper flips one low bit
    // in every second run.
    let racy = UnitVerifier::new();
    let calls = std::sync::atomic::AtomicUsize::new(0);
    racy.set_tamper(Some(std::sync::Arc::new(move |_, out: &mut [f32]| {
        if calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 2 == 1 {
            out[300] = f32::from_bits(out[300].to_bits() ^ 1);
        }
    })));
    assert!(!racy.verdict(&backend, &GemmSpec::new(64, 64, false, true, GemmUnit::CoopF16).with_out(GemmOut::F16)).0.ok, "launches that differ are not verified");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(failures.is_empty(), "{} of {count} failed:\n{}", failures.len(), failures.join("\n"));
}
