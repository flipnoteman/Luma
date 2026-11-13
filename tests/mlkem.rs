//! The lazy ML-KEM of `examples/mlkem.rs` (KeyGen, Encaps and Decaps in array operations only):
//! byte for byte against the RustCrypto `ml-kem` crate on random instances (tampered ciphertexts
//! included), against the NIST ACVP vectors, and the rejection sampling's fallback to more blocks.

#[allow(dead_code)]
#[path = "../examples/mlkem.rs"]
mod mlkem;

use luma::*;
use mlkem::{complete, download, params, upload, verify, verify_acvp};

#[test]
fn random_instances_match_the_ml_kem_crate() {
    let ctx = Context::new().unwrap();
    for p in params::ALL {
        verify(&ctx, p, 48).unwrap();
    }
}

#[test]
fn acvp_vectors_match() {
    verify_acvp(&Context::new().unwrap()).unwrap();
}

/// KeyGen of `n` instances of ML-KEM-512 starting with `blocks` squeezed blocks per matrix entry,
/// against the reference; returns the blocks it ended with.
fn keygen_from(blocks: usize) -> usize {
    let ctx = Context::new().unwrap();
    let p = params::ML_KEM_512;
    let n = 16;
    let d: Vec<Vec<u8>> = (0..n).map(|i| (0..32).map(|b| (i * 31 + b * 7 + 1) as u8).collect()).collect();
    let z: Vec<Vec<u8>> = (0..n).map(|i| (0..32).map(|b| (i * 13 + b * 11 + 5) as u8).collect()).collect();
    let mut kem = mlkem::Kem::new(&ctx, p, n).unwrap();
    kem.blocks = blocks;
    let (d_w, z_w) = (upload(&ctx, &d).unwrap(), upload(&ctx, &z).unwrap());
    let (ek, _) = complete(&mut kem, |kem| kem.keygen(&d_w, &z_w), |(ek, dk)| eval_all(&[ek, dk])).unwrap();
    for (i, ek) in download(&ek).unwrap().iter().enumerate() {
        assert_eq!(*ek, mlkem::host::keygen(p, d[i].as_slice().try_into().unwrap(), z[i].as_slice().try_into().unwrap()).ek, "instance {i}");
    }
    kem.blocks
}

#[test]
fn sampling_squeezes_more_blocks_until_every_polynomial_is_complete() {
    // One block of 168 bytes holds at most 112 candidates, fewer than the 256 coefficients: the
    // operation is repeated with 2 and then 4 blocks.
    assert!(keygen_from(1) >= 4);
}

#[test]
fn sampling_from_more_blocks_than_needed_gives_the_same_polynomials() {
    // 8 and 16 blocks: groups start at slots past what the packed slot count holds.
    assert_eq!(keygen_from(8), 8);
    assert_eq!(keygen_from(16), 16);
}
