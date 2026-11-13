//! The lazy Keccak of `examples/keccak.rs` (SHA3-256 and SHAKE128 in array operations only):
//! known answers, a random batch against the plain-Rust sponge, and that a chain of several
//! permutations (30 000 operations) is cut into a few kernels, not one per operation made after
//! a cut.

#[allow(dead_code)]
#[path = "../examples/keccak.rs"]
mod keccak;

use keccak::{bytes, instance_bytes, message_words, read_all, sponge, sponge_host};
use luma::*;

#[test]
fn sha3_256_and_shake128_match_the_reference() {
    let ctx = Context::new().unwrap();
    let n = 64;
    for (rate, suffix, out_len, len) in [(136, 0x06u8, 32, 0usize), (136, 0x06, 32, 135), (136, 0x06, 32, 300), (168, 0x1F, 200, 33)] {
        let messages: Vec<Vec<u8>> = (0..n).map(|i| bytes(i as u64 * 31 + len as u64 + 1, len)).collect();
        let words = message_words(&ctx, &messages, len).unwrap();
        let got = read_all(&sponge(&ctx, n, &words, len, rate, suffix, out_len).unwrap()).unwrap();
        for (i, message) in messages.iter().enumerate() {
            assert_eq!(instance_bytes(&got, i, out_len), sponge_host(message, rate, suffix, out_len), "{len}-byte message {i}");
        }
    }
    let empty = read_all(&sponge(&ctx, 1, &[], 0, 136, 0x06, 32).unwrap()).unwrap();
    let hex: String = instance_bytes(&empty, 0, 32).iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(hex, "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a");
}

#[test]
fn a_chain_of_permutations_is_a_few_kernels() {
    let ctx = Context::new().unwrap();
    // A device that binds few buffers (llvmpipe: 16) cannot take the 50 words of the state in one kernel.
    if !ctx.profiling_supported() || ctx.tuning_profile() == TuningProfile::Cpu {
        return;
    }
    let (n, len) = (256, 1000);
    let messages: Vec<Vec<u8>> = (0..n).map(|i| bytes(i as u64 + 5, len)).collect();
    let words = message_words(&ctx, &messages, len).unwrap();
    // Eight permutations, absorbed one block each.
    let (got, profile) = ctx.profile(|| read_all(&sponge(&ctx, n, &words, len, 136, 0x06, 32).unwrap()).unwrap()).unwrap();
    for (i, message) in messages.iter().enumerate().step_by(37) {
        assert_eq!(instance_bytes(&got, i, 32), sponge_host(message, 136, 0x06, 32));
    }
    assert!(profile.launches() <= 60, "{} kernels for 8 permutations\n{profile}", profile.launches());
}
