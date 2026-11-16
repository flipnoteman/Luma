//! A device that binds few storage buffers to a kernel (llvmpipe and many integrated GPUs: 16,
//! Metal: 29) used to run the lazy Keccak of `examples/keccak.rs` in thousands of kernels: its
//! state is 50 arrays that every round reads, so every kernel that computes a round needs 50
//! bindings, and the expression was cut one operation at a time. The arrays are now packed into
//! one buffer that kernels read through one binding. This runs the sponge with the limit of a
//! 16-binding device (`LUMA_FUSION_MAX_INPUTS`) and checks the results and the kernel count.

#[allow(dead_code)]
#[path = "../examples/keccak.rs"]
mod keccak;

use keccak::{bytes, instance_bytes, message_words, read_all, sponge, sponge_host};
use luma::*;

#[test]
fn the_sponge_on_a_device_with_16_bindings_is_a_few_dozen_kernels() {
    // Nothing else runs in this process yet: the limit is read when the context is created.
    unsafe { std::env::set_var("LUMA_FUSION_MAX_INPUTS", "13") };
    let ctx = Context::new().unwrap();
    let (n, len) = (64, 300);
    let messages: Vec<Vec<u8>> = (0..n).map(|i| bytes(i as u64 + 9, len)).collect();
    let words = message_words(&ctx, &messages, len).unwrap();
    // Three permutations.
    let run = || read_all(&sponge(&ctx, n, &words, len, 136, 0x06, 32).unwrap()).unwrap();
    let check = |got: &[Vec<u32>]| {
        for (i, message) in messages.iter().enumerate().step_by(5) {
            assert_eq!(instance_bytes(got, i, 32), sponge_host(message, 136, 0x06, 32), "message {i}");
        }
    };
    if !ctx.profiling_supported() {
        check(&run());
        return;
    }
    let (got, profile) = ctx.profile(run).unwrap();
    check(&got);
    // 7120 kernels without packing.
    assert!(profile.launches() <= 120, "{} kernels for 3 permutations\n{profile}", profile.launches());
    // And again, over the arrays that are windows now.
    check(&run());
}
