use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// A fast, non-cryptographic hasher (the multiply-rotate scheme of rustc's `FxHasher`) for the
/// context's caches, whose keys are small integers, buffer handles and kernel sources. The
/// default `SipHash` costs several times as much per lookup, which adds up on the launch path.
#[derive(Default, Clone, Copy)]
pub(crate) struct FxHasher(u64);

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let word = chunk
                .iter()
                .rev()
                .fold(0u64, |word, &byte| (word << 8) | u64::from(byte));
            self.add(word);
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    /// The multiplication leaves the best-mixed bits at the top; hash tables index with the low
    /// ones, and handles are aligned, so rotate them down.
    #[inline]
    fn finish(&self) -> u64 {
        self.0.rotate_left(26)
    }
}

pub(crate) type FxBuild = BuildHasherDefault<FxHasher>;

pub(crate) type FxMap<K, V> = HashMap<K, V, FxBuild>;
