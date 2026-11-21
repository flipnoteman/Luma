//! Word loops of the node id sets (see [`super::Node::ids`]), which every operation made pays
//! for once per word of the set. The generic code of a build for the baseline x86-64 has no
//! population count instruction and works on two words at a time; where the processor has them
//! the same loops are compiled for POPCNT and AVX2 and picked once per call.
//!
//! The sets are also what `create` allocates most of: one of the size of the expression below
//! the node, per node, freed when the node is. Expressions made again and again (a hash of a
//! new batch, a step of a loop) ask for the sizes the last one gave back, so the buffers of
//! dropped nodes are kept for the next ones, within a budget.

use std::cell::RefCell;

/// Most words (8 bytes each) of buffers a thread keeps for later nodes.
const POOL_WORDS: usize = 1 << 20;
/// Longest buffer worth keeping; longer ones (an expression of more than 64 * this many ids
/// below a node) are rare and the allocator is the better judge of them.
const POOL_MAX_LEN: usize = 1024;

/// Buffers of dropped nodes by length.
struct Pool {
    by_len: Vec<Vec<Box<[u64]>>>,
    words: usize,
}

thread_local! {
    static POOL: RefCell<Pool> = const { RefCell::new(Pool { by_len: Vec::new(), words: 0 }) };
}

/// A zeroed buffer of `len` words: one a dropped node gave back, or a new one.
pub(super) fn zeroed(len: usize) -> Box<[u64]> {
    let recycled = if len <= POOL_MAX_LEN {
        POOL.try_with(|pool| {
            let mut pool = pool.borrow_mut();
            let buffer = pool.by_len.get_mut(len)?.pop()?;
            pool.words -= len;
            Some(buffer)
        })
        .ok()
        .flatten()
    } else {
        None
    };
    match recycled {
        Some(mut buffer) => {
            buffer.fill(0);
            buffer
        }
        None => vec![0; len].into_boxed_slice(),
    }
}

/// Keeps the buffer of a node that is dropped for [`zeroed`], unless the thread holds enough.
pub(super) fn recycle(buffer: Box<[u64]>) {
    let len = buffer.len();
    if len == 0 || len > POOL_MAX_LEN {
        return;
    }
    // `try_with`: nodes are dropped while a thread ends too.
    let _ = POOL.try_with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.words + len > POOL_WORDS {
            return;
        }
        if pool.by_len.len() <= len {
            pool.by_len.resize_with(len + 1, Vec::new);
        }
        pool.words += len;
        if let Some(bucket) = pool.by_len.get_mut(len) {
            bucket.push(buffer);
        }
    });
}

/// `dst |= src` word by word, over the shorter of the two.
#[inline]
pub(super) fn or_into(dst: &mut [u64], src: &[u64]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the processor has AVX2.
        return unsafe { or_into_avx2(dst, src) };
    }
    or_into_generic(dst, src);
}

/// The number of bits set in `words`.
#[inline]
pub(super) fn count_ones(words: &[u64]) -> usize {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("popcnt") {
        // SAFETY: the processor has POPCNT.
        return unsafe { count_ones_popcnt(words) };
    }
    count_ones_generic(words)
}

#[inline(always)]
fn or_into_generic(dst: &mut [u64], src: &[u64]) {
    for (word, &bits) in dst.iter_mut().zip(src) {
        *word |= bits;
    }
}

#[inline(always)]
fn count_ones_generic(words: &[u64]) -> usize {
    words.iter().map(|word| word.count_ones() as usize).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn or_into_avx2(dst: &mut [u64], src: &[u64]) {
    or_into_generic(dst, src);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
unsafe fn count_ones_popcnt(words: &[u64]) -> usize {
    count_ones_generic(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn or_and_count_agree_with_the_generic_loops() {
        let a: Vec<u64> = (0..37u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i << 7)).collect();
        let b: Vec<u64> = (0..21u64).map(|i| i.wrapping_mul(0xD6E8_FEB8_6659_FD93).rotate_left(13)).collect();
        let (mut fast, mut slow) = (a.clone(), a);
        or_into(&mut fast, &b);
        or_into_generic(&mut slow, &b);
        assert_eq!(fast, slow);
        assert_eq!(count_ones(&fast), count_ones_generic(&slow));
        assert_eq!(count_ones(&[]), 0);
    }
}
