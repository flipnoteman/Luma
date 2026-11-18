//! Host memory copies shared by every backend's upload and readback paths.

use std::mem::MaybeUninit;

/// Copies below this size stay on the calling thread.
const PARALLEL_COPY_MIN: usize = 8 << 20;

/// `dst.copy_from_slice(src)` split over several threads for large copies: one core cannot
/// saturate unified memory, and the page faults of a fresh destination spread out too. The
/// threads store past the cache (see [`copy_streaming`]).
pub fn copy_parallel(src: &[u8], dst: &mut [MaybeUninit<u8>]) {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
    if src.len() < PARALLEL_COPY_MIN || threads == 1 {
        dst.write_copy_of_slice(src);
        return;
    }
    let chunk = src.len().div_ceil(threads).next_multiple_of(4096);
    std::thread::scope(|scope| {
        for (s, d) in src.chunks(chunk).zip(dst.chunks_mut(chunk)) {
            scope.spawn(move || copy_streaming(s, d));
        }
    });
}

/// `dst.copy_from_slice(src)` with non-temporal stores: the destination lines are not read into
/// the cache before being overwritten, which a large readback (bound by host memory bandwidth
/// while the GPU's transfer writes the next chunk) would pay for with a third of its traffic.
#[cfg(target_arch = "x86_64")]
pub fn copy_streaming(src: &[u8], dst: &mut [MaybeUninit<u8>]) {
    use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_sfence, _mm_stream_si128};
    const LINE: usize = 64;
    assert_eq!(src.len(), dst.len(), "source and destination lengths differ");
    let head = dst.as_ptr().align_offset(LINE).min(dst.len());
    let (dst_head, dst_body) = dst.split_at_mut(head);
    let (src_head, src_body) = src.split_at(head);
    dst_head.write_copy_of_slice(src_head);
    let (src_lines, src_tail) = src_body.as_chunks::<LINE>();
    let (dst_lines, dst_tail) = dst_body.as_chunks_mut::<LINE>();
    for (s, d) in src_lines.iter().zip(dst_lines) {
        for part in (0..LINE).step_by(16) {
            // SAFETY: `part + 16 <= LINE`, so both pointers stay inside their 64-byte chunks;
            // the load is unaligned, and `d` is 64-byte aligned (`head`), as the store requires.
            // SSE2 is part of the x86_64 baseline.
            unsafe {
                let value = _mm_loadu_si128(s.as_ptr().add(part).cast::<__m128i>());
                _mm_stream_si128(d.as_mut_ptr().add(part).cast::<__m128i>(), value);
            }
        }
    }
    // SAFETY: SSE is part of the x86_64 baseline. Orders the streaming stores before whatever
    // reads the destination next, on any thread.
    unsafe { _mm_sfence() };
    dst_tail.write_copy_of_slice(src_tail);
}

#[cfg(not(target_arch = "x86_64"))]
pub fn copy_streaming(src: &[u8], dst: &mut [MaybeUninit<u8>]) {
    dst.write_copy_of_slice(src);
}

/// Backs a large readback destination with transparent huge pages: faulting in, and later
/// freeing, its 4 KiB pages one by one costs about as much as the copy itself.
#[cfg(target_os = "linux")]
pub fn advise_huge_pages(dst: &mut [MaybeUninit<u8>]) {
    const PAGE: usize = 4096;
    if dst.len() < PARALLEL_COPY_MIN {
        return;
    }
    let start = (dst.as_mut_ptr() as usize).next_multiple_of(PAGE);
    let end = (dst.as_mut_ptr() as usize + dst.len()) & !(PAGE - 1);
    // SAFETY: the page-aligned range lies within `dst`; `madvise` only changes how the kernel
    // backs it, never its contents, and failure is harmless.
    unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE) };
}

#[cfg(not(target_os = "linux"))]
pub fn advise_huge_pages(_: &mut [MaybeUninit<u8>]) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(len: usize) {
        let src: Vec<u8> = (0..len).map(|i| (i.wrapping_mul(31) ^ (i >> 8)) as u8).collect();
        let mut dst = vec![MaybeUninit::<u8>::uninit(); len];
        advise_huge_pages(&mut dst);
        copy_parallel(&src, &mut dst);
        // SAFETY: `copy_parallel` initialised every byte of `dst`.
        let out: Vec<u8> = dst.iter().map(|b| unsafe { b.assume_init() }).collect();
        assert!(out == src, "copy of {len} bytes differs");
    }

    #[test]
    fn small_copy() {
        check(0);
        check(1000);
        check(PARALLEL_COPY_MIN - 1);
    }

    #[test]
    fn large_copy() {
        check(PARALLEL_COPY_MIN + 4099);
        check(3 * PARALLEL_COPY_MIN);
    }
}
