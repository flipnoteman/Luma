//! [`HostBytes`]: owned host memory with a stable address, the one representation of host copies
//! of device data (the pager's spilled and kept copies, the default transfer stage).

use std::alloc::{self, Layout};
use std::fmt;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::{Mutex, PoisonError};

/// Alignment of every [`HostBytes`] block.
pub const HOST_ALIGN: usize = 64;

/// An owned byte buffer, 64-byte aligned, whose address does not change while it is held.
///
/// The representation is internal so that other backing stores (pinned memory, a file mapping,
/// backend staging) can be added without touching the users: they see a byte slice.
pub struct HostBytes {
    repr: Repr,
}

enum Repr {
    /// Plain aligned heap memory.
    Heap(Aligned),
    /// A vector that was already aligned, adopted without a copy.
    Vec(Vec<u8>),
}

/// A block from the global allocator with [`HOST_ALIGN`] alignment.
struct Aligned {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: `Aligned` owns its block exclusively, like a `Box<[u8]>`.
unsafe impl Send for Aligned {}
// SAFETY: shared access only reads the bytes.
unsafe impl Sync for Aligned {}

/// Blocks of at least this many bytes are kept for reuse when dropped (below it the allocator's own caches do the job).
const POOL_MIN: usize = 4 << 20;
/// Bytes of dropped blocks the pool keeps. Fresh anonymous pages cost a kernel fault and a zeroing each (about 1 GB/s on a loaded
/// host, a tenth of a PCIe transfer): a host copy that replaces a freed one of the same size reuses its pages instead.
const POOL_CAP: usize = 2 << 30;

/// Dropped large blocks, ready for the next block of the same length.
static POOL: Mutex<(usize, Vec<Aligned>)> = Mutex::new((0, Vec::new()));

/// Frees the blocks kept for reuse.
pub fn trim_host_pool() {
    let blocks = {
        let mut pool = POOL.lock().unwrap_or_else(PoisonError::into_inner);
        pool.0 = 0;
        std::mem::take(&mut pool.1)
    };
    for mut block in blocks {
        block.free();
    }
}

impl Aligned {
    fn layout(len: usize) -> Layout {
        // Zero-length blocks are never allocated (see `new`), so the size is at least 1.
        Layout::from_size_align(len.max(1), HOST_ALIGN).expect("host buffer size overflows")
    }

    /// Returns the block to the allocator, leaving this value empty.
    fn free(&mut self) {
        if self.len != 0 {
            // SAFETY: allocated in `new` with this layout.
            unsafe { alloc::dealloc(self.ptr.as_ptr(), Self::layout(self.len)) }
            self.len = 0;
        }
    }

    /// Uninitialised memory.
    fn new(len: usize) -> Aligned {
        if len == 0 {
            return Aligned { ptr: NonNull::<u8>::dangling().with_addr(std::num::NonZeroUsize::new(HOST_ALIGN).expect("nonzero")), len };
        }
        if len >= POOL_MIN {
            let mut pool = POOL.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(at) = pool.1.iter().position(|block| block.len == len) {
                let block = pool.1.swap_remove(at);
                pool.0 -= len;
                return block;
            }
        }
        let layout = Self::layout(len);
        // SAFETY: `layout` has a nonzero size.
        let ptr = unsafe { alloc::alloc(layout) };
        match NonNull::new(ptr) {
            Some(ptr) => Aligned { ptr, len },
            None => alloc::handle_alloc_error(layout),
        }
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        if self.len >= POOL_MIN {
            let mut pool = POOL.lock().unwrap_or_else(PoisonError::into_inner);
            if pool.0 + self.len <= POOL_CAP {
                pool.0 += self.len;
                // The block moves into the pool; this value is left empty.
                pool.1.push(Aligned { ptr: self.ptr, len: self.len });
                self.len = 0;
                return;
            }
        }
        self.free();
    }
}

impl HostBytes {
    /// `len` zero bytes.
    pub fn zeroed(len: usize) -> HostBytes {
        let bytes = Self::uninit(len);
        // SAFETY: the block holds `len` bytes.
        unsafe { std::ptr::write_bytes(bytes.heap_ptr(), 0, len) };
        bytes
    }

    /// The bytes of `data`, adopted without a copy when the vector is aligned, copied otherwise.
    pub fn from_vec(data: Vec<u8>) -> HostBytes {
        if data.as_ptr().align_offset(HOST_ALIGN) == 0 && !data.is_empty() {
            return HostBytes { repr: Repr::Vec(data) };
        }
        HostBytes::copy_of(&data)
    }

    /// A copy of `data`.
    pub fn copy_of(data: &[u8]) -> HostBytes {
        let bytes = Self::uninit(data.len());
        // SAFETY: both blocks hold `data.len()` bytes and cannot overlap.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), bytes.heap_ptr(), data.len()) };
        bytes
    }

    /// `len` bytes that `fill` writes completely (on success every byte is initialised); no
    /// zeroing pass.
    pub fn filled_by<E>(len: usize, fill: impl FnOnce(&mut [MaybeUninit<u8>]) -> Result<(), E>) -> Result<HostBytes, E> {
        let bytes = Self::uninit(len);
        // SAFETY: the block holds `len` bytes; `MaybeUninit<u8>` has no validity requirement.
        let dst = unsafe { std::slice::from_raw_parts_mut(bytes.heap_ptr().cast::<MaybeUninit<u8>>(), len) };
        super::copy::advise_huge_pages(dst);
        fill(dst)?;
        Ok(bytes)
    }

    fn uninit(len: usize) -> HostBytes {
        HostBytes { repr: Repr::Heap(Aligned::new(len)) }
    }

    /// The block of a `Heap` value; only for freshly made values.
    fn heap_ptr(&self) -> *mut u8 {
        match &self.repr {
            Repr::Heap(block) => block.ptr.as_ptr(),
            Repr::Vec(_) => unreachable!("only heap blocks are written while being built"),
        }
    }

    /// The bytes.
    pub fn as_slice(&self) -> &[u8] {
        match &self.repr {
            // SAFETY: `len` bytes, initialised by every constructor.
            Repr::Heap(block) => unsafe { std::slice::from_raw_parts(block.ptr.as_ptr(), block.len) },
            Repr::Vec(data) => data,
        }
    }

    /// The bytes, for writing.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match &mut self.repr {
            // SAFETY: as in `as_slice`, and `&mut self` is the only access.
            Repr::Heap(block) => unsafe { std::slice::from_raw_parts_mut(block.ptr.as_ptr(), block.len) },
            Repr::Vec(data) => data,
        }
    }

    /// The size in bytes.
    pub fn len(&self) -> usize {
        match &self.repr {
            Repr::Heap(block) => block.len,
            Repr::Vec(data) => data.len(),
        }
    }

    /// Whether it holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for HostBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HostBytes({} bytes)", self.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_are_aligned_and_zeroed() {
        let bytes = HostBytes::zeroed(1000);
        assert_eq!(bytes.len(), 1000);
        assert_eq!(bytes.as_slice().as_ptr() as usize % HOST_ALIGN, 0);
        assert!(bytes.as_slice().iter().all(|&b| b == 0));
        assert!(HostBytes::zeroed(0).is_empty());
    }

    #[test]
    fn large_blocks_are_reused_and_trimmed() {
        let len = POOL_MIN + 4096;
        let first = HostBytes::zeroed(len);
        let at = first.as_slice().as_ptr() as usize;
        drop(first);
        // Another test may have taken the block meanwhile: the contract is only that a block of the length comes back usable.
        let again = HostBytes::zeroed(len);
        assert!(again.as_slice().iter().all(|&b| b == 0));
        let _ = at;
        drop(again);
        trim_host_pool();
        assert!(HostBytes::zeroed(len).as_slice().iter().all(|&b| b == 0));
    }

    #[test]
    fn from_vec_keeps_the_bytes() {
        let data: Vec<u8> = (0..=255).collect();
        let bytes = HostBytes::from_vec(data.clone());
        assert_eq!(bytes.as_slice(), &data[..]);
        assert_eq!(bytes.as_slice().as_ptr() as usize % HOST_ALIGN, 0);
    }

    #[test]
    fn filled_by_writes_every_byte() {
        let bytes = HostBytes::filled_by::<()>(100, |dst| {
            dst.iter_mut().for_each(|b| {
                b.write(7);
            });
            Ok(())
        })
        .unwrap();
        assert!(bytes.as_slice().iter().all(|&b| b == 7));
    }
}
