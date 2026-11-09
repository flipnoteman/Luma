use std::collections::HashMap;

use wgpu::{Buffer, BufferUsages};

/// Smallest buffer the pool hands out. Also the size class of every dims/params buffer.
const MIN_BUFFER_SIZE: u64 = 16;

/// Upper bound on the bytes the pool keeps alive in idle buffers. Buffers returned beyond this
/// are released to the driver instead of being retained.
const MAX_POOLED_BYTES: u64 = 1 << 30;

/// Size class for a buffer holding `bytes` bytes: the next power of two (at least
/// [`MIN_BUFFER_SIZE`]), clamped to `max_size`. `None` if `bytes` itself exceeds `max_size`.
///
/// Buffers of a class may be larger than the data they hold, so users must track the logical
/// size themselves and never rely on `Buffer::size()` / WGSL `arrayLength`.
pub(crate) fn size_class(bytes: u64, max_size: u64) -> Option<u64> {
    // Keep copy sizes aligned even when the limit is not.
    let max_size = max_size & !(wgpu::COPY_BUFFER_ALIGNMENT - 1);
    if bytes > max_size {
        return None;
    }
    let class = bytes.max(MIN_BUFFER_SIZE).checked_next_power_of_two().unwrap_or(max_size);
    Some(class.min(max_size))
}

/// Free lists of idle buffers keyed by (size class, usage flags).
#[derive(Debug, Default)]
pub(crate) struct BufferPool {
    free: HashMap<(u64, BufferUsages), Vec<Buffer>>,
    pooled_bytes: u64,
}

impl BufferPool {
    /// Takes an idle buffer of exactly `class` bytes with exactly `usage`, if one exists.
    pub(crate) fn take(&mut self, class: u64, usage: BufferUsages) -> Option<Buffer> {
        let buffer = self.free.get_mut(&(class, usage))?.pop()?;
        self.pooled_bytes -= class;
        Some(buffer)
    }

    /// Returns a buffer to the pool, or drops it if the pool is full. The buffer must not be
    /// mapped. Its contents are left as they are: whoever takes it next must overwrite what
    /// it reads.
    pub(crate) fn put(&mut self, buffer: Buffer) {
        let class = buffer.size();
        if self.pooled_bytes + class > MAX_POOLED_BYTES {
            return;
        }
        self.pooled_bytes += class;
        self.free.entry((class, buffer.usage())).or_default().push(buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::size_class;

    const MAX: u64 = 1 << 32;

    #[test]
    fn rounds_up_to_power_of_two_with_minimum() {
        assert_eq!(size_class(1, MAX), Some(16));
        assert_eq!(size_class(16, MAX), Some(16));
        assert_eq!(size_class(17, MAX), Some(32));
        assert_eq!(size_class(12_000_012, MAX), Some(16_777_216));
        assert_eq!(size_class(1 << 20, MAX), Some(1 << 20));
    }

    #[test]
    fn clamps_to_device_limit() {
        let max = (3 << 30) + 2; // not a power of two, not 4-aligned
        assert_eq!(size_class(1 << 30, max), Some(1 << 30));
        assert_eq!(size_class((1 << 30) + 4, max), Some(2 << 30));
        assert_eq!(size_class((2 << 30) + 4, max), Some(3 << 30));
        assert_eq!(size_class(3 << 30, max), Some(3 << 30));
        assert_eq!(size_class((3 << 30) + 4, max), None);
    }
}
