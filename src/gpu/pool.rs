use std::collections::{HashMap, VecDeque};

use wgpu::{Buffer, BufferUsages};

/// Smallest buffer the pool hands out. Also the size class of every dims/params buffer.
const MIN_BUFFER_SIZE: u64 = 16;

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

/// Free lists of idle buffers keyed by (size class, usage flags), each oldest first.
#[derive(Debug)]
pub(crate) struct BufferPool {
    free: HashMap<(u64, BufferUsages), VecDeque<(u64, Buffer)>>,
    pooled_bytes: u64,
    /// Stamps buffers in the order they were returned.
    clock: u64,
    /// Upper bound on the bytes kept alive in idle buffers. Past it the least recently returned
    /// buffers are released, so the pool keeps what a loop is reusing right now. Scales with the
    /// context's memory budget, so idle buffers cannot crowd out array data.
    cap: u64,
}

impl BufferPool {
    /// An empty pool that keeps at most `cap` bytes of idle buffers.
    pub(crate) fn new(cap: u64) -> Self {
        BufferPool { free: HashMap::new(), pooled_bytes: 0, clock: 0, cap }
    }

    /// Takes the most recently returned idle buffer of exactly `class` bytes with exactly
    /// `usage`, if one exists.
    pub(crate) fn take(&mut self, class: u64, usage: BufferUsages) -> Option<Buffer> {
        let (_, buffer) = self.free.get_mut(&(class, usage))?.pop_back()?;
        self.pooled_bytes -= class;
        Some(buffer)
    }

    /// Returns a buffer to the pool and releases the least recently returned ones while the
    /// pool holds more than its cap; those are returned. A buffer larger than the
    /// limit is still kept as long as nothing else is pooled. The buffer must not be mapped.
    /// Its contents are left as they are: whoever takes it next must overwrite what it reads.
    pub(crate) fn put(&mut self, buffer: Buffer) -> Vec<Buffer> {
        let class = buffer.size();
        self.clock += 1;
        self.pooled_bytes += class;
        self.free.entry((class, buffer.usage())).or_default().push_back((self.clock, buffer));
        let mut released = Vec::new();
        while self.pooled_bytes > self.cap {
            let oldest = self
                .free
                .iter()
                .filter_map(|(key, list)| list.front().map(|(stamp, _)| (*stamp, *key)))
                .min_by_key(|(stamp, _)| *stamp);
            let Some((stamp, key)) = oldest else { break };
            if stamp == self.clock {
                // Only the buffer just returned is left.
                break;
            }
            if let Some((_, old)) = self.free.get_mut(&key).and_then(VecDeque::pop_front) {
                self.pooled_bytes -= key.0;
                released.push(old);
            }
        }
        released
    }

    /// Empties the pool, returning every idle buffer.
    pub(crate) fn drain(&mut self) -> Vec<Buffer> {
        self.pooled_bytes = 0;
        self.free.drain().flat_map(|(_, list)| list.into_iter().map(|(_, b)| b)).collect()
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
