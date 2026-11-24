use std::collections::{HashMap, VecDeque};

use luma_core::{Buffer, BufferUsage};

/// Free lists of idle buffers keyed by (size class, usage flags), each oldest first.
#[derive(Debug)]
pub(crate) struct BufferPool {
    free: HashMap<(u64, BufferUsage), VecDeque<(u64, Buffer)>>,
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

    /// Takes the most recently returned idle buffer with exactly `usage` of `class` bytes or, if
    /// there is none, of the nearest larger class within the same octave (up to the next power of
    /// two, the size a power-of-two rule would have given the request), if one exists. A request
    /// then reuses everything the old rule would have, at most as wasteful, so finer classes
    /// cannot leave more idle buffers behind.
    pub(crate) fn take(&mut self, class: u64, usage: BufferUsage) -> Option<Buffer> {
        let octave = class.next_power_of_two();
        let mut at = class;
        loop {
            if let Some((_, buffer)) = self.free.get_mut(&(at, usage)).and_then(VecDeque::pop_back) {
                self.pooled_bytes -= at;
                return Some(buffer);
            }
            at = luma_core::hal::size_class(at + 1, u64::MAX).filter(|&next| next <= octave)?;
        }
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

    /// Bytes of the idle buffers held.
    pub(crate) fn pooled_bytes(&self) -> u64 {
        self.pooled_bytes
    }
}

#[cfg(test)]
mod tests {
    use luma_core::hal::size_class;

    const MAX: u64 = 1 << 32;

    #[test]
    fn small_sizes_round_up_to_power_of_two_with_minimum() {
        assert_eq!(size_class(1, MAX), Some(16));
        assert_eq!(size_class(16, MAX), Some(16));
        assert_eq!(size_class(17, MAX), Some(32));
        assert_eq!(size_class(2400, MAX), Some(4096));
        assert_eq!(size_class(4096, MAX), Some(4096));
    }

    #[test]
    fn large_sizes_round_to_eighth_octaves() {
        assert_eq!(size_class(4097, MAX), Some(4608));
        assert_eq!(size_class(5000, MAX), Some(5120));
        assert_eq!(size_class(5100, MAX), Some(5120));
        assert_eq!(size_class(6200, MAX), Some(6656));
        assert_eq!(size_class(8192, MAX), Some(8192));
        assert_eq!(size_class(12_000_012, MAX), Some(12_582_912));
        assert_eq!(size_class(1 << 20, MAX), Some(1 << 20));
    }

    #[test]
    fn classes_bound_the_waste_and_are_monotone() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut sizes: Vec<u64> = (0..20_000)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                // Log-uniform between 4 KiB and 1 GiB.
                let octave = (state >> 59) % 18; // 0..17
                let base = 4096u64 << octave;
                base + (state >> 8) % base
            })
            .collect();
        sizes.extend([4097, 4608, 4609, 1 << 30]);
        sizes.sort_unstable();
        let mut previous = 0;
        for bytes in sizes {
            let class = size_class(bytes, u64::MAX).unwrap();
            assert!(class >= bytes, "{bytes} -> {class}");
            assert!(class as f64 <= 1.125 * bytes as f64, "{bytes} -> {class}");
            assert_eq!(class % 512, 0, "{bytes} -> {class}");
            assert!(class >= previous, "{bytes} -> {class} after {previous}");
            previous = class;
        }
    }

    #[test]
    fn clamps_to_device_limit() {
        let max = (3 << 30) + 2; // not a power of two, not 4-aligned
        assert_eq!(size_class(1 << 30, max), Some(1 << 30));
        assert_eq!(size_class((1 << 30) + 4, max), Some((1 << 30) + (1 << 27)));
        assert_eq!(size_class((2 << 30) + 4, max), Some((2 << 30) + (1 << 28)));
        assert_eq!(size_class(3 << 30, max), Some(3 << 30));
        assert_eq!(size_class((3 << 30) + 4, max), None);
        assert_eq!(size_class((3 << 30) - 4, max), Some(3 << 30));
    }
}
