# File-backed arrays

An `Array` can live in a **file**. [`Array::open`] and [`Array::create`] give an array whose data is a
byte range of a file; every operation that works on arrays works on it, and the data never has to
fit in device or host memory. Nothing is read when the file is opened. A kernel that needs a part
of the array (a *segment*: one buffer's worth) makes the pager read that part of the file; when the
device needs room, a part that was not written is simply dropped (the file is the copy: no transfer)
and a part that was written goes back to the file first.

```rust
use luma::{Array, FileOptions};

let dir = std::env::temp_dir().join(format!("luma-guide-files-{}", std::process::id()));
std::fs::create_dir_all(&dir)?;
let path = dir.join("field.f32");

// A file of 1000 zeros, writable; the first `flush` completes it.
let mut field = Array::<f32>::create(&path, [100, 10], FileOptions::raw())?;
let x = Array::<f32>::from_slice(&(0..1000).map(|i| i as f32).collect::<Vec<_>>(), [100, 10])?;
(&x * 2.0 + 1.0).store(&mut field)?; // recorded on the device, written to the file as segments leave
field.close()?;                      // flush, fdatasync, report errors

// Open it again, read-only: the whole array is a lazy source like any other.
let field = Array::<f32>::open(&path, FileOptions::raw())?;
assert_eq!(field.shape().dims(), [100, 10, 1, 1]); // the sidecar `field.f32.luma` gave the shape
assert_eq!(field.sum().item()?, (0..1000).map(|i| (2 * i + 1) as f32).sum::<f32>());
drop(field);
std::fs::remove_dir_all(&dir)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Formats

| `FileOptions` | File | Shape and element type |
| --- | --- | --- |
| `raw()` | little-endian elements, the first dimension fastest | `shape(..)`, or the sidecar `<file>.luma` that `create` writes (`dtype`, `shape`, `offset`, `state`) |
| `raw().offset(n)` | the same, starting at byte `n` | to open a region of a larger file |
| `npy()` | a NumPy `.npy` file (versions 1 to 3, little endian) | from the header |

An `.npy` file is **not reordered**: a C-order file of shape `(a, b, c)` is an array of shape
`[c, b, a]` (the first dimension varies fastest in luma), a Fortran-order file keeps its dimensions,
and `create` writes Fortran order, so NumPy sees the same shape and the same indices. The element
type `T` has to be the file's: `f32`, `u32`, `i32`, `f64`, `u8`, `i8`, `u16`, `i16`, `f16` and
`C32` (`bf16` raw only); `bool` (bit-packed on the device) and `Df32` files are not supported, nor are
foreign or big-endian dtypes (`open` names the dtype it found).

## Writing

* `FileOptions::writable()` opens for writing (`create` always does); without it every write
  (`write`, `assign`, `+=`, a `&mut` kernel argument, `store`) fails with `LumaError::ReadOnly`.
* A file array is **never taken over in place**: `&file_array * 2.0` allocates its own result
  even when the file array was dropped, so a temporary can never write into a file.
* `array.store(&mut target)` evaluates `array` straight into `target`, one kernel per segment
  of `target`, so neither has to fit in memory. `array.write_to(path, opts)` stores into a temporary
  file next to `path`, syncs it and renames it onto `path`, so the file appears whole or not at all
  (an `.npy` atomically; a raw file and its sidecar are renamed one after the other, which is not atomic).
* `flush()` writes every changed segment back and `fdatasync`s; `close()` does it and returns the
  error. **Dropping an array also writes back what is left, but can only log a failure**: call
  `close` where you need to know. A failed write-back is returned by the call that waits for it
  (the operation that needed the room, `flush`, `close`) and leaves the segment changed in memory,
  so nothing is lost and the next `flush` tries again.
* A file made by `create` is marked as being written (the sidecar says `state = writing`, an `.npy`
  gets a marker file `<file>.luma-writing`) until its first `flush`; opening it before that fails
  unless `FileOptions::recover()` says to use it as it is.

## Locks and sharing

A file is locked while any array on it lives: a shared advisory lock for read-only arrays, an
exclusive one for writable arrays (`FileOptions::lock(false)` skips it; a file system without locks
proceeds unlocked with a warning). A lock held by another process is a `LumaError::Io` whose source
has kind `WouldBlock`. Inside one process an extent that overlaps an open one is refused
(`ResourceBusy`) unless both are read-only, and an identical re-open returns an array on the same
segments.

## What it costs

Wave 2 of the file arrays has no schedules: every kernel reads the segments it needs from the file
(through the page cache, with the I/O threads of `LUMA_IO_THREADS`), so a chain that is cut into
*k* kernels reads the file *k* times when the array is larger than the budget. `to_vec` and
`read_into` of a file array that is not in memory read the file straight into the destination.
`Context::paging_stats()` has the counters (`io_read_bytes`, `io_written_bytes`, `home_drops`,
`home_writebacks`, `file_resident_bytes`). On the CPU backend, which has no device budget, the host budget
(`ContextOptions::host_budget`) bounds the memory file data takes; other CPU arrays stay unbounded.

## Arrays of more than 2³² elements

A file may hold more elements than a `u32` counts (the dimensions each stay below 2³², their
product below 2⁴⁸): such an array exists only as a file array. Elementwise operations, casts,
comparisons and selects run on it, and so do the whole-array `sum`, `prod`, `min`, `max`, `any`
and `all` (per-segment partial results combined in segment order). Everything that needs a 32-bit
index or count is refused with `LumaError::DimensionTooLarge` (or `Chunked`): views, shifts,
`reshape`, generators, `mean`, `var`, `std`, `count`, axis reductions, `argmax`, sorts. Divide the
`sum` by the element count in `f64` yourself:

```rust,no_run
use luma::{Array, FileOptions};
// A 16 GiB file of `f32`: [65536, 65536] elements, more than a `u32` counts.
let x = Array::<f32>::open("big.f32", FileOptions::raw().shape([65536, 65536]))?;
let mean = f64::from(x.sum().item()?) / x.len() as f64;
# Ok::<(), luma::LumaError>(())
```
