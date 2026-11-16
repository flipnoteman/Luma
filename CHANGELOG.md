# Changelog

All notable changes to Luma are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- A pipeline cache that grew past the 32 MB kept on disk is dropped with an `info` log message
  naming its size, instead of silently: every later process then compiles its kernels again
  (about 9.5 s for bowshock's 3D solver). The cache stays one file per device: one cache per
  pipeline, which would let old pipelines be evicted one at a time, takes about 1.77 MB per
  pipeline on NVIDIA's driver (150 kernels: 266 MB).

## [0.4.1] - 2026-11-15

### Added

- `Array::filter(&mask)` (the elements where an `Array<bool>` or non-zero `Array<u32>` mask holds, in
  flat order) and `nonzero()` (their flat indices) for every `ScalarType` incl. `bool`, as a
  count / scan / scatter compaction on the GPU: about as fast as a copy (0.15 ms for 2^24 `f32`
  with half kept on an RTX 3090, equal to a hand-written kernel). They return `Result<Option<_>>`:
  `None` when nothing is selected (luma has no empty arrays), and they synchronize once (~100 µs)
  to read the count. `filter_padded(&mask, fill)` and `nonzero_padded(fill)` do not: a full-length
  array (kept elements, then `fill`) and the count as a one-element array.
- `topk(k)` and `bottomk(k)` for `u32`, `i32` and `f32`: the `k` largest (smallest) values and
  their flat indices, by a full radix sort.
- `Array::argmax`, `argmin` (flat index, a one-element `u32` array recorded without waiting, like `sum`) and
  `argmax_axis`, `argmin_axis` (the index along an axis) for
  `f32`, `u32` and `i32`: the first of equal elements wins and an `f32` NaN is the first one found
  by both (NumPy). Reducing passes carrying a (value, index) pair, a pending operand computed
  inside the first pass; `argmax` of `2^26` `f32` and `argmax_axis(1)` of `[64, 2^22]` run at about
  860 GB/s on an RTX 3090 (`bench_datasci --only argmax`).
- `Array::searchsorted(&values, Side)`, `Array::digitize(&edges, right)` and `Array::isin(&set)`
  for `u32`, `i32` and `f32` (NumPy semantics; NaNs sort last): lazy binary searches, one per
  value, fused with the expression that makes the values (a lookup of `2^24` values in `2^16`
  edges takes 0.89 ms, 10% less than a hand-written WGSL search). `luma::Side`.
- NaN-aware reductions: `nansum`, `nanmin`, `nanmax` (`f32`, `f64`, `Df32`) and `nanmean`
  (`f32`), lazy one-element arrays, and `nansum_axis`, `nanmean_axis`, `nanmin_axis`,
  `nanmax_axis` for `f32`, with the masking fused into the reduction's first pass (NumPy's
  results: all-NaN gives 0 for the sum and NaN for the others).
- `round` (halves to even, as NumPy), `trunc`, `log1p` and `expm1` for `f32` (`round` and
  `trunc` also `f64`, `log1p` and `expm1` also `Df32`, accurate to 44 bits): lazy and fused;
  `log1p` and `expm1` keep their accuracy for `x` near 0. The fuzzer covers all four.
- Guide: searching sorted arrays, NaN-aware reductions, the new elementwise functions and a
  section of recipes (`diff`, `cumprod`, `softmax` / `logsumexp`, `one_hot`, shuffles, `cov` /
  `corr`, `clip`) in the data chapter.

### Changed

- Devices that bind few storage buffers to a kernel (llvmpipe and many integrated GPUs: 16, Metal:
  29) no longer cut an expression that reads many arrays into a kernel per operation. The lazy
  Keccak of `examples/keccak.rs` (a state of 50 arrays that every round reads) took 7120 kernels
  for three permutations with 16 bindings; it takes 39, and its host time drops from 1.07 s to
  0.18 s (lavapipe, warm; 24 s to 2.5 s the first time, which compiles every kernel). On
  llvmpipe's own limit of 45 bindings 5933 kernels become 24 (3.2 s to 0.12 s). The cut lost
  nothing but the arrays' separate buffers: arrays that a kernel would read beyond its bindings
  (or that a multi-output kernel writes beyond them) now live in one buffer that they are windows
  of, and kernels read the windows through one binding with an offset in the uniform. A write
  (`write`, `+=`, `assign`, a custom kernel taking `&mut`) gives its array a buffer of its own, so
  arrays behave as before; only the arrays used that way pay a copy. Devices with many bindings
  (an RTX 3090 binds 256) never pack. `LUMA_FUSION_MAX_INPUTS=n` lowers the limit to test this on
  any GPU.
- Choosing the arrays a cut evaluates together (`evaluate_alongside`) counts the nodes a candidate
  adds by merging two sorted lists instead of one binary search per node: building the
  three-permutation Keccak on an RTX 3090 takes 54 ms instead of 139 ms.
- The block and scalar tridiagonal solves on parts (`solve_block_tridiagonal_parts`,
  `solve_tridiagonal_parts` and their `_from` forms) choose between cyclic reduction and Thomas
  with a rule of their own: the rule of the stacked solves chose Thomas for thousands of lines
  along the contiguous axis, where its threads read each part a sector per element. Lines along
  the contiguous axis now always take cyclic reduction (RTX 3090: 5x5 blocks, 1680 lines of 187
  cells, 3.2 ms to 0.33 ms; 4x4 blocks, 8192 lines of 187, 15 ms to 0.9 ms), and strided lines
  take it up to `8192 / (2 max(B, 2))` lines (half that when a line spills out of workgroup
  memory) instead of `8192 / 4B`. Over a sweep of blocks 1-5, 24-512 cells and 256-16384 lines
  along both axes the automatic choice now takes 66 ms in total, where always the faster of the
  two would take 68 ms (clock noise) and the old rule took 332 ms.

### Fixed

- A device whose calibration always looks contended (CI's lavapipe on a shared 4-vCPU runner) was
  measured again, twice, by every process and every device it opened, about a second each, and
  the result thrown away: a test opening many contexts spent most of its time calibrating (100 s
  for one test on CI, 7 s without calibration). A contended calibration is now noted in the cache
  directory, and processes use the profile's numbers for a day before measuring again.
- A kernel that declares a binding its entry point never reads can no longer fail the launch with
  `create_bind_group: Number of bindings ... does not match ...` and, because threads share one
  batch command encoder, turn the work of every other thread in that batch into garbage (the
  third time: a fused `slice` and `trunc` allocated uniform slots for a view the lowering never
  applies, found by `fuzz::threads`). wgpu derives the pipeline layout from what the shader
  reads; a launch of one of luma's own kernels now binds exactly the arguments the built
  pipeline uses (reflected once, when it is built), skipping the declared-but-unread ones, and
  checks the argument count and kinds before it records anything, so a mismatch is an error of
  that array only. The lowering no longer allocates uniform slots for the views its instructions
  name but do not apply.
- A kernel that converts a `bool` array to `Df32` and runs no other arithmetic of a `Df32` failed
  with `create_bind_group: Number of bindings in bind group descriptor (3) does not match ... (2)`
  (found by the fuzzer, mostly on `LUMA_DEVICE=cpu`): the program counted the cast as a reader of
  the double-float guard uniform and bound it, but the cast reads nothing from it, so the shader
  compiler dropped the binding.
- `iterate_all` (and `iterate`, `iterate_until`) whose step is a lazy expression that reads a
  reduction of the state (`s.sum() + &c`, `s / s.sum()`) could compute that reduction once, on
  the placeholder that stands for the state while the step is traced: not the recording, and a
  pooled buffer holding unspecified data (what other threads had left in it, so the result
  depended on what ran at the same time: found by the concurrent fuzzer, `fuzz::threads`).
  Flattening the traced step for a register loop no longer evaluates a leaf that reads the
  placeholders; such a step is recorded and replayed with the reduction in it.
- `iterate_all` and `iterate` over an array of several buffers returned the placeholder of the
  state when the step returned its argument unchanged (`-(-x)`), which shares the state's
  buffers: `state *= result` failed with `Kernel arguments 0 and 1 share one buffer`. The
  result is now an array of its own.
- `iterate_until_all` indexed its host arrays with the step count and stop condition it read
  back from the GPU, so a corrupt value (a driver or device fault) panicked with
  `range end index ... out of range`; it now returns `LumaError::Gpu`.

### Known limitations

- Packing arrays into shared buffers on devices with few storage bindings has run on lavapipe
  (16 and 45 bindings) and on an RTX 3090 with `LUMA_FUSION_MAX_INPUTS`, not on Metal or an
  integrated GPU.
- The kernels new in this release (`filter` / `nonzero`, `argmax` / `argmin`, `searchsorted`)
  have not been run on RADV, whose results for workgroup-memory kernels are known to be wrong
  (see 0.3.0).

## [0.4.0] - 2026-11-14

### Breaking

- Comparisons (`lt`, `le`, `gt`, `ge`, `eq`, `ne`) and `is_finite` / `is_nan` / `is_inf` return
  `Array<bool>` instead of `Array<u32>`; `select`, `sum_where` and `Fields::select` take an
  `Array<bool>` or, as before, an `Array<u32>` (non-zero is true: the `Condition` trait). What
  still compiles unchanged: `select(&x.gt(0.0), a, b)`, `sum_where`, `bool * bool` (logical and),
  `mask * &x` / `&x * mask` / `mask * 2.0` and `+` / `-` with a number (the booleans become the 0
  and 1 of the number's type), and `.count()`. What breaks: code that names `Array<u32>` as the
  type of a comparison (`let m: Array<u32> = x.gt(0.0)`), passes one to a `u32` parameter, adds or
  multiplies two comparisons expecting a `u32` result (`a.gt(0.0) + b.gt(0.0)` has no `Add`;
  write `a.gt(0.0).to_u32() + b.gt(0.0).to_u32()`, or `|` for a logical or), divides, shifts or
  `minimum`s them, uses them as an index (`take`, `put`), or reads them back as `Vec<u32>`
  (`to_vec` gives `Vec<bool>`). `sum`, `cumsum` and `sum_axis` of a `bool` array count and return
  `u32` arrays; `reduce_axis`, `scan`, `mean` and the other numeric reductions are not defined for
  `bool`. An integer literal next to a `bool` array needs a suffix (`mask * 2u32`): `u32` and
  `i32` are both promoted to. See the migration chapter of the guide.
- A boolean array takes one bit per element on the GPU (a comparison written to memory is 32
  times smaller than the `u32` array of 0.3), so a custom kernel that reads or writes one sees
  bit-packed words (see the custom kernels chapter of the guide). Writes that cover whole arrays
  (comparisons, `& | ^ !`, `select`, `full`) are packed by the workgroup, and so is `assign` into
  a slice whose rows are runs of 64 or more consecutive elements (whole words inside a run, atomic
  operations only for the words at the ends of a run, which it shares with elements the write
  leaves alone); `put`, `stack`, transposed regions and short rows set and clear single bits with
  atomic operations, which is slower than the same write of another type. `count()`, `any()`
  and `all()` of an evaluated boolean array reduce its words (`countOneBits`), 32 times fewer
  elements than reading the bits one by one.
- The element-type trait `GpuType` no longer requires `bytemuck::Pod` (`bool` is not `Pod`): it
  has an associated `Repr` (the `Pod` form uploaded to the GPU) and conversions, which the
  library routes every host/GPU copy through. `luma::Numeric` (sealed: `u32`, `i32`, `f32`, `f64`,
  `Df32`) bounds the sums, scans and generic reductions that `bool` does not have.

### Added

- A user guide (`guide/`, an mdBook) published at <https://flipnoteman.github.io/Luma/> with a
  version per release (`/vX.Y.Z/`) and `/main/`: arrays, lazy evaluation and fusion, views and
  writes, reductions and scans, records, loops, precision, integer and bit operations, custom
  kernels, a performance guide and worked examples (heat equation, Euler solver, ML-KEM, Keccak,
  CPA). Its code blocks are tested as doctests.
- `Array<i32>`: `+ - * / %` (wrapping; `x / 0` is `x`, `i32::MIN / -1` is `i32::MIN`, `x % 0` is
  0), unary minus, `abs`, `sign`, `& | ^ !`, arithmetic `<< >>`, `minimum` / `maximum`,
  comparisons, `select`, views, reductions, scans and loops; `i32` scalars and `luma::Scalar<i32>`.
- `Array<Df32>::exp`, `ln`, `tanh` and `powf(Df32)`: lazy, fused like the other `Df32` operations,
  relative error below `1e-13` (at most `4.3e-14` measured) for `exp` on `[-55, 80]`, `ln` from
  `1e-24` to `1e30` and near 1, `tanh` on `[-30, 30]` and `powf` of `x` in `[1, 3000]` or
  `[1e-6, 1e6]` with exponents `-0.5..3`. `f64` arrays still have none of them.
- Casts between `f32`, `u32` and `i32`: `to_f32` (from `u32`, `i32`), `to_u32` (from `f32`,
  `i32`), `to_i32` (from `f32`, `u32`). Float to integer truncates toward zero and saturates
  (`u32` at `4294967040`, `i32` at `-2147483648..=2147483520`); NaN becomes 0. `u32` and `i32`
  convert by reinterpreting the bits.
- `Array<bool>`, the element type of comparisons and classifications (stored on the GPU as one bit
  per element: `n` booleans are `ceil(n / 32)` `u32` words, element `i` is bit `i % 32` of word
  `i / 32` and the unused bits of the last word are 0; a custom kernel sees `array<u32>` of the
  words and reads a bit as `(a[i >> 5u] >> (i & 31u)) & 1u`): `from_slice` / `to_vec` / `item` with
  plain `bool`s, `full(true)`, `& | ^ !` between arrays and with `bool`s, `count()` (a `u32`
  array), `sum()` (the same count, as NumPy), `any()` and `all()` (one-element `bool` arrays),
  `to_u32()`, `to_i32()`, `to_f32()`, `to_f64()`, `to_df32()`, `min` / `max`, `sum_axis`,
  `cumsum`, `min_axis` / `max_axis`, views, `take`, `put`, `select` and loops.
- NumPy-style promotions: `bool_array * &number_array` (either order, with a number too:
  `mask * 2.0`), `+` and `-` for `f32`, `u32`, `i32`, `f64` and `Df32` arrays; a product is
  `select(mask, x, 0)` (a masked-out `NaN` or infinity gives 0; `x *= &mask` likewise), a sum or difference converts the
  mask to the number's type first. `bool * bool` is the logical and. There is no `f64` scalar
  promotion, so that `mask * 2.0` is an `f32` array.
- `select`, `sum_where` and `Fields::select` also accept `Array<u32>` conditions (`Condition`).
- The fuzzer covers `i32`, `bool` and the new casts.
- `matmul` (`a.matmul(&b)`, `luma::matmul`) for `f32`: tiled kernels (128x128 and 32x32 output
  tiles, K split over workgroups when the output has few tiles); pending operands, transposes
  included, are computed inside the kernel's loads instead of being stored first.
- `Array::reduce_axis` (up to four of sum, min, max and sum of squares along an axis in one pass
  over the data) and, for `f32`, `mean_axis`, `var_axis` (population variance, two-pass) and
  `std_axis`.
- `put_add`, `put_min` and `put_max` for `u32`, `i32` and `f32` arrays (`AtomicElem`): scatters
  with atomic accumulation, accumulated per workgroup first for targets of up to 4096 elements;
  `f32` sums are not bitwise reproducible. `bincount` and `Array<f32>::histogram`, whose bucket
  expression fuses into the scatter.
- `sort`, `sort_descending`, `argsort`, `argsort_descending` and `sort_axis` for `u32`, `i32` and
  `f32` arrays (`SortKey`): a stable radix sort on the GPU, about 2 ms for 2^24 keys on an RTX 3090.
  Floats sort in total order, NaNs last.
- Group reductions `group_sum`, `group_min`, `group_max` (`u32`, `i32`, `f32`), `group_count`
  and `group_mean` (`f32`): scatter-based, the values and codes fuse into one kernel; codes past
  `groups` are skipped.
- `factorize` (sorted distinct values and dense codes) and `unique` for `u32`, `i32` and `f32`;
  all NaNs form one value, `-0.0` and `0.0` one value.
- `quantile`, `median`, `quantile_axis` and `median_axis` for `f32` (NumPy's linear method, on
  the sorted array).
- `examples/bench_datasci.rs`: every primitive above against a hand-written WGSL kernel of the
  same algorithm (`--gate` fails on a GPU-time ratio above the case's threshold).
- `solve_block_tridiagonal_parts` and `_from`: the block-tridiagonal solve on parts, the entries
  of the blocks and the components of the right-hand side as arrays of their own (the result is
  `[Array<f32>; B]`). Lazy parts are computed inside the solver kernel, so no stack, no unstack and
  no evaluation of the parts: a step from the expressions to the solution is one launch. Falls back
  to the stacked solve when the expressions need more buffers than a kernel binds, for blocks of 6
  to 8, when the Thomas solve is chosen and its scratch exceeds a buffer, and when a part
  cannot be fused.
- `solve_tridiagonal_parts` and `_from`: up to 64 independent scalar systems on one grid in one
  launch, with lazy parts fused likewise. `solve_tridiagonal` already batched systems along an
  extra axis (`[2, ni, nj]` along axis 1 or 2); this is now documented.
- `examples/bench_solve.rs`: the block line solves against a hand-written cyclic reduction kernel
  (parallel cyclic reduction with a closed-form 4x4 inverse), and the solves on parts against the
  stacked forms.
- `SplitReason::Shared`: profiles list the cuts where a pending expression that other pending
  operations read too is stored instead of recomputed.

### Changed

- Fields that read each other at the element itself (`-a` next to `a`, a function of a field
  next to the field) are evaluated in one multi-output kernel by `Fields::cache`, `eval_all`
  and the other multi-array evaluations, also when their expressions read shifted ones. Only a
  field another reads through a view (slice, shift, gather) is still computed first. A record of
  stage data (several cached fields that depend on each other) took one kernel per dependency
  level before.
- `sum_axis`, `min_axis` and `max_axis` no longer evaluate a pending operand first: its expression
  is computed inside the first pass of the reduction (scans still evaluate it).
- The error of `slice_mut`, `flip_mut` or `transpose_mut` on a read-only view (an array made by
  `slice`, `flip` or `transpose`) says that the array is a view of another one and that the
  mutable view is taken on the array that owns the data, with the combined range.
- The cyclic-reduction line solve (one workgroup per line, chosen for few, long lines) is up to
  1.7x faster on blocks of 2 to 4: rows are held as `vec4`s in workgroup memory, the blocks of
  2 to 4 are inverted from their cofactors instead of by elimination (a zero diagonal needs no
  pivot; the block is scaled by its largest entry first, so any uniform magnitude solves), the `vec4`
  coefficient buffers are read whole, and a workgroup has one invocation per row (rounded up to a
  warp, capped at 256, or at 128 for blocks of 3 and 4 along a strided axis) instead of one per
  eight (blocks of 2: one per two rows). 4x4 blocks, 70 lines of 144 rows on an
  RTX 3090: 21 us to 13 us (a hand-written parallel cyclic reduction kernel takes 15 us). The
  scratch of a line that does not fit workgroup memory holds `vec4` rows.

### Removed

- The ArrayFire comparison benchmarks (`benchmarks/arrayfire`, the `luma-bench` crate).

### Fixed

- Debug builds could segfault in the Vulkan loader (`vkSetDebugUtilsObjectNameEXT`) when one
  thread opened a `Context` while another built kernels: the loader links a new device into a
  list it walks to name objects without holding its lock. Devices are opened without
  `VK_EXT_debug_utils` object names now; validation layers still report through their own output.

### Known limitations

- The kernels new in this release that use workgroup memory (`matmul`, sorts, the per-workgroup
  scatters, packed boolean stores without 32-wide subgroups) have not been run on RADV, whose
  results for workgroup-memory kernels are already known to be wrong (see 0.3.0).
- `bench_datasci --gate` compares wall-clock times when the device has no timestamp queries; that
  path has not been exercised (lavapipe has them).

## [0.3.0] - 2026-11-13

### Breaking

- Reductions (`sum`, `min`, `max`, `mean`, `norm`, `reduce`) are lazy and return a one-element
  `Array`; read it with `.item()?` (or `.to_array()?` for `reduce`). `[f32; 2]` / `[f32; 4]`
  reductions still return host values.
- Arrays broadcast per axis over extents of 1 (`[N, 1] + [1, M]` -> `[N, M]`, `select` branches,
  compound assignment), plus explicit `broadcast_to(shape)`; `&x / &x.sum()` stays on the GPU.

### Added

- Views that fuse into the reading kernel: `slice`, `shift_fill` / `shift_clamp` /
  `shift_mirror`, `pad` (`Edge`, `Mirror`, `Wrap`, `Zero`), `flip`, `transpose`, `index`, `take`,
  `concat` / `concat_slice`, `broadcast_to`, and `Array::copy` (`Fields::copy`).
- Mutable views (`ViewMut` via `slice_mut`, `flip_mut`, `transpose_mut`) with in-place `assign`
  (lazy when the base is pending) and compound assignment (`+=`, `-=`, ...), `put` (scatter),
  `luma::view` for strided custom-kernel reads, `LumaError::ReadOnly`, and `pub use encase`. Read
  views reject mutating operations instead of detaching.
- Records: `Fields` trait and the `luma::fields!` macro for structs of arrays,
  with arithmetic, `dot`, `copy`, `broadcast_to`, views and `ViewMut` (`assign`, `+=`) in one
  kernel for all fields.
- Bitwise operations on `u32` arrays, fused like the rest: `& | ^ !`, `<< >>` (and `+=`-style
  assignments), `rotate_left`, `rotate_right`; amounts are taken modulo 32. `u32` states work in
  `Fields` records and `iterate_all` (register loop). Example `keccak`: Keccak-f[1600], SHA3-256
  and SHAKE128 over a batch of inputs in lazy array operations only.
- Math: `tanh`, `sign`, `floor`, `ceil`, `atan2`, `hypot`, `copysign`; `%` on `u32`.
- Precision: `Array<f64>` (devices with 64-bit shader floats) and `Df32` double-float (about 48
  bits, every device), casts between them, `sum_compensated` / `norm_compensated`.
- Loops: `iterate_until`, and `iterate_until_all` with a GPU-side stop rule
  (`Until::record` / `stop_when` / `check_every`), per-step history, step index and stop reason.
- Line primitives: `solve_tridiagonal` / `solve_block_tridiagonal` (blocks 1-8, `_from`
  variants), `scan`, `cumsum`, `sum_axis` / `min_axis` / `max_axis`. Few long lines use a
  cyclic-reduction solver on NVIDIA (`LUMA_TRIDIAGONAL=thomas|cyclic`), 3-22x faster.
- `item`, `to_array`, `sum_where`, `luma::Scalar` (uniform that changes without recompiling),
  `Array: Clone`, `cache()`, `fusion_limit`.
- `stack` / `stack_slice`: equal-shaped arrays joined along a new axis in one kernel that computes
  one element of every part and writes the stacked layout (16 lazy expressions of 144x70 into
  `[16, 144, 70]`: one launch, the host cost of 16 views, 15 selects and a coordinate array gone).
- `is_finite`, `is_nan`, `is_inf` on `f32`, `f64` and `Df32` arrays (lazy `u32` masks, tested on
  the bits of the exponent through the guard the double-float library uses, so a device's
  fast math cannot fold the test of a computed value).
- Cost model: `CostModel` / `Context::cost_model`, calibrated per device
  (`ContextOptions::calibrate`, `LUMA_CALIBRATE`).
- Simplification of operations as they are built; value-changing rewrites and a cheaper `Df32`
  add only with `ContextOptions::fast_math` / `LUMA_FAST_MATH=1`.
- On-disk cache of compiled pipelines and calibration in `$XDG_CACHE_HOME/luma`
  (`LUMA_CACHE_DIR`, `LUMA_CACHE=off`); dropped above 32 MB.
- Diagnostics: `Profile::splits` with `SplitReason`, `check_finite` / `LUMA_CHECK_FINITE`
  (`LumaError::NonFinite`), `LumaError::Unsupported`.
- Settings: `LUMA_FUSION_MAX_NODES`, `LUMA_FUSION_MAX_LIVE`, `LUMA_STAGING`, `LUMA_SPECULATE`,
  `LUMA_DF_FMA`, `ContextOptions::df_fma`.
- Benchmarks `bench_pde`, `bench_build`, `bench_threads`; new `luma-bench` workloads.
- Example `mlkem`: ML-KEM (FIPS 203) KeyGen, Encaps and Decaps for ML-KEM-512, -768 and -1024 over
  batches of independent instances in lazy array operations only (no custom kernel), checked byte
  for byte against the RustCrypto `ml-kem` crate (implicit rejection included) and NIST ACVP
  vectors (test `mlkem`). RTX 3090, batches of 65536, keys and ciphertexts on the device: 2.4 / 2.1 /
  1.9 M ops/s (KeyGen / Encaps / Decaps) for ML-KEM-1024, 3.7 / 3.0 / 2.7 for -768 and 6.1 / 4.8 /
  4.2 for -512; all 32 CPU threads with the `ml-kem` crate do 0.15-0.52.

### Changed

- `reshape` of a lazy view (slice, flip, transpose, pad, shift) stays lazy, a handle on the
  pending view as a clone is: `x.slice(0, k..k + 1).reshape([ni, nj])` reads `x` inside the kernel
  that uses it instead of evaluating the slice. Other lazy arrays are still evaluated by it.
- Fusion is decided by the cost model: an expression read at several indices is recomputed,
  staged in workgroup memory, or evaluated first (with its sibling fields), whichever is cheapest.
  The fusion limit is 16384 operations (nesting 512 levels); identical expressions are one node.
- Cuts count the operations that are still pending: after a cut, later operations no longer
  inherit the size of what was evaluated (a 10,000-operation chain made hundreds of kernels).
  An expression whose scalars outgrow one uniform buffer is cut (`SplitReason::Scalars`).
- Shifts and rotations by a constant are compiled with the amount as a literal.
- Region `assign` compiles to `if`/`else`, so ghost expressions cost only their own cells.
- A select on the coordinates is judged by the work it saves per cell that reaches it, not per
  cell of the kernel, so the chain of nested selects that `concat` of many parts makes computes
  every part only where it is taken: joining 168 arrays of 590 000 elements 13.8 to 1.0 ms.
- The pool of idle buffers keeps up to half of the memory budget (was an eighth, at most 1 GB):
  a program that cycles through gigabytes of temporaries every call reuses them instead of
  allocating again, which costs about 20 ms per GB on an RTX 3090 (ML-KEM-768 KeyGen of 65536
  instances 137 to 37 ms).
- `iterate*` trace the step once and replay its kernels; `eval` loops are recognized and batched;
  readbacks in convergence loops can be answered from steps computed ahead.
- `iterate_all` register loops called again from the same place with the same step graph over
  other arrays reuse the earlier program (no flattening), and from the third call on the GPU
  starts before `step` is traced; the steps of a loop are cut into launches of about 40 ms by
  the cost model instead of a fixed work budget (32 lazy Keccak permutations over 2^20 states:
  one launch, 25.7 to 15.7 ms).
- Reductions fuse their elementwise input and short epilogues (`norm`: 2 launches); an operation
  on a reduction no longer forces it early.
- `Df32` arithmetic 1.5-2x faster (`fma` products where the device's `fma` is fused, one-step
  division and square root).
- Faster scans, axis reductions and tridiagonal solves; less host time to build and flatten
  expressions.
- Results on RTX 3090 (`bench_pde`): `euler/until_all` 2048² 72 to 17 ms, `muscl` 26-34% faster,
  Euler `iterate_all` 2048² 40.8 to 8.4 ms.

### Fixed

- Wrong results with threads sharing a context, and up to 10x slowdown from compile locks.
- `assign` / view / write ordering bugs (writes lost, stale views, loops corrupting arrays created
  before them), found by the new differential fuzzer.
- `Df32` kernels failing to bind their uniform; casts cancelled by the shader compiler.
- Input-buffer count wrong after `cache()`.
- Slicing, flipping or transposing a reshaped read-only view read the wrong elements.
- A deadlock in `item()` readbacks that evict under a memory budget while speculating.
- `hypot` with a NaN operand and a zero operand returned 0 instead of NaN.
- Region assigns into lazy expressions (the ghost cells and solid blocks of a finite-volume
  solver) made up to 30 times as many launches: the work of a shared expression was counted once
  per path through it (doubling with every level), so small ghost expressions were evaluated in
  kernels of their own, and a cut forced by the instruction cap inside another cut evaluated the
  arrays built on top of the expression piece by piece. One SSP-RK3 step with a solid block on a
  156x76 grid, single precision, no `cache()`: 89 launches / 320 us to 14 / 60 us.
- NVIDIA driver hang or crash in `exit`: luma waits for driver-started threads and ends the
  process with `_exit` if one stays busy (`LUMA_EXIT_WAIT=0` skips the wait).
- Cyclic-reduction index wrap above $2^{23}$ cells, pager self-deadlock when spilling during a
  traced `iterate*` step, `ReadOnly` alias checks on `+=`/`write`/mutable kernel args, `NaN`
  residuals falsely converging `iterate_until*`, `luma::fields!` record `+=` mutating shared buffers in
  place, `hypot` overflow and Metal `tanh` saturation above `|x| >= 44`, tiled `conv1d` out-of-window
  `Inf` contamination, and Linux `musl` builds.

### Testing

- Differential fuzzer against a CPU oracle, invariant validation in debug builds, CI on lavapipe
  with a nightly fuzz job.

### Known limitations

- NVIDIA Linux driver 615.x can still crash or hang in its own shader-compiler thread at
  process exit; luma's exit wait and `_exit` fallback make it rare, not impossible.
- On NVIDIA, creating a device (`Context::new`) blocks submissions of other devices in the
  process until their in-flight GPU work finishes (once per device creation).
- RADV integrated GPUs give wrong results for kernels that use workgroup memory (scans,
  staging, cyclic tridiagonal solves); the cyclic solver is only chosen automatically on NVIDIA.
- `solve_tridiagonal` / `solve_block_tridiagonal` take `f32` arrays only.

## [0.2.0] - 2026-11-11

### Added

- Generator arrays: `Array::uniform`, `Array::normal`, `Array::random_bits` (Threefry-2x32-20, a
  counter-based generator: element `i` depends only on the seed and `i`), `Array::range` and
  `Array::linspace`, each with an `_on` variant. They are computed inside the kernels that read
  them, so a reduction over them reads no memory. Array-code Monte Carlo (pi, European call) over
  2^24 samples: 2.1x and 2.3x faster than ArrayFire CUDA on an RTX 3080 Laptop.
- `iterate_all`: a loop over several arrays (`[&zx, &zy, &count]`). When the step is elementwise,
  the whole loop runs as one kernel that keeps the state in registers, for any step count;
  anything else (shifts, arrays split into several buffers) runs step by step. Mandelbrot 1024²,
  256 iterations: 0.46 ms on an RTX 3080 Laptop, against 3.6 ms for the unevaluated chain.
- ArrayFire comparison benchmarks in `benchmarks/arrayfire` (separate workspace, not part of the crate).
- `select` accepts scalar branches; the result's shape and context come from the mask.
- CPU fallback: when no GPU opens, Luma runs through Mesa llvmpipe (a software Vulkan driver that
  must be installed). `LUMA_DEVICE=cpu` or `Context::cpu()` force it, `LUMA_DEVICE=gpu` refuses it.
  New `DeviceKind`, `ContextOptions`, `Context::with_options`, `Context::is_cpu` and
  `TuningProfile::Cpu`.
- Arrays larger than one device buffer are split over several buffers automatically; every
  built-in operation (elementwise, fills, reductions, shifts, `iterate`, `conv1d`, `conv1d_fft`)
  runs over them in pieces.
- Array data beyond the device memory budget moves to host memory and back on demand, with a
  one-time warning. `ContextOptions::memory_budget` / `LUMA_MEMORY_BUDGET`,
  `ContextOptions::chunk_bytes` / `LUMA_CHUNK_BYTES`, `Context::spilled_bytes`.
- `Array::read_into`: read back into host memory you already own.
- Compound assignment (`x += ..`) fuses with the surrounding expression and writes in place.
- `luma::wgpu` re-export, so the wgpu types in `LumaError` and `Context::features` can be named.
- `LumaError::NoDevice` and `LumaError::Chunked`.
- CI workflow that builds and tests on the software driver.
- Typed `Array<T>` (`u32` and `f32`) with elementwise add, subtract, multiply and divide, scalar
  operations and operators (`+`, `-`, `*`, `/`), an `array!` macro, and several arrays alive at once.
- `conv1d` with stride and padding, and `conv1d_fft` (FFT-based convolution for any kernel length).
- Explicit `Context` that owns the device, queue, pipeline cache and a buffer pool; `LumaError`
  and `luma::Result` replace string errors, with validation of shapes, strides and cross-context use.
- Custom WGSL kernels: `ctx.kernel(wgsl).build()` reflects bindings and workgroup size with naga,
  and `args!` checks read / read-write / uniform arguments.
- Lazy evaluation: elementwise expressions are fused into one kernel (up to 128 operations,
  several outputs), with in-place reuse of buffers, shifts and `Array::iterate` (time-blocked
  stencils). Errors from lazy operations are reported when the result is read.
- Vector reductions (`sum`, `min`, `max`, `norm`, several at once with `reduce`), using subgroup
  operations where the device has them; `LUMA_SUBGROUPS` overrides.
- `flush`, profiling and `Context::features` (subgroup, f16 and cooperative-matrix support).
- Per-GPU tuning: Apple, NVIDIA and generic profiles chosen from the adapter, clamped to the
  device limits; `LUMA_TUNING` overrides.
- Zero-copy uploads on unified-memory devices; large uploads and readbacks use several threads.
- Criterion benchmarks of application workloads against multi-threaded CPU baselines, and the
  `bench_conv1d` example.
- MIT and Apache-2.0 licenses, a rewritten README and a custom-kernel example.

### Changed

- **Breaking:** `Array::eval` and `eval_all` no longer wait for the GPU, like ArrayFire's `eval`:
  they record and hand the batch over, and `Context::sync`, readbacks and reductions wait. A GPU
  error raised by the batch shows up at the next waiting call. A loop that evaluates every step
  no longer pays a full round trip per step (about 35 µs on NVIDIA/Vulkan on Linux).
- `eval` and `eval_all` postpone the computation while the GPU is busy, until an `eval` finds it
  idle or something waits. Requested arrays dropped in the meantime are never computed, and the
  steps in between fuse. Mandelbrot evaluated every iteration (1024², 256 iterations): 25.6 to
  4.8 ms on an RTX 3080 Laptop, against 25.1 ms for ArrayFire CUDA.
- Work evaluated at the fusion limit is handed to the GPU right away, so a long chain built
  without `eval` runs while the host records the rest (mandelbrot 1024², 256 unevaluated
  iterations: 5.6 to 3.7 ms on an RTX 3080 Laptop).
- At the fusion limit, live arrays that share pending work with the biggest operand are evaluated
  together in multi-output kernels, instead of that operand alone.
- `eval` holds back up to 16 kernels while the GPU is still busy and hands them over with the next
  submitting or waiting call (one submission instead of one per call). Call `Context::flush`
  before long work that does not touch Luma.
- `Context::sync` and readbacks (`to_vec`, `read_into`, reductions) poll before blocking in the
  driver, for 1.5x the longest of the last four waits (at least 500 µs; waits predicted beyond 4 ms
  block as before), so a result that is about to arrive is not picked up late by a sleeping thread.
- Uniform values of a batch are copied by the batch's own command buffer (first 4 KiB) instead of a
  second one, saving about 2 µs per submission.
- Building and flattening lazy expressions is cheaper (about 40% less host time for a 50-operation
  expression).
- Builds on stable Rust (1.99); the nightly toolchain pin is gone.
- `LumaError`, `TuningProfile`, `DeviceKind` and `ContextOptions` are `#[non_exhaustive]`:
  matches need a wildcard arm, and `ContextOptions` is built with its chainable methods
  (`ContextOptions::default().chunk_bytes(4096)`).
- GPU devices are pooled per process and reused by later contexts instead of being destroyed.
- On GPUs whose buffer limit is below 4 GiB of contiguous binding, arrays between 2 GiB and 4 GiB
  now span two buffers; custom kernels on such arrays fail with `LumaError::Chunked`.
- The example `double_test` is now `basics`.
- Upgraded to wgpu 30; the adapter's limits are requested and errors are returned instead of
  panicking.
- The global async executor (and the `tokio`, `uuid`, `flume` and `once_cell` dependencies) is
  replaced by a synchronous `Context`; only reading data back waits for the GPU.
- WGSL shaders are embedded in the crate instead of being read from a directory at run time.

### Fixed

- On Linux with the NVIDIA driver, a process could hang, abort or segfault inside `exit` when the
  driver's own exit handler ran while its background worker threads were still busy (seen after
  long runs; about 1 in 4 full `luma-bench` runs on driver 615.78). Luma now registers an exit
  handler that runs first and waits, at most about 4 s, for the GPU and those workers to go idle.
  It destroys nothing, so the driver's deadlock on destroying a device cannot occur. It adds
  about 60 ms to every exit on such systems; `LUMA_EXIT_WAIT=0` skips it.
- A fused expression whose operands were computed after it was built no longer exceeds the
  device's storage-buffer binding limit; its operands are computed first instead.
- Arrays uploaded with `from_slice` on unified-memory GPUs now count against the memory budget and
  can be spilled; before, only arrays computed on the GPU did. Newly resident data is no longer
  the first eviction candidate.
- Moving a segment to host memory reads it back through a staging buffer of at most 32 MiB instead
  of one as large as the segment, which could fail when the device was full.
- FFT passes no longer read and write past the work buffer when the folded grid launches more
  workgroups than there are tiles (more than 65535 tiles).
- Batches without uniforms no longer copy the 4 KiB uniform prefix.
- Chunked readbacks record one wait for the adaptive poll instead of one per piece, so short tail
  waits no longer shrink the poll budget.
- Freeing bind groups of reclaimed buffers ends the open compute pass first, so a later launch
  cannot reuse a stale binding.
- Uniform uploads no longer allocate and free a host-visible memory block on every submission
  (about 0.3 ms per submission on NVIDIA/Vulkan when no small upload buffer was alive).
- Workgroup grids larger than one row are folded evenly. Before, 65536 workgroups (for example 2^24
  elements) were launched as 65535 x 2 with one nearly empty row, costing about 10% on
  bandwidth-bound kernels of that size.
- Arrays larger than 65,535 workgroups of elementwise work are dispatched correctly, and the
  dimensions are passed to shaders in the layout they expect.
- Integer division by zero keeps its guard in fused kernels (`u32` `x / 0` gives `x`).
- Stencil probe aliasing, shift root flattening and upload / staging buffer reuse; uploads reclaim
  remapped buffers and retry on out-of-memory.
- `conv1d` handles inputs larger than 65,535 elements.
- `conv1d_fft` returned wrong results on devices that run fewer than 16 shader invocations in
  lockstep (a data race in the FFT's local pass). The fix makes `conv1d_fft` about 12% slower on
  an RTX 3090.

### Known limitations

- Custom WGSL kernels need arrays that fit one buffer.
- Convolution weights must fit one buffer; on chunked inputs `conv1d` kernels are limited to half
  a buffer and `conv1d_fft` kernels to a sixteenth.

## 0.1.3 - 2024-08-08

### Added

- `array!` macro for creating an `Array`.
- A global `Executor` that is created once and shared by all arrays.

### Changed

- Updated the README and dependencies.

## 0.1.0 - 2024-07-23

> **Warning:** 0.1.0, the version published on crates.io, was non-functional and must not be
> used. Use 0.2.0 or later.

### Added

- First crate with a `wgpu` dependency and an `Array` struct holding data and four dimensions.
- A WGSL compute shader and an `Executor` that doubles every value of an array (`double_test`).
- A library layout, with the shader loaded from a hard-coded relative path.

[Unreleased]: https://github.com/flipnoteman/Luma/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/flipnoteman/Luma/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/flipnoteman/Luma/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/flipnoteman/Luma/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/flipnoteman/Luma/releases/tag/v0.2.0
