# HANDOFF: SetMerge (branch `gap-setmerge`, worktree `~/Documents/lp-setmerge`, base 0edb1ed)

Task: sorted-input `set_union` / `set_intersect` are 0.50-0.58 of ArrayFire on CUDA. Find why the previous
fused (single-kernel merge-path + look-back) version hit CUDA_ERROR_ILLEGAL_ADDRESS, then build a correct
fused (or two-kernel, no global atomics) merge that beats ArrayFire on both GPUs. I own `src/setops.rs`,
`src/setmerge.rs` (the merge kernel file) and tests for them. No queued gpuq jobs at handoff.

## Done
* Worktree + target seeded. `tools/afcompare`: the set rows (`15 set_*`) moved out of group `bykey` into a
  new group `sets` (`bykey::sets`, registered in `main.rs`, array length 24 -> 25), so `--only sets` runs only
  the set rows in seconds. (README `--only` list still needs `sets` added.)
* No change to `src/setmerge.rs` / `src/setops.rs` yet.
* Reconstructed the lost fused kernel as `HANDOFF-SetMerge-prev-fused.patch` (git diff vs 0edb1ed of
  `src/setmerge.rs`, recovered from the previous agent's session transcript
  `~/.omp/agent/sessions/-Documents-luma/2026-11-19T07-14-38-776Z_*/AfSortScan.jsonl`, entries 774 and 778;
  a digest is in `/tmp/setmerge_trace_keep.txt` if /tmp survives). It compiled with `cargo check`; not run.
  Apply with `git apply HANDOFF-SetMerge-prev-fused.patch` in a scratch worktree to reproduce.

## Baseline (afcompare `--only sets`, AFCOMPARE_KERNELS=1, binary /tmp/sm-afc-base, job ids
0-20261122-105312-20814 local / -22664 remote; logs in ~/.cache/bowshock-gpuq/done/)
sorted distinct u32, 3 181 963 elements each (m=4M draws below 2^23, deduped):

| row | RTX 3090 luma / AF ms (AF/luma) | RTX 3080 Laptop luma / AF ms |
|---|---|---|
| set_union sorted | 0.1744 / 0.0988 (0.57) | 0.3138 / 0.1812 (0.58) |
| set_intersect sorted | 0.1496 / 0.0801 (0.54) | 0.2796 / 0.1404 (0.50) |
| set_union unsorted 4M | 0.663 / 0.767 (1.16) | 1.344 / 1.523 (1.13) |
| set_intersect unsorted 4M | 0.648 / 0.749 (1.15) | 1.340 / 1.491 (1.11) |
| set_unique unsorted 4M | 0.322 / 0.340 (1.05) | 0.640 / 0.665 (1.04) |

GPU time of the sorted rows: 3090 union SetMerge x3 = 153.8 us + scan ~12 us (wall 174 us, so launch gaps are
only ~8 us: the kernels themselves are slow); 3090 intersect 129.6 us; laptop union 291.5 us, intersect 259.2 us.
Memory floors: 3090 union ~46 MB traffic = ~49 us, intersect ~30 MB = ~32 us; laptop (448 GB/s) union ~103 us,
intersect ~67 us. ArrayFire's union is concat + radix sort + unique; its intersect is Thrust's fused
set_intersection (decoupled look-back).

## Findings about the earlier ILLEGAL_ADDRESS (not yet root-caused)
* Internal stencil kernels (`jit.stencil_kernel`) are built with `KernelTrust::Trusted`: **no bounds checks**
  and workgroup memory is NOT zeroed. (Only user kernels, `src/kernel.rs:35,67`, are `Checked`.) So "bounds-checked
  build" must be forced: build the kernel with `kernel::build_core(ctx, &KernelKey::new(src,"main",None), label,
  KernelTrust::Checked)` behind a debug env var; that will catch OOB on CPU / wgpu / CUDA.
* Previous diag runs (3090 + laptop): faults on the first call in every variant, incl. ticket replaced by
  `workgroup_id` (DIAG=3). DIAG=4 (look-back skipped) leaves `base_excl` uninitialised workgroup memory, so
  that variant was invalid as a test. Old 3-kernel path with `LUMA_LOOKBACK_SCAN=off` passes.
  Suspects not yet excluded: (a) CUDA codegen of subgroup ops in non-converged code: `crates/luma-cuda/src/codegen/conv.rs`
  only uses full-mask `*.sync` when the op is provably reached by a whole warp; else a partial-warp
  `activemask` loop path (ops.rs `sg_name`, `combine`). The previous look-back had `subgroupBallot` after a
  divergent `while (w == 0u)` spin inside `if (lane < j)` inside `if (t < 32u)`. (b) `Partition` now binding an
  extra `st` array (uniform at binding 4) and zeroing `st[g]`, `st[tiles+1]`. (c) `out` allocated as
  `most` elements then `.slice(0, ..k)`. (d) `atomic_load` emission in `ops.rs:440` (check that spins cannot be hoisted).
* Method to continue: apply the patch in a scratch worktree; make it runnable on CPU (needs the no-subgroup
  scan variant; CPU `LUMA_LOOKBACK_SCAN=on`) and under `Checked` trust; shrink inputs (2 elements, 1 tile) and
  bisect via env toggles (ticket, look-back, subgroup builtins, extra binding), now with `compute-sanitizer`
  (`/usr/local/cuda-13.4/bin/compute-sanitizer` on the 3090 after the reboot, `/opt/cuda/bin` on the laptop)
  inside gpuq jobs.

## Planned design (nothing written yet)
Two launches + final readback, CUDA only (`tuning.lookback_scan && tuning.subgroup32 && total < 2^30`); other
backends keep the existing 3-kernel path.
1. Partition kernel (as now) also zeroes `st` (tiles + 2 words: status per tile, ticket at `tiles+1`).
2. Fused kernel, tile = 256 threads x 8 items (try 16): ticket for tile id; load the tile with a fixed 8-iteration
   coalesced loop over the virtual concatenation [a0..a1) ++ [b0..b1) into smem `ka`/`kb`; per-thread merge-path
   search; one unrolled walk (generator emits 8 explicit steps, keys + keep flags in registers); block scan of
   counts with subgroup scans (warp totals in smem); tile aggregate published as `(1<<30)|count`, tile 0 as
   inclusive `(2<<30)|count`; look-back by warp 0 using a *uniform* loop (ballot of "not ready", reload until
   none; ballot of inclusive flags; `firstTrailingBit`; `subgroupAdd`); lane 0 publishes inclusive and the base via
   smem; barrier; stage kept keys into smem (aliasing `ka`, after a barrier) and store coalesced through the
   inverse key transform (`KeyTransform::inverse`; a float zero comes out as +0.0 and NaN as the canonical NaN, as
   `sort` does). Output buffer worst case `na+nb` (union) / `min(na,nb)` (intersect), result = slice (copy to
   an exact-size array if the result is under half of it); kept count read from `st[tiles-1] & 0x3fffffff`
   with the single end sync (removes the mid-way readback of the 3-kernel path).
3. Give the passes distinct profile labels ("SetPartition", "SetCount", "SetWrite", "SetFused") so
   AFCOMPARE_KERNELS shows each kernel.
4. Tests (tests/all/bykey.rs): large sorted inputs (hundreds of tiles, incl. > 2^16 tiles-worth), duplicates
   inside a/b, disjoint / identical / one empty-result intersections, NaN and +-0 (documented conventions),
   i32 / u32 / f32, tile-boundary sizes (2047, 2048, 2049, 1 element), Op::Intersect and Op::Union, host-set
   reference.
5. Profile with `ncu` (after reboot) the fused kernel on the 3090: memory throughput, smem bank conflicts,
   stalls; goal >= 60 % of peak BW so union <= ~75 us and intersect <= ~55 us on the 3090 (AF 98.8 / 80.1 us),
   and beat AF 181 / 140 us on the laptop.
6. Gates (see assignment): CPU clippy + nextest; CUDA and wgpu `cargo test --release --locked --no-fail-fast
   --lib --tests` per host with fresh LUMA_CACHE_DIR; CHANGELOG entry (additive); README `--only` list gets `sets`.

## State of the repo
Uncommitted before this handoff: afcompare `sets` group + this file + the patch (committed in the WIP commit).
