# HANDOFF SparseDeep (branch gap-sparse, worktree ~/Documents/lp-gapsparse, base 0edb1ed)

Task: close luma-vs-cuSPARSE gaps: power-law SpMV A x 9.2M nnz (0.80 3090 / 0.64 laptop), power-law 1.1M (0.95/0.90),
CSR->CSC 9.2M (0.92 / 0.87 in the task text; my baseline run: 1.01 on 3090, 0.94 laptop). Own: src/sparse.rs, src/segments.rs,
tools/afcompare/src/{sparse,cusparse}.rs. No code change of substance yet; only temporary experiment knobs (see below).

## Setup done
- Worktree + target seeded; afcompare built (tools/afcompare/target/release/afcompare).
- Binaries for gpuq jobs: ~/Documents/lp-gapsparse-bin/{base,xp1} on both hosts (rsynced). `base` = unmodified baseline,
  `xp1` = with XP_ROWS / XP_LANES knobs. Logs of the baseline: ~/Documents/lp-gapsparse-bin/base-{local,remote}.log.
- Run form: `gpuq submit --host X --prio 0 -- bash -c 'cd ~/Documents/lp-gapsparse-bin; AF_PATH=/opt/arrayfire LUMA_BACKEND=cuda AFCOMPARE_KERNELS=1 ./xp1 --only sparse --out /tmp/o 2>&1 | grep ...'`
  (sleep >60 s in a tool call backgrounds it; job logs are in ~/.cache/bowshock-gpuq/done/<id>/log).
- Queued/running jobs at stop: local 0-20261122-105436-12599 (sp-xp1-local, LANES sweep; partial log has XP_NONE and LANES=1),
  remote 0-20261122-105437-17366 died (ssh closed, status 255) -> resubmit.
- No ncu/nsys yet; will exist after reboot: /usr/local/cuda-13.4/bin/{ncu,nsys,compute-sanitizer} (3090), /opt/cuda/bin (laptop).

## Measurements (AFCOMPARE_KERNELS=1, baseline, ms GPU per kernel)
RTX 3090 (936 GB/s):
- power-law 800k rows 9.17M nnz, SpMV A x: luma segmented path (FusedAxisReduce 204.8us + SegmentsCarry 5.1us) = 0.2124 ms vs cuSPARSE 0.1698 (0.80).
  A^T x (CSC copy, sparse_spmv_rows kernel, Poisson col lengths) 172us = 0.1758 vs cuSPARSE 0.1768 (1.01).
  uniform 600k rows 9.6M nnz: luma 169us rows kernel vs cuSPARSE 0.1774 (1.03). banded 2M rows 10M nnz: 127us (floor 96MB/936GB/s=103us) vs 0.1786.
  => cuSPARSE is skew-insensitive (~170us for 9.2-9.6M random-col nnz = ~18.5 ns... 5.4e10 nnz/s); luma's row kernel matches it on non-skewed data;
  the segmented path costs ~33us more than the row kernel on the same nnz count (not DRAM/L2 bound: 73MB val+col read ~ 80us floor).
- power-law 100k rows 1.1M nnz: luma 30.7us+4.1us carry (0.0374 ms) vs cuSPARSE 0.0357 (0.96).
- Row-kernel experiment (XP_ROWS=1, forcing sparse_spmv_rows + long-rows for max_len>64, 3090): LANES=1 -> 0.2432 ms (0.70) at 9.2M and 0.0617 ms at 1.1M:
  WORSE than segmented path; lanes 2..5 results not received. So plain row/lane kernel loses on the skewed matrix (tail imbalance) -> need nnz-balanced kernel.
- CSR->CSC 9.17M nnz, 3090: total 2.617 ms (cuSPARSE csr2csc 2.634). Kernels: gather(take) other 880us + gather(take) val 873us (= 67%),
  radix sweeps 193+192+152+93+52 = 684us, FusedPutAdd (bincount atomics) 145us, Fill 17, ScanLookback 15, misc 8.
  Laptop: 5.766 ms vs cuSPARSE 5.408 (0.94): gathers 1775+1765us, radix ~1515us, putadd 613us.
- Laptop (448 GB/s) power-law 9.2M SpMV: segmented 570us (+6 carry) vs cuSPARSE 370us (0.64); A^T x rows kernel 450us; uniform 372us vs 386; banded 291us.
  Laptop 1.1M: 56.3+4.1us vs cuSPARSE 58.6 (0.90 reported as 0.0651 vs 0.0586).

## Code state
src/sparse.rs has two TEMPORARY env knobs (XP_ROWS forces the rows kernel for SpMV, XP_LANES sets lanes_log); REMOVE before final.

## Plan / next steps (nothing blind: use ncu now)
1. ncu the baseline FusedAxisReduce (segments tile pass) kernel on the 9.2M power-law matrix (needs an afcompare diag mode or a tiny luma example that runs one SpMV):
   stall reasons, DRAM throughput, L2 hit rate of x gathers, occupancy, smem bank conflicts; the 20-step global binary search per thread
   (`lo/hi` over offs in tile_body) and the 7-round smem segmented scan are the suspects for the +33us over the row kernel.
2. Implement an SpMV-specific nnz-balanced (merge-path / CSR-stream) kernel in sparse.rs: tile found by block-level binary search of the diagonal,
   stage row ends in smem, per-thread search in smem, products staged coalesced (vec4 loads of col/val where aligned), block-wide carry-out fix-up
   in a small second kernel (deterministic, no atomics). Keep banded/uniform on sparse_spmv_rows (those wins must stay: 1.2-1.5x banded, 1.0x uniform).
   Alternative cheap variant: keep segments.rs but replace the per-thread global search by one block search + smem, drop carry kernel by fused in-kernel
   lookback if profile says the launch/carry costs.
3. CSR->CSC: replace argsort + two takes (`regroup` in sparse.rs) by sort_by_key twice (keys=col, payload row; keys=col, payload val; both stable)
   -> expected ~2x684 radix + pointer ~ 1.7 ms vs 2.6. Better: a second payload word in src/sort.rs radix sweep (not my file: ask Main) -> ~1.0-1.4 ms.
   pointer(): bincount uses atomics (145us): can be replaced by searchsorted on the sorted keys / scatter of boundaries after the sort.
   Verify identical arrays vs cuSPARSE (afcompare checks `same_parts`).
4. Gates (CPU clippy+nextest; per-host gpuq cuda+wgpu cargo test), CHANGELOG additive entry, README/afcompare results rows per GPU, report before/after per GPU.

## Job ids (cancelled at stop)
- 0-20261122-105436-12599 (local, sp-xp1-local): xp1 binary, loop over XP_NONE / XP_ROWS=1 XP_LANES=1..5, printing powerlaw/uniform CSR A x rows. Had finished XP_NONE and LANES=1 (see log in ~/.cache/bowshock-gpuq/done/ if it survives).
- 0-20261122-105437-17366 (remote, sp-xp1-remote): same sweep on the laptop; died at start (ssh 255). Resubmit.
