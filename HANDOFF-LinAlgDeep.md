# HANDOFF LinAlgDeep (branch gap-linalg, worktree ~/Documents/lp-gaplinalg, base 0edb1ed)

Owned: src/dense/, tools/afcompare/src/linalg*.rs. No library code changed yet; only two new probes in
tools/afcompare/src/linalg.rs (`LINALG_ROWS=hostprobe`, `launchprobe`). Logs saved in ~/Documents/gaplinalg-logs/
(base-3090.log, base-3080.log = baseline afcompare linalg with AFCOMPARE_KERNELS=1; probe-3090.log; probe-3080.log is an
ssh failure "No route to host" -> rerun on remote). No gpuq jobs queued by me that still matter.
Run: `cd tools/afcompare && AF_PATH=/opt/arrayfire LUMA_BACKEND=cuda AFCOMPARE_KERNELS=1 LINALG_ROWS=cholesky,triangular,spd,lu,solve,inverse_det,qr cargo run --release -- --only linalg --out ../../out`
(rank rows: LINALG_ROWS=svd LINALG_SVD=rank). Submit via gpuq from the worktree (jobs build on host).

## Baseline RTX 3090 (luma ms / AF ms / AF÷luma) = the assignment's numbers
chol 256/1024/4096: 0.117/0.076 (0.65), 0.654/0.318 (0.49), 4.565/3.207 (0.70)
solve SPD 1rhs: 0.216/0.119, 1.158/0.447, 6.945/3.817 ; trsm 16rhs: 0.052/0.053, 0.260/0.230, 1.273/0.980
LU: 0.417/0.354, 2.568/2.119, 16.92/14.95 ; solve: 0.575/0.417, 3.417/2.326, 21.34/15.07
inverse 0.585/0.466, 3.502/2.844, 28.50/26.19 ; det ~ LU ; QR 0.446/0.706, 3.634/3.917, 31.23/25.92

## hostprobe (3090; enqueue = host time until eval returns, total = wall incl. sync)
chol: n256 enq 14us total 134 (5 launches); n1024 enq 86 total 755 (34); n4096 enq 373 total 4607 (154)
-> chol/trsm/solve_spd are GPU-bound (host far below total): CUDA-graph replay would NOT help them.
lu: n256 enq 245 total 481 (51 launches); n1024 enq 1498 total 2882 (338); n4096 enq 7428 total 17139 (1766)
lu solve(1 rhs, with factor): n256 190us, n1024 920us, n4096 4520us (AF's whole getrs ~0.1-0.2ms at 4096)
solve: n4096 enq 12474 total 21640 (2215); inverse n4096 enq 13423 total 28739; qr n4096 enq 13039 total 31505 (1964)
-> LU/solve/inverse/QR: host enqueue is 45-60% of wall with 300-2300 launches; (iterate-replay launchprobe: tiny
   kernels eager 3.0us/launch vs replayed 1.3us/launch; 1M-element kernels 8.5-9 us either way).
GPU profile at 4096: chol = panel 1248 + diag 1171 + gemm ~1690 + operand copies ~630 us (Fused(Shift) copies of strided
blocks fed to matmul: native gemm only reads whole buffers, so strided views are copied; fix needs matmul.rs
(not mine) to take strided/leading-dimension operands). LU 4096: lu_leaf_solve 255 launches 3.76 ms, calu 7.3 ms,
~4.5 ms of small Fused gather/select/copy kernels, matmul ~3.6 ms.

## Findings / plan (in priority order)
1. Solve phase (solve/solve_spd/trsm with few rhs) is 3-40x slower than AF's: new right-looking blocked trsv/trsm for k<=~32
   rhs in tri.rs (kernel per block: strip update rows>block with x block in smem + tile of L loaded coalesced into smem,
   thread per row, the workgroup owning the next diagonal block also solves it). Diagonal block solve latency (trsm_block
   ~25us per 128 block) is the chain: consider inverted diagonal blocks (all blocks inverted in one parallel launch) only
   inside solve/solve_spd (not generic solve_triangular; stability). Expected: solve SPD 4096 0.55->~0.73, solve 0.71->~0.86.
2. LU: cut launch count (recursion bottoms at 32-col calu panels: ~15 launches per panel; leaf_solve x255; gather_rows copies of
   whole right part; tiny perm kernels). Host-bound part (enqueue 7.4 ms of 17.1 at 4096) -> fewer/larger launches, or plan replay
   (needs jit/ infra outside my files; ask Main).
3. Rank (0.07/0.03): rank-n/2 matrices at default tol fail QR certification (see svd/rank.rs doc) -> Jacobi. Householder
   bidiagonalisation noise (~3 sqrt(n) eps ||A||_F) is about the cut n eps s_max, so a values-only bidiagonal path would not be
   reliably exact; keep exact; measure where Jacobi time goes first (decompose_ranking in svd.rs, svd/block.rs).
4. Chol: diag(128) 36us + panel 38us per block both latency bound; idea: fuse diag+panel in one launch with flag sync (panel
   workgroups preload tiles while diag factors); copies need matmul.rs strided operand support.
5. QR 4096 0.83 (tsqr_down/leaf/merge 14 ms of 31): not looked at.
Profilers available after reboot: ncu/nsys at /usr/local/cuda-13.4/bin on the 3090 -> use them inside gpuq jobs for chol_diag/chol_panel/trsm_block/calu kernels.
Gates (clippy, nextest CPU, GPU cuda+wgpu tests on both hosts) not yet run (no library change yet). Document LU tournament pivoting growth (multiplier bound 8) when touching lu.rs.
