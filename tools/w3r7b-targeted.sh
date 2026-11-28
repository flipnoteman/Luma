#!/usr/bin/env bash
# W3-R7 part 2 targeted GPU tests: the harness, the half-precision units, the numerics bits and the groups the diff touches, cuda and wgpu.
set -u
rc=0
echo "=== luma-hal-cuda: harness and kernels"
cargo test --release --locked --no-fail-fast -p luma-hal-cuda --test unit_verify --test gemm_lanes 2>&1 | grep -E "^test |test result|panicked|^error|SKIPPED" | head -40 || rc=1
for backend in cuda wgpu; do
  echo "=== backend=$backend"
  LUMA_BACKEND=$backend cargo test --release --locked --no-fail-fast --lib -- jit::comprehension numerics:: caps:: halves 2>&1 | grep -E "^test .*FAILED|test result|panicked|^error" | head -40 || rc=1
  LUMA_BACKEND=$backend cargo test --release --locked --no-fail-fast --test all -- matmul_half:: matmul_tf32:: matmul:: bmm:: half:: 2>&1 | grep -E "^test .*FAILED|test result|panicked|^error" | head -40 || rc=1
done
exit $rc
