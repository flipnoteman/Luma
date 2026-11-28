#!/usr/bin/env bash
# W3-R7 part 3 targeted GPU tests: the wgpu cooperative-matrix harness, then the groups the diff touches on cuda and wgpu.
set -u
rc=0
cargo test --release --locked --no-fail-fast -p luma-hal-wgpu --test unit_verify 2>&1 | grep -E "^test |test result|panicked|^error|SKIPPED" | head -20 || rc=1
for backend in cuda wgpu; do
  echo "=== backend=$backend"
  LUMA_BACKEND=$backend cargo test --release --locked --no-fail-fast --lib -- jit::comprehension numerics:: caps:: matmul 2>&1 | grep -E "^test .*FAILED|test result|panicked|^error" | head -40 || rc=1
  LUMA_BACKEND=$backend cargo test --release --locked --no-fail-fast --test all -- matmul_half:: matmul_tf32:: matmul:: bmm:: props:: half:: structured 2>&1 | grep -E "^test .*FAILED|test result|panicked|^error" | head -40 || rc=1
done
exit $rc
