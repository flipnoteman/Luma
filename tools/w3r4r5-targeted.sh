#!/usr/bin/env bash
# W3-R4R5 targeted GPU tests: the rule modules and the groups the diff touches, cuda and wgpu, default and algebraic numerics.
set -u
rc=0
for backend in cuda wgpu; do for num in default algebraic; do
  echo "=== backend=$backend numerics=$num"
  if [ $num = algebraic ]; then export LUMA_NUMERICS=algebraic; else unset LUMA_NUMERICS; fi
  LUMA_BACKEND=$backend cargo test --release --locked --no-fail-fast --lib -- jit::rules jit::structured structure:: reduce:: 2>&1 | grep -E "^test .*FAILED|test result|panicked|^error" | head -40 || rc=1
  LUMA_BACKEND=$backend cargo test --release --locked --no-fail-fast --test all -- matmul:: bmm:: structure:: axis_reduce:: multi_reduce:: reduce:: dense:: 2>&1 | grep -E "^test .*FAILED|test result|panicked|^error" | head -40 || rc=1
done; done
exit $rc
