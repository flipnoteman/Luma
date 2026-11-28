#!/usr/bin/env bash
# W3-R7 part 3 (TriangularK on the shader kernels): the props, matmul and bmm groups on cuda and wgpu.
set -u
rc=0
for backend in wgpu cuda; do
  echo "=== backend=$backend"
  LUMA_BACKEND=$backend cargo test --release --locked --no-fail-fast --test all -- props:: matmul:: bmm:: 2>&1 | grep -E "^test .*FAILED|test result|panicked|^error" | head -40 || rc=1
done
exit $rc
