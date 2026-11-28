#!/usr/bin/env bash
# W3-R7 part 2: ragged f16/bf16 products, the padded-copy unit path (default numerics) against the widening path (strict numerics).
set -u
cargo +stable build --release --locked -p luma-bench > /dev/null 2>&1 || exit 2
for mode in default strict; do
  echo "=== numerics=$mode"
  if [ $mode = strict ]; then export LUMA_NUMERICS=strict; else unset LUMA_NUMERICS; fi
  LUMA_BACKEND=cuda cargo +stable run --release --locked -p luma-bench -- --only matmul/f16,matmul/bf16 2>&1 | tail -40
done
