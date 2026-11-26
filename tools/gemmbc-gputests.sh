#!/usr/bin/env bash
# GPU suites of w2-gemm-bc: cuda and wgpu x trace-follow default/off, one job.
mkdir -p out
rc=0
cargo test --release --locked --no-fail-fast -p luma-hal-cuda --lib --tests > out/tests-hal-cuda.txt 2>&1; r=$?
echo "hal-cuda exit $r"; tail -4 out/tests-hal-cuda.txt; [ $r = 0 ] || rc=1
for be in cuda wgpu; do
  for tf in default off; do
    echo "=== $be follow=$tf"
    if [ $tf = off ]; then export LUMA_TRACE_FOLLOW=off; else unset LUMA_TRACE_FOLLOW; fi
    LUMA_BACKEND=$be cargo test --release --locked --no-fail-fast --lib --tests > out/tests-$be-$tf.txt 2>&1; r=$?
    echo "$be follow=$tf exit $r"; tail -6 out/tests-$be-$tf.txt; [ $r = 0 ] || rc=1
  done
done
exit $rc
