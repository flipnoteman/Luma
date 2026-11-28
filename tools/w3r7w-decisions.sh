#!/usr/bin/env bash
# W3-R7 part 3: decisiondiff --quick over compute and ml, base against the branch and base against base (one job, default numerics),
# with prebuilt luma-bench binaries (same toolchain, separate target dirs).
# Usage: gpuq submit --host local -- bash tools/w3r7w-decisions.sh BASE_BIN NEW_BIN WORK_DIR   (LUMA_BACKEND from the environment)
set -u
base_bin=$1 new_bin=$2 work=$3
mkdir -p "$work"
unset LUMA_NUMERICS
for run in base base2 new; do
  bin=$base_bin; [ $run = new ] && bin=$new_bin
  LUMA_BENCH_DECISIONS=$work/$run.jsonl "$bin" --quick --suite compute,ml > "$work/$run.txt" 2>&1 || echo "bench $run exit $?"
done
echo "=== base vs base"; python3 tools/decisiondiff "$work/base.jsonl" "$work/base2.jsonl" | tail -15
echo "=== base vs branch"; python3 tools/decisiondiff "$work/base.jsonl" "$work/new.jsonl" | tail -80
echo "=== base vs branch, Unit rows allowed"; python3 tools/decisiondiff "$work/base.jsonl" "$work/new.jsonl" --allow-added-site Unit | tail -40
exit 0
