#!/usr/bin/env bash
# W3-R7 part 2: decisiondiff --quick over compute and ml with prebuilt binaries (integ twice, then the branch), default numerics.
# Usage: gpuq submit --host local -- bash tools/w3r7c-decisions.sh BASE_BIN NEW_BIN OUT_DIR
set -u
base=$1 new=$2 work=$3
mkdir -p "$work"
unset LUMA_NUMERICS
for run in base base2 new; do
  bin=$base; [ $run = new ] && bin=$new
  LUMA_BENCH_DECISIONS=$work/$run.jsonl "$bin" --quick --suite compute,ml > "$work/$run.txt" 2>&1 || echo "bench $run exit $?"
done
echo "=== integ vs integ"; python3 tools/decisiondiff "$work/base.jsonl" "$work/base2.jsonl" | tail -n 15
echo "=== integ vs branch"; python3 tools/decisiondiff "$work/base.jsonl" "$work/new.jsonl" | tail -n 60
echo "=== integ vs branch, Unit rows allowed"; python3 tools/decisiondiff "$work/base.jsonl" "$work/new.jsonl" --allow-added-site Unit | tail -n 40
exit 0
