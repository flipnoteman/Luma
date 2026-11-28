#!/usr/bin/env bash
# W3-R7 part 2: decisiondiff --quick over compute and ml, integ against the branch and integ against integ, default numerics (one job).
# Usage: gpuq submit -- bash tools/w3r7w-decisions.sh BASE_REF
set -u
base=${1:-integ}
work=${TMPDIR:-/tmp}/w3r7w-dec.$$
mkdir -p "$work"
git worktree add -f "$work/base" "$base" > /dev/null 2>&1 || exit 2
for side in base new; do
  dir=$work/base; [ $side = new ] && dir=$PWD
  (cd "$dir" && cargo +stable build --release --locked -p luma-bench --target-dir "$work/t-$side" > "$work/build-$side.log" 2>&1) || { tail "$work/build-$side.log"; exit 2; }
done
unset LUMA_NUMERICS
for run in base base2 new; do
  bin=$work/t-base/release/luma-bench; [ $run = new ] && bin=$work/t-new/release/luma-bench
  LUMA_BENCH_DECISIONS=$work/$run.jsonl "$bin" --quick --suite compute,ml > "$work/$run.txt" 2>&1 || echo "bench $run exit $?"
done
echo "=== integ vs integ"; python3 tools/decisiondiff "$work/base.jsonl" "$work/base2.jsonl" | tail -15
echo "=== integ vs branch"; python3 tools/decisiondiff "$work/base.jsonl" "$work/new.jsonl" | tail -60
echo "=== integ vs branch, Unit rows allowed"; python3 tools/decisiondiff "$work/base.jsonl" "$work/new.jsonl" --allow-added-site Unit | tail -40
git worktree remove --force "$work/base" > /dev/null 2>&1
exit 0
