#!/usr/bin/env bash
# W3-R4R5b: decisiondiff --quick over compute and ml, integ against the branch, and integ against integ, default and algebraic numerics
# (one job). Usage: gpuq submit -- bash tools/w3r4r5b-decisions.sh BASE_REF
set -u
base=${1:-a98d2f8a}
work=${TMPDIR:-/tmp}/w3r4r5b-dec.$$
mkdir -p "$work"
git worktree add -f "$work/base" "$base" > /dev/null 2>&1 || exit 2
for side in base new; do
  dir=$work/base; [ $side = new ] && dir=$PWD
  (cd "$dir" && cargo +stable build --release --locked -p luma-bench --target-dir "$work/t-$side" > "$work/build-$side.log" 2>&1) || { tail "$work/build-$side.log"; exit 2; }
done
rc=0
for num in default algebraic; do
  for run in base base2 new; do
    bin=$work/t-base/release/luma-bench; [ $run = new ] && bin=$work/t-new/release/luma-bench
    if [ $num = algebraic ]; then export LUMA_NUMERICS=algebraic; else unset LUMA_NUMERICS; fi
    LUMA_BENCH_DECISIONS=$work/$num-$run.jsonl "$bin" --quick --suite compute,ml > "$work/$num-$run.txt" 2>&1 || echo "bench $num $run exit $?"
  done
  echo "=== $num: integ vs integ"; python3 tools/decisiondiff "$work/$num-base.jsonl" "$work/$num-base2.jsonl" | tail -15
  echo "=== $num: integ vs branch"; python3 tools/decisiondiff "$work/$num-base.jsonl" "$work/$num-new.jsonl" | tail -40
done
git worktree remove --force "$work/base" > /dev/null 2>&1
exit $rc
