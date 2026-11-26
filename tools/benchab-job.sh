#!/usr/bin/env bash
# The measuring half of tools/benchab.sh (it ships this script and the two luma-bench binaries to the GPU queue); not run by hand.
# Every suite is split into units (luma-bench --list-groups: a `suite/group`, or the `suite/prefix` of cases outside any group),
# and the two binaries alternate unit by unit: A B B A, then B A A B for the next chunk, so a GPU power or clock state that
# drifts over the job hits both sides alike. Each run starts with a 400 ms warm-up kernel (luma-bench --warm). Sample n of a side
# (n = 1 .. 2 x rounds) of every unit lands in out/<label>-<backend>-<suite>-<side><n>.jsonl (and .txt), the files benchcmp.py reads.
# Input (environment): A B (binaries) LA LB (labels) BACKENDS SUITES ROUNDS DEADLINE (seconds into the job after which no run starts)
# CHUNK (units per alternation step, 0 = auto: at most 48 steps per suite) QUICK ONLY (luma-bench flags) LEGACY (1: whole-suite runs)
set -uo pipefail
mkdir -p out
A=$(cd "$(dirname "$A")" && pwd)/$(basename "$A") B=$(cd "$(dirname "$B")" && pwd)/$(basename "$B")
CA=$(mktemp -d) CB=$(mktemp -d)   # a kernel/pipeline cache per side, empty at the start of the job (a shared one near its size cap
                                  # makes context creation cost ~1 ms per cached MB and vary with whatever runs left in it)
extra=(); [ -n "${QUICK:-}" ] && extra+=(--quick); [ -n "${ONLY:-}" ] && extra+=(--only "$ONLY")
total_units=0 ran=0

bin_of() { [ "$1" = a ] && echo "$A" || echo "$B"; }
lab_of() { [ "$1" = a ] && echo "$LA" || echo "$LB"; }
cache_of() { [ "$1" = a ] && echo "$CA" || echo "$CB"; }
in_time() { [ "$SECONDS" -lt "$DEADLINE" ]; }

# list_units SIDE BACKEND SUITE -> units on stdout; status 2 when the binary has no --list-groups.
list_units() {
  local bin out rc
  bin=$(bin_of "$1")
  out=$(LUMA_BACKEND=$2 LUMA_CACHE_DIR=$(cache_of "$1") "$bin" --suite "$3" "${extra[@]}" --list-groups 2>>out/stderr-"$1".log); rc=$?
  [ $rc -eq 0 ] || return 2
  grep -E '^[a-z0-9_]+/' <<<"$out" || true
}

# sample_run SIDE BACKEND SUITE N UNITS(comma list or "" = whole suite): one process, appended to sample N of the side.
sample_run() {
  local side=$1 be=$2 suite=$3 n=$4 units=$5 lab f tmp rc sel=()
  lab=$(lab_of "$side"); f=out/$lab-$be-$suite-$side$n
  in_time || { echo "benchab: $side $be $suite sample $n not started: deadline"; return 0; }
  tmp=$(mktemp -d)
  [ -n "$units" ] && sel=(--group "$units" --warm)
  LUMA_BACKEND=$be LUMA_CACHE_DIR=$(cache_of "$side") LUMA_BENCH_REF=$lab "$(bin_of "$side")" --suite "$suite" "${extra[@]}" "${sel[@]}" \
    --json "$tmp/j" > "$tmp/t" 2>> out/stderr-"$side".log
  rc=$?
  if [ $rc -eq 3 ]; then rm -rf "$tmp"; return 0; fi   # no case in this unit on this side (the other side has it)
  [ $rc -eq 0 ] || [ $rc -eq 1 ] || echo "benchab: $side $be $suite sample $n exit $rc (${units:-whole suite})"
  [ -f "$tmp/j" ] && cat "$tmp/j" >> "$f.jsonl"
  { [ -n "$units" ] && echo "== $units"; cat "$tmp/t"; } >> "$f.txt"
  ran=$((ran + 1))
  rm -rf "$tmp"
}

# calibrate_once BACKEND: luma measures its cost model (launch cost, bandwidth, ...) once per cache dir, and the planner's choices
# follow it, so two sides that each measured their own differ for the whole job by the noise of that one measurement (consistent
# 5-15 % on small cases, the same in every round, which no interleaving cancels). The B binary measures once, on a warm GPU, in a
# scratch dir; the model is copied into both side caches. Without a stored model (a contended measurement) both sides run on the
# fixed fallback numbers (LUMA_CALIBRATE=0).
calibrate_once() {
  local cs
  cs=$(mktemp -d)
  rm -f "$CA"/cost-*.bin "$CA"/contended-*.bin "$CA"/fma-*.bin "$CB"/cost-*.bin "$CB"/contended-*.bin "$CB"/fma-*.bin
  LUMA_BACKEND=$1 LUMA_CACHE_DIR=$cs "$B" --suite none --warm >/dev/null 2>>out/stderr-b.log \
    || LUMA_BACKEND=$1 LUMA_CACHE_DIR=$cs "$B" --suite none >/dev/null 2>>out/stderr-b.log
  if compgen -G "$cs/cost-*.bin" >/dev/null; then
    cp "$cs"/cost-*.bin "$cs"/fma-*.bin "$CA"/ 2>/dev/null; cp "$cs"/cost-*.bin "$cs"/fma-*.bin "$CB"/ 2>/dev/null
    echo "$(date +%T) $1: one cost model for both sides ($(ls "$cs" | tr '\n' ' '))"
  else
    export LUMA_CALIBRATE=0
    echo "$(date +%T) $1: no cost model measured, both sides run on the fallback numbers"
  fi
  rm -rf "$cs"
}

for be in $BACKENDS; do
  unset LUMA_CALIBRATE
  calibrate_once "$be"
  command -v nvidia-smi >/dev/null && [ "$be" = cuda -o "$be" = wgpu ] \
    && echo "$(date +%T) $be GPU: $(nvidia-smi --query-gpu=name,pstate,clocks.sm,clocks.max.sm,temperature.gpu --format=csv,noheader 2>&1 | head -1)"
  for suite in ${SUITES//,/ }; do
    ua=$(list_units a "$be" "$suite"); ra=$?
    ub=$(list_units b "$be" "$suite"); rb=$?
    if [ "${LEGACY:-0}" = 1 ] || [ $ra -ne 0 ] || [ $rb -ne 0 ]; then
      # A binary without --list-groups / --group (an older ref): whole-suite runs, A B B A per round.
      [ "${LEGACY:-0}" = 1 ] || echo "benchab: $suite: a binary lacks --list-groups, whole-suite A/B (drift-prone)"
      for r in $(seq 1 "$ROUNDS"); do
        if [ $((r % 2)) = 1 ]; then o='a b b a'; else o='b a a b'; fi
        k_a=0 k_b=0
        for side in $o; do
          if [ "$side" = a ]; then k_a=$((k_a + 1)); n=$((2 * (r - 1) + k_a)); else k_b=$((k_b + 1)); n=$((2 * (r - 1) + k_b)); fi
          sample_run "$side" "$be" "$suite" "$n" ""
        done
        echo "$(date +%T) $be $suite round $r done (whole suite)"
      done
      continue
    fi
    # B's order, then the units only A has.
    mapfile -t units < <({ echo "$ub"; echo "$ua"; } | grep -v '^$' | awk '!seen[$0]++')
    if [ ${#units[@]} -eq 0 ]; then echo "benchab: $be $suite: no unit selected${ONLY:+ by --only '$ONLY'}"; continue; fi
    chunk=${CHUNK:-0}; [ "$chunk" -gt 0 ] 2>/dev/null || chunk=$(( (${#units[@]} + 47) / 48 ))
    chunks=()
    for ((i = 0; i < ${#units[@]}; i += chunk)); do
      c=""; for ((j = i; j < i + chunk && j < ${#units[@]}; j++)); do c+="${c:+,}${units[j]}"; done
      chunks+=("$c")
    done
    total_units=$((total_units + ${#units[@]}))
    echo "$(date +%T) $be $suite: ${#units[@]} units in ${#chunks[@]} steps, $ROUNDS round(s)"
    for r in $(seq 1 "$ROUNDS"); do
      ci=0
      for c in "${chunks[@]}"; do
        if [ $(((r + ci) % 2)) = 1 ]; then o='a b b a'; else o='b a a b'; fi
        k_a=0 k_b=0
        for side in $o; do
          if [ "$side" = a ]; then k_a=$((k_a + 1)); n=$((2 * (r - 1) + k_a)); else k_b=$((k_b + 1)); n=$((2 * (r - 1) + k_b)); fi
          sample_run "$side" "$be" "$suite" "$n" "$c"
        done
        ci=$((ci + 1))
      done
      echo "$(date +%T) $be $suite round $r done"
    done
  done
  command -v nvidia-smi >/dev/null && [ "$be" = cuda -o "$be" = wgpu ] \
    && echo "$(date +%T) $be GPU at the end: $(nvidia-smi --query-gpu=pstate,clocks.sm,clocks.max.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader 2>&1 | head -1)"
done
echo "benchab: SKIPPED / FAILED cases, runs per side-backend-suite:"
grep -H -E '^(SKIPPED|FAILED): ' out/*.txt 2>/dev/null | sed -E 's|^out/(.*)-[ab][0-9]+[.]txt:|\1 |' | sort | uniq -c | cut -c1-400 || true
if [ "$ran" -eq 0 ]; then
  echo "benchab: nothing ran: no suite/group matched${ONLY:+ --only '$ONLY'} (a group is picked by suite/group, a case by suite/group/case or group/case)"
  exit 1
fi
