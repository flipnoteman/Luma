#!/usr/bin/env bash
# Interleaved A/B of two luma refs in one GPU-queue job (or locally on the CPU backend): both
# binaries are built first, then every suite runs A B B A (round 1), B A A B (round 2), ..., so
# clock, thermal and host-load drift hit both sides alike.
# Usage: benchab.sh BASE NEW HOST [SUITES] [BACKENDS] [--rounds N] [--quick] [--timeout SECS]
#   BASE/NEW  git refs of this repo or paths to luma trees
#   HOST      local | remote | any (gpuq host) | cpu (no gpuq: run here with LUMA_DEVICE=cpu)
#   SUITES    comma list (default: ml,cfd,datasci,compute,jit,sparse)   BACKENDS  default "cuda wgpu"
#   --rounds  A/B pairs per side and suite (default 1: A B B A, two samples per side)
#   --timeout gpuq timeout of the job (default 7200). gpuq copies out/ back only when the job ends by itself, so the
#             job stops starting runs 300 s before the timeout and ends normally (the results so far are kept).
# luma-bench exits 1 when a case is skipped (its check failed) or failed; that round still has the JSON of every case
# that ran. The runs' stdout is kept next to it (out/<label>-<backend>-<suite>-<side><n>.txt) and the job log ends with
# a count of the SKIPPED / FAILED lines per side, backend and suite.
# Results: out/<label>-<backend>-<suite>-<side><n>.jsonl in the job's out dir (cpu: the scratch dir);
# compare with tools/benchcmp.py --base 'DIR/<base>-cuda-*' --new 'DIR/<new>-cuda-*'.
set -euo pipefail
pos=() quick="" rounds=1 timeout=7200
while [ $# -gt 0 ]; do
  case $1 in --quick) quick=--quick ;; --rounds) rounds=$2; shift ;; --timeout) timeout=$2; shift ;; *) pos+=("$1") ;; esac
  shift
done
[ ${#pos[@]} -ge 3 ] || { sed -n 2,16p "$0"; exit 2; }
base=${pos[0]} new=${pos[1]} host=${pos[2]} suites=${pos[3]:-ml,cfd,datasci,compute,jit,sparse} backends=${pos[4]:-cuda wgpu}
[ "$suites" = - ] && suites=ml,cfd,datasci,compute,jit,sparse
backends=${backends//,/ }
here=$(cd "$(dirname "$0")/.." && pwd)
PRIO=${PRIO:-0}   # gpuq priority of the job (0-9, lower runs first): PRIO=5 for a bench that may wait
GPUQ=${GPUQ:-$HOME/Documents/bowshock/tools/gpuq/gpuq}

label_of() { if [ -d "$1" ]; then basename "$(cd "$1" && pwd)"; else echo "${1//\//_}"; fi; }
sha_of() { if [ -d "$1" ]; then git -C "$1" rev-parse --short HEAD 2>/dev/null || echo path; else git -C "$here" rev-parse --short "$1"; fi; }
la=$(label_of "$base") lb=$(label_of "$new")
[ "$la" != "$lb" ] || { echo "benchab: BASE and NEW have the same label $la"; exit 2; }
snap=${BENCHREF_DIR:-$HOME/.cache/luma-benchref}/ab-$la-$lb-$host
rm -rf "$snap"; mkdir -p "$snap/a" "$snap/b"
put() {
  if [ -d "$1" ]; then rsync -a --exclude target --exclude .git --exclude out "$1"/ "$2"/
  else git -C "$here" archive "$1" | tar -x -C "$2"; fi
}
put "$base" "$snap/a"; put "$new" "$snap/b"
# Build both binaries here (nice'd), before the job takes a GPU: the queue's prebuilder cannot build two trees into
# two target dirs, and a build inside the job keeps the GPU idle for minutes. The job ships only the two binaries.
T=${BENCHAB_TARGET:-$HOME/.cache/luma-benchref/target}
mkdir -p "$snap/bin" "$snap/out"
for s in a b; do
  ref=$([ $s = a ] && sha_of "$base" || sha_of "$new")
  # Fresh mtimes so the shared target dir never reuses another ref's artifacts.
  find "$snap/$s" -type f -exec touch {} +
  (cd "$snap/$s" && env -u CARGO_TARGET_DIR LUMA_BENCH_REF=$ref nice -n 10 cargo build --release -p luma-bench --target-dir "$T/ab-$s" > "$snap/out/build-$s.log" 2>&1) \
    || { tail -30 "$snap/out/build-$s.log"; echo "benchab: build $s failed"; exit 1; }
  cp "$T/ab-$s/release/luma-bench" "$snap/bin/$s"
  rm -rf "${snap:?}/$s"
done
git -C "$snap" init -q && git -C "$snap" add -A >/dev/null && git -C "$snap" -c user.name=b -c user.email=b@b commit -qm "ab $la $lb" >/dev/null

run="mkdir -p out
A=\$PWD/bin/a B=\$PWD/bin/b
# Each side gets its own kernel/pipeline cache, empty at the start of the job: a shared cache near its size cap
# makes context creation cost ~1 ms per cached MB and varies with whatever other runs left in it.
CA=\$(mktemp -d) CB=\$(mktemp -d)
one() { # side backend suite round
  local bin=\$A lab=$la cache=\$CA; [ \$1 = b ] && bin=\$B lab=$lb cache=\$CB
  [ \$SECONDS -lt $((timeout - 300)) ] || { echo \"benchab: \$1 \$2 \$3 round \$4 not started: deadline\"; return 0; }
  LUMA_CACHE_DIR=\$cache LUMA_BENCH_REF=\$lab \$bin --suite \$3 $quick --json out/\$lab-\$2-\$3-\$1\$4.jsonl > out/\$lab-\$2-\$3-\$1\$4.txt 2>> out/stderr-\$1.log || echo \"benchab: \$1 \$2 \$3 round \$4 exit \$?\"
}"
loop="for be in BACKENDS; do
  for suite in ${suites//,/ }; do
    for r in \$(seq 1 $rounds); do
      if [ \$((r % 2)) = 1 ]; then o='a b b a'; else o='b a a b'; fi
      n=0; for side in \$o; do n=\$((n + 1)); one \$side \$be \$suite \$r\$n; done
      echo \"\$(date +%T) \$be \$suite round \$r done\"
    done
  done
done
echo \"benchab: SKIPPED / FAILED cases, runs per side-backend-suite:\"
grep -H -E '^(SKIPPED|FAILED): ' out/*.txt | sed -E 's|^out/(.*)-[ab][0-9]+[.]txt:|\\1 |' | sort | uniq -c | cut -c1-400 || true"
if [ "$host" = cpu ]; then
  cmd="$run
export LUMA_DEVICE=cpu CUDA_VISIBLE_DEVICES=-1 VK_ICD_FILENAMES=/nonexistent
${loop/BACKENDS/cpu}"
  echo "benchab: cpu A/B of $la vs $lb in $snap"
  (cd "$snap" && env -u LUMA_BACKEND -u LUMA_CACHE_DIR CARGO_TARGET_DIR="${BENCHREF_TARGET:-$HOME/.cache/luma-benchref/target}" bash -c "$cmd")
  echo "results: $snap/out/"
  exit 0
fi
cmd="$run
${loop/BACKENDS/$backends}"
cmd=${cmd//'$bin --suite'/'LUMA_BACKEND=$2 $bin --suite'}
cd "$snap"
id=$("$GPUQ" submit --host "$host" --prio "$PRIO" --name "ab-$la-$lb-$host" --timeout "$timeout" -- bash -c "$cmd")
echo "job: $id  (A $la vs B $lb, host $host, suites $suites, backends: $backends, rounds $rounds)"
echo "results: ${GPUQ_DIR:-$HOME/.cache/bowshock-gpuq}/done/$id/out/ ; compare: tools/benchcmp.py --base 'OUT/$la-cuda-*' --new 'OUT/$lb-cuda-*'"
