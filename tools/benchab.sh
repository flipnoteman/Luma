#!/usr/bin/env bash
# Interleaved A/B of two luma refs in one GPU-queue job (or locally on the CPU backend): both binaries are built first, then
# the two alternate unit by unit (a unit is a luma-bench group of cases, `luma-bench --list-groups`): A B B A for one unit, B A A B
# for the next, ..., each run after a 400 ms GPU warm-up, so the clock, thermal and host-load drift of a long job hits both sides
# alike (alternating whole suites let it hit whichever side ran in the slow state: 10-30 unrelated 1.05-1.5x "regressions").
# Both sides share one cost model (calibrated once on the warm GPU; a model measured per side differs for the whole job by the noise
# of one measurement) and every run is pinned to the same 8 cores (BENCHAB_PIN=cpu list | off).
# Usage: benchab.sh BASE NEW HOST [SUITES] [BACKENDS] [--rounds N] [--quick] [--timeout SECS] [--only SUBSTR] [--chunk N] [--aa]
#   BASE/NEW  git refs of this repo or paths to luma trees
#   HOST      local | remote | any (gpuq host) | cpu (no gpuq: run here with LUMA_DEVICE=cpu)
#   SUITES    comma list (default: ml,cfd,datasci,compute,jit,sparse)   BACKENDS  default "cuda wgpu"
#   --rounds  A/B pairs per side and unit (default 1: A B B A, two samples per side); benchcmp.py takes the median over the samples
#   --only    luma-bench --only PATTERN: a group by `suite/group`, a case by `suite/group/case` or the short `group/case`
#             (`matmul/square`), several with commas, `X$` exact; a job that selects no case fails loudly
#   --aa      A/A test: NEW's binary is a copy of BASE's (the flags then show the tool's noise floor)
#   --chunk   units per alternation step (default: auto, at most 48 steps per suite; 1 = every group alternates on its own)
#   --timeout gpuq timeout of the job (default 7200). gpuq copies out/ back only when the job ends by itself, so the
#             job stops starting runs 300 s before the timeout and ends normally (the results so far are kept).
# luma-bench exits 1 when a case is skipped (its check failed) or failed; that round still has the JSON of every case
# that ran. The runs' stdout is kept next to it (out/<label>-<backend>-<suite>-<side><n>.txt) and the job log ends with
# a count of the SKIPPED / FAILED lines per side, backend and suite.
# Results: out/<label>-<backend>-<suite>-<side><n>.jsonl in the job's out dir (cpu: the scratch dir), sample n (1 .. 2 x rounds)
# of every unit of the suite; compare with tools/benchcmp.py --base 'DIR/<base>-cuda-*' --new 'DIR/<new>-cuda-*'.
set -euo pipefail
pos=() aa="" quick="" only="" chunk=0 rounds=1 timeout=7200
while [ $# -gt 0 ]; do
  case $1 in --quick) quick=1 ;; --only) only=$2; shift ;; --chunk) chunk=$2; shift ;; --rounds) rounds=$2; shift ;; --aa) aa=1 ;; --timeout) timeout=$2; shift ;; *) pos+=("$1") ;; esac
  shift
done
[ ${#pos[@]} -ge 3 ] || { sed -n 2,25p "$0"; exit 2; }
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
# --aa: B is a copy of the A binary, so every change the comparison shows is the noise floor of the tool itself.
[ -z "$aa" ] || cp "$snap/bin/a" "$snap/bin/b"
# The job: tools/benchab-job.sh (shipped in the snapshot) alternates the two binaries unit by unit; its input is the environment.
cp "$here/tools/benchab-job.sh" "$snap/job.sh"
git -C "$snap" init -q && git -C "$snap" add -A >/dev/null && git -C "$snap" -c user.name=b -c user.email=b@b commit -qm "ab $la $lb" >/dev/null

if [ -n "$only" ]; then
  n=$(LUMA_DEVICE=cpu CUDA_VISIBLE_DEVICES=-1 VK_ICD_FILENAMES=/nonexistent "$snap/bin/b" --suite "$suites" --only "$only" --list-groups 2>/dev/null | grep -c -E '^[a-z0-9_]+/' || true)
  [ "${n:-0}" -gt 0 ] || echo "benchab: warning: --only '$only' selects no group of $suites on the CPU backend (the job fails if it selects none on the GPU either)" >&2
fi
if [ "$host" = cpu ]; then
  echo "benchab: cpu A/B of $la vs $lb in $snap"
  (cd "$snap" && env -u LUMA_BACKEND -u LUMA_CACHE_DIR LUMA_DEVICE=cpu CUDA_VISIBLE_DEVICES=-1 VK_ICD_FILENAMES=/nonexistent \
    A="$PWD/bin/a" B="$PWD/bin/b" LA="$la" LB="$lb" BACKENDS=cpu SUITES="$suites" ROUNDS="$rounds" CHUNK="$chunk" QUICK="$quick" ONLY="$only" PIN="${BENCHAB_PIN:-}" DEADLINE=$((timeout - 300)) bash job.sh)
  echo "results: $snap/out/"
  exit 0
fi
cd "$snap"
id=$("$GPUQ" submit --host "$host" --prio "$PRIO" --name "ab-$la-$lb-$host" --timeout "$timeout" -- \
  env A='bin/a' B='bin/b' LA="$la" LB="$lb" BACKENDS="$backends" SUITES="$suites" ROUNDS="$rounds" CHUNK="$chunk" QUICK="$quick" ONLY="$only" PIN="${BENCHAB_PIN:-}" DEADLINE=$((timeout - 300)) bash job.sh)
echo "job: $id  (A $la vs B $lb, host $host, suites $suites, backends: $backends, rounds $rounds)"
echo "results: ${GPUQ_DIR:-$HOME/.cache/bowshock-gpuq}/done/$id/out/ ; compare: tools/benchcmp.py --base 'OUT/$la-cuda-*' --new 'OUT/$lb-cuda-*'"
