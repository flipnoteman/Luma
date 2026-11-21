#!/usr/bin/env bash
# Interleaved A/B of two luma refs in one GPU-queue job (or locally on the CPU backend): both
# binaries are built first, then every suite runs A B B A (round 1), B A A B (round 2), ..., so
# clock, thermal and host-load drift hit both sides alike.
# Usage: benchab.sh BASE NEW HOST [SUITES] [BACKENDS] [--rounds N] [--quick]
#   BASE/NEW  git refs of this repo or paths to luma trees
#   HOST      local | remote | any (gpuq host) | cpu (no gpuq: run here with LUMA_DEVICE=cpu)
#   SUITES    comma list (default: ml,cfd,datasci,compute,jit,sparse)   BACKENDS  default "cuda wgpu"
#   --rounds  A/B pairs per side and suite (default 1: A B B A, two samples per side)
# Results: out/<label>-<backend>-<side><n>.jsonl in the job's out dir (cpu: the scratch dir);
# compare with tools/benchcmp.py --base 'DIR/<base>-cuda-*' --new 'DIR/<new>-cuda-*'.
set -euo pipefail
pos=() quick="" rounds=1
while [ $# -gt 0 ]; do
  case $1 in --quick) quick=--quick ;; --rounds) rounds=$2; shift ;; *) pos+=("$1") ;; esac
  shift
done
[ ${#pos[@]} -ge 3 ] || { sed -n 2,11p "$0"; exit 2; }
base=${pos[0]} new=${pos[1]} host=${pos[2]} suites=${pos[3]:-ml,cfd,datasci,compute,jit,sparse} backends=${pos[4]:-cuda wgpu}
[ "$suites" = - ] && suites=ml,cfd,datasci,compute,jit,sparse
backends=${backends//,/ }
here=$(cd "$(dirname "$0")/.." && pwd)
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
# Fresh mtimes so a shared target dir never reuses another ref's artifacts.
find "$snap" -type f -exec touch {} +
git -C "$snap" init -q && git -C "$snap" add -A >/dev/null && git -C "$snap" -c user.name=b -c user.email=b@b commit -qm "ab $la $lb" >/dev/null

run="export PATH=\$HOME/.cargo/bin:\$PATH
mkdir -p out
T=\${CARGO_TARGET_DIR:-\$PWD/target}
for s in a b; do
  ref=\$([ \$s = a ] && echo $(sha_of "$base") || echo $(sha_of "$new"))
  (cd \$s && LUMA_BENCH_REF=\$ref cargo build --release -p luma-bench --target-dir \$T/ab-\$s > ../out/build-\$s.log 2>&1) || { tail -30 out/build-\$s.log; echo \"benchab: build \$s failed\"; exit 1; }
done
A=\$T/ab-a/release/luma-bench B=\$T/ab-b/release/luma-bench
one() { # side backend suite round
  local bin=\$A lab=$la; [ \$1 = b ] && bin=\$B lab=$lb
  LUMA_BENCH_REF=\$lab \$bin --suite \$3 $quick --json out/\$lab-\$2-\$1\$4.jsonl > /dev/null 2>> out/stderr-\$1.log || echo \"benchab: \$1 \$2 \$3 round \$4 exit \$?\"
}"
loop="for be in BACKENDS; do
  for suite in ${suites//,/ }; do
    for r in \$(seq 1 $rounds); do
      if [ \$((r % 2)) = 1 ]; then o='a b b a'; else o='b a a b'; fi
      n=0; for side in \$o; do n=\$((n + 1)); one \$side \$be \$suite \$r\$n; done
      echo \"\$(date +%T) \$be \$suite round \$r done\"
    done
  done
done"
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
id=$("$GPUQ" submit --host "$host" --prio 0 --name "ab-$la-$lb-$host" --timeout 7200 -- bash -c "$cmd")
echo "job: $id  (A $la vs B $lb, host $host, suites $suites, backends: $backends, rounds $rounds)"
echo "results: ${GPUQ_DIR:-$HOME/.cache/bowshock-gpuq}/done/$id/out/ ; compare: tools/benchcmp.py --base 'OUT/$la-cuda-*' --new 'OUT/$lb-cuda-*'"
