#!/usr/bin/env bash
# Run luma-bench for a luma ref through the GPU queue (or locally on the CPU backend).
# Usage: benchref.sh REF HOST [SUITES] [BACKENDS] [--quick] [--cold]
#   REF      git ref of this repo, `wt` (working tree of the repo holding this script) or a path to a luma tree
#   HOST     local | remote | any (gpuq host) | cpu (no gpuq: run here with LUMA_DEVICE=cpu)
#   SUITES   comma list (default: all; `-` = all)       BACKENDS  space/comma list (default "cuda wgpu"; cpu host ignores it)
#   --quick  quick sizes     --cold  empty LUMA_CACHE_DIR (measures cold kernel-cache first calls)
# Results: cpu: <scratch>/out/<ref>-cpu.jsonl; gpuq: $GPUQ_DIR/done/<id>/out/<ref>-<backend>.jsonl; compare with tools/benchcmp.py.
set -euo pipefail
pos=() quick="" cold=0
for a in "$@"; do
  case $a in --quick) quick=--quick ;; --cold) cold=1 ;; *) pos+=("$a") ;; esac
done
[ ${#pos[@]} -ge 2 ] || { sed -n 2,9p "$0"; exit 2; }
ref=${pos[0]} host=${pos[1]} suites=${pos[2]:--} backends=${pos[3]:-cuda wgpu}
backends=${backends//,/ }
here=$(cd "$(dirname "$0")/.." && pwd)
GPUQ=${GPUQ:-$HOME/Documents/bowshock/tools/gpuq/gpuq}

if [ "$ref" = wt ]; then
  src=$here; sha=$(git -C "$here" rev-parse --short HEAD)$(git -C "$here" diff --quiet HEAD -- 2>/dev/null || echo +wt); label=wt
elif [ -d "$ref" ]; then
  src=$(cd "$ref" && pwd); sha=$(git -C "$src" rev-parse --short HEAD 2>/dev/null || echo path); label=$(basename "$src")
else
  src=""; sha=$(git -C "$here" rev-parse --short "$ref"); label=$ref
fi
label=${label//\//_}

# scratch git dir (gpuq snapshots a git repo, excluding target/out/.git)
snap=${BENCHREF_DIR:-$HOME/.cache/luma-benchref}/$label-$host
rm -rf "$snap"; mkdir -p "$snap"
if [ -n "$src" ]; then
  rsync -a --exclude target --exclude .git --exclude out "$src"/ "$snap"/
else
  git -C "$here" archive "$ref" | tar -x -C "$snap"
fi
# Fresh mtimes: an archive carries the commit's times, which can be older than the artifacts a
# shared target dir holds for another ref, and cargo would then reuse those.
find "$snap" -type f -exec touch {} +
git -C "$snap" init -q && git -C "$snap" add -A >/dev/null && git -C "$snap" -c user.name=b -c user.email=b@b commit -qm "luma $sha" >/dev/null

sargs=(); [ "$suites" = - ] || sargs=(--suite "$suites")
coldline=""; [ $cold = 1 ] && coldline='export LUMA_CACHE_DIR=$(mktemp -d)'
run="export PATH=\$HOME/.cargo/bin:\$PATH LUMA_BENCH_REF=$sha
mkdir -p out
$coldline
cargo build --release -p luma-bench > out/build.log 2>&1 || { tail -30 out/build.log; echo 'benchref: build failed'; exit 1; }
tail -2 out/build.log
BIN=\${CARGO_TARGET_DIR:-target}/release/luma-bench"
if [ "$host" = cpu ]; then
  loop="LUMA_DEVICE=cpu CUDA_VISIBLE_DEVICES=-1 VK_ICD_FILENAMES=/nonexistent \$BIN ${sargs[*]} $quick --json out/$label-cpu.jsonl"
  cmd="$run
$loop"
  echo "benchref: cpu run of $label ($sha) in $snap"
  (cd "$snap" && env -u LUMA_BACKEND -u LUMA_CACHE_DIR CARGO_TARGET_DIR="${BENCHREF_TARGET:-$HOME/.cache/luma-benchref/target}" bash -c "$cmd")
  echo "results: $snap/out/$label-cpu.jsonl"
  exit 0
fi
cmd="$run
for be in $backends; do echo \"== backend \$be\"; LUMA_BACKEND=\$be \$BIN ${sargs[*]} $quick --json out/$label-\$be.jsonl; done"
cd "$snap"
id=$("$GPUQ" submit --host "$host" --prio 0 --name "bench-$label-$host" --timeout 3600 -- bash -c "$cmd")
echo "job: $id  ($label = $sha, host $host, backends: $backends)"
echo "wait:    $GPUQ wait $id"
echo "results: ${GPUQ_DIR:-$HOME/.cache/bowshock-gpuq}/done/$id/out/$label-<backend>.jsonl (log: $GPUQ log $id); compare: tools/benchcmp.py BASE NEW"
