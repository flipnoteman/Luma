#!/usr/bin/env bash
# Usage: AF_PATH=<arrayfire prefix> ./run.sh [--quick]
# Builds and runs luma-bench, then af-bench once per backend in AF_BACKENDS (default "cuda opencl"),
# sequentially, and writes results/report.md. See README.md.
set -euo pipefail
cd "$(dirname "$0")"

quick=()
case "${1:-}" in
    "") ;;
    --quick) quick=(--quick) ;;
    *) echo "usage: AF_PATH=<arrayfire prefix> $0 [--quick]" >&2; exit 2 ;;
esac

if [[ -z "${AF_PATH:-}" || ! -f "$AF_PATH/include/arrayfire.h" ]]; then
    echo "error: AF_PATH must be set to the ArrayFire install prefix (the directory containing include/arrayfire.h)." >&2
    echo "       Current value: '${AF_PATH:-}'. See benchmarks/arrayfire/README.md for how to install ArrayFire." >&2
    exit 1
fi
if [[ -d "$AF_PATH/lib64" ]]; then
    af_lib="$AF_PATH/lib64"
elif [[ -d "$AF_PATH/lib" ]]; then
    af_lib="$AF_PATH/lib"
else
    echo "error: neither \$AF_PATH/lib64 nor \$AF_PATH/lib exists (AF_PATH=$AF_PATH)." >&2
    exit 1
fi
export AF_PATH
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C target-cpu=native -C link-arg=-Wl,-rpath,$af_lib"
export LD_LIBRARY_PATH="$af_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cargo build --release
mkdir -p results
rm -f results/*.csv results/report.md
bin=target/release

"$bin/luma-bench" "${quick[@]}" > results/luma.csv
for b in ${AF_BACKENDS:-cuda opencl}; do
    if ! "$bin/af-bench" --backend "$b" "${quick[@]}" > "results/af-$b.csv"; then
        echo "warning: af-bench backend '$b' failed; skipping it." >&2
        rm -f "results/af-$b.csv"
    fi
    if ! AF_CUDA_MAX_JIT_LEN=256 AF_OPENCL_MAX_JIT_LEN=256 "$bin/af-bench" --backend "$b" --label "$b-tuned" "${quick[@]}" > "results/af-$b-tuned.csv"; then
        echo "warning: tuned af-bench backend '$b' failed; skipping it." >&2
        rm -f "results/af-$b-tuned.csv"
    fi
done

python3 report.py results/*.csv > results/report.md
cat results/report.md
