# luma-bench

Workload benchmark suites for Luma (`--list` shows them). Every case records first-call cost,
median/min, GPU time, launches and memory, and is verified against a host reference.

The `sparse` suite covers sparse linear algebra and graph workloads (SpMV, CG, format conversions,
PageRank, BFS, connected components, shortest paths, SpMM); `LUMA_BENCH_KERNELS=1` lists the kernels
of one call under each of its rows.

## Tools

Run a luma ref through the GPU queue (or locally on CPU) and compare two runs:

```sh
tools/benchref.sh REF HOST [SUITES] [BACKENDS] [--quick] [--cold]
#  REF: git ref | wt (working tree) | path     HOST: local|remote|any|cpu
#  prints the gpuq job id; JSON lines land in $GPUQ_DIR/done/<id>/out/<ref>-<backend>.jsonl
#  (default ~/.cache/bowshock-gpuq). cpu host runs inline into ~/.cache/luma-benchref/<ref>-cpu/out/.
#  --cold: empty LUMA_CACHE_DIR (cold kernel cache).
tools/benchref.sh v0.4.0 local - "cuda wgpu"
tools/benchref.sh wt local ml cuda --quick

tools/benchcmp.py BASE.jsonl NEW.jsonl [--threshold 0.05] [--metric median_ms|min_ms|gpu_ms|first_ms|mem_mib] [--all]
tools/benchcmp.py --base base-a.jsonl base-b.jsonl --new 'new-*.jsonl'   # best (min) of repeats per side
```

`benchcmp.py` flags REGRESSION / IMPROVED (median AND min beyond the threshold), first-call
slowdowns >25%, memory growth >10% and launch-count increases; prints per-suite geometric-mean
ratios and exits 1 on any REGRESSION.
