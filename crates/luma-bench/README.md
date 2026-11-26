# luma-bench

Workload benchmark suites for Luma (`--list` shows them). Every case records first-call cost,
median/min, GPU time, launches and memory, and is verified against a host reference.

The `sparse` suite covers sparse linear algebra and graph workloads (SpMV, CG, format conversions,
PageRank, BFS, connected components, shortest paths, SpMM); `LUMA_BENCH_KERNELS=1` lists the kernels
of one call under each of its rows. The other `LUMA_BENCH_*` variables are in the guide's environment variables page.

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
tools/benchcmp.py --base base-a.jsonl base-b.jsonl --new 'new-*.jsonl'   # median of the files of a side (--agg min: the best)
```

A/B of two refs: `tools/benchab.sh BASE NEW HOST [SUITES] [BACKENDS] [--rounds N] [--only PATTERN] [--chunk N]` builds both
binaries and alternates them group by group (`--list-groups`, `--group`; `--warm` warms the GPU first) in one queue job: A B B A for
one group, B A A B for the next, so power-state and clock drift hits both sides alike. Sample n of a side lands in
`<label>-<backend>-<suite>-<side><n>.jsonl`; `benchcmp.py` pairs sample n of both sides.
`--only` picks a group by `suite/group`, a case by `suite/group/case` or `group/case`; a run that selects no case exits 3.

`benchcmp.py` flags REGRESSION / IMPROVED (median AND min beyond the threshold; with several samples per side also every
pair of samples agreeing in direction and the change exceeding the spread of a side's samples, else NOISY), first-call
slowdowns >25%, memory growth >10% and launch-count increases; prints per-suite geometric-mean
ratios and exits 1 on any REGRESSION.
