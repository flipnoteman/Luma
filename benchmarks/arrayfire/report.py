#!/usr/bin/env python3
"""Merge benchmark CSVs into markdown tables: python3 report.py results/*.csv > report.md"""
import csv
import sys
from collections import OrderedDict

WORKLOADS = ["heat", "converge", "life", "blackscholes", "mandelbrot", "fir"]
TITLES = {
    "heat": "Heat diffusion (periodic 5-point stencil)",
    "converge": "Heat diffusion with a convergence readback every step",
    "life": "Game of Life (torus, f32 cells)",
    "blackscholes": "Black-Scholes call prices",
    "mandelbrot": "Mandelbrot (array-level, fixed iterations)",
    "fir": "FIR filter (full linear convolution)",
}


EXCLUDED = {"custom_kernel", "lazy_chain", "per_step_cold", "eval_every_1_cold"}


def pairs(workload, variants):
    """(luma variant, arrayfire variant) pairs that do the same work."""
    out = []
    for v in variants:
        if v not in EXCLUDED:
            out.append((v, v))
    return out


def size_key(s):
    nums = [int(t) for t in s.replace("x", " ").replace("=", " ").replace(",", " ").split() if t.isdigit()]
    return nums


def fmt(ms):
    return "" if ms is None else f"{ms:.3f}"


def main(paths):
    # (workload, size, variant) -> {lib_backend: median_ms}
    data = OrderedDict()
    units = {}
    cols = []
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                col = f"{r['library']}-{r['backend']}"
                if col not in cols:
                    cols.append(col)
                key = (r["workload"], r["size"], r["variant"])
                data.setdefault(key, {})[col] = float(r["median_ms"])
                units[r["workload"]] = r["unit"]
    luma_cols = [c for c in cols if c.startswith("luma-")]
    af_cols = sorted((c for c in cols if not c.startswith("luma-")), key=lambda c: (c != "arrayfire-cuda", c))
    cols = luma_cols + af_cols
    luma = luma_cols[0] if luma_cols else None
    af = "arrayfire-cuda" if "arrayfire-cuda" in af_cols else (af_cols[0] if af_cols else None)
    af_name = "af-" + af.split("-", 1)[1] if af else "arrayfire"

    def ratio(af_ms, luma_ms):
        if af_ms is None or luma_ms is None:
            return ""
        return f"{af_ms / luma_ms:.2f}x"

    print("# Luma vs ArrayFire\n")
    print(f"Median time per run in ms (lower is better). Speedup = {af_name} ms / luma ms (>1 means Luma is faster).\n")
    for w in WORKLOADS:
        keys = [k for k in data if k[0] == w]
        if not keys:
            continue
        print(f"## {TITLES[w]} ({units[w]} = work/s)\n")
        print("| size | variant | " + " | ".join(cols) + f" | luma vs {af_name} |")
        print("|" + "---|" * (len(cols) + 3))
        keys.sort(key=lambda k: (size_key(k[1]), k[1]))
        by_size = OrderedDict()
        for k in keys:
            by_size.setdefault(k[1], []).append(k)
        for size, ks in by_size.items():
            for k in ks:
                d = data[k]
                print(f"| {size} | {k[2]} | " + " | ".join(fmt(d.get(c)) for c in cols) + f" | {ratio(d.get(af) if af else None, d.get(luma))} |")
        print("\n**Like for like** (equivalent variants; ratio = af ms / luma ms, >1 means Luma is faster)\n")
        print(f"| size | luma variant | luma ms | arrayfire variant | {af_name} ms | ratio af/luma |")
        print("|---|---|---|---|---|---|")
        for size, ks in by_size.items():
            variants = [k[2] for k in ks]
            for lv, av in pairs(w, variants):
                lm = data.get((w, size, lv), {}).get(luma)
                am = data.get((w, size, av), {}).get(af)
                if lm is None or am is None:
                    continue
                print(f"| {size} | {lv} | {fmt(lm)} | {av} | {fmt(am)} | {ratio(am, lm)} |")
        print("\n**Best of variants per size**\n")
        print(f"| size | best luma ms (variant) | best {af_name} ms (variant) | luma vs {af_name} |")
        print("|---|---|---|---|")
        for size, ks in by_size.items():
            def best(col):
                c = [(data[k][col], k[2]) for k in ks if col in data[k] and not (col == luma and k[2] in EXCLUDED)]
                return min(c) if c else None
            bl = best(luma) if luma else None
            ba = best(af) if af else None
            ls = f"{bl[0]:.3f} ({bl[1]})" if bl else ""
            as_ = f"{ba[0]:.3f} ({ba[1]})" if ba else ""
            print(f"| {size} | {ls} | {as_} | {ratio(ba[0] if ba else None, bl[0] if bl else None)} |")
        if w == "mandelbrot":
            print("\nNote: Luma's `custom_kernel` variant is a hand-written kernel with per-pixel early exit, not array code, "
                  "so it is excluded from the best-of-variants comparison.")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: report.py results/*.csv")
    main(sys.argv[1:])
