#!/usr/bin/env python3
"""Compare luma-bench JSON-lines results.

usage: benchcmp.py BASE.jsonl [BASE2 ...] NEW.jsonl [NEW2 ...] [--threshold 0.05]
                   [--metric median_ms|min_ms|gpu_ms|first_ms|mem_mib] [--all]

Several files per side: pass `--base A B C --new D E` or one argument per side
(each may be a glob). Cases are matched by (suite, case, size, backend). A case measured
in several files of a side (the rounds of tools/benchab.sh: <label>-<backend>-<suite>-<side><n>.jsonl,
sample n of both sides adjacent in time) is reported as the median over its samples
(`--agg min` keeps the best one). A REGRESSION / IMPROVED needs median AND min beyond the threshold
and, when each side has several samples, every pair of samples (file n of base against file n of
new) to agree in direction and the median ratio to exceed the spread of the samples of a side
((max - min) / median, the larger of the two sides); a change beyond the threshold that fails this is
flagged NOISY and not counted.
Exit status 1 if any REGRESSION.
"""
import argparse, glob, json, math, re, statistics, sys
from collections import defaultdict

METRICS = ("median_ms", "min_ms", "gpu_ms", "first_ms", "mem_mib")


def natural(path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path)]


def load(patterns):
    """{key: [record, ...]} with the records of a case in file order (natural sort: a2 before a10)."""
    files = []
    for p in patterns:
        g = sorted(glob.glob(p), key=natural)
        files += g if g else [p]
    out = defaultdict(list)
    for f in files:
        with open(f) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln.startswith("{"):
                    continue
                r = json.loads(ln)
                k = (r.get("suite", ""), r["case"], str(r.get("size", "")), r.get("backend", ""))
                m = re.search(r"-[ab](\d+)\.jsonl$", f)
                r["_n"] = int(m.group(1)) if m else len(out[k])
                out[k].append(r)
    return out


def aggregate(samples, agg):
    """One record from the samples of a case on a side: the median (or min) of every metric."""
    pick = statistics.median if agg == "median" else min
    rec = dict(samples[-1])
    for m in METRICS:
        vals = [r[m] for r in samples if r.get(m) is not None]
        rec[m] = pick(vals) if vals else None
    ls = [r.get("launches") for r in samples if r.get("launches") is not None]
    rec["launches"] = min(ls) if ls else None
    return rec


def spread(samples, met):
    """(max - min) / median of a metric over the samples of one side (0 for fewer than two)."""
    vals = [r[met] for r in samples if r.get(met)]
    if len(vals) < 2:
        return 0.0
    return (max(vals) - min(vals)) / statistics.median(vals)


def fmt(v):
    return "-" if v is None else f"{v:.3f}" if v < 100 else f"{v:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="BASE NEW (one glob/file each) unless --base/--new used")
    ap.add_argument("--base", nargs="+")
    ap.add_argument("--new", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--metric", choices=METRICS, default="median_ms")
    ap.add_argument("--all", action="store_true", help="show unchanged cases too")
    ap.add_argument("--agg", choices=("median", "min"), default="median", help="how the samples of a case on a side are combined")
    a = ap.parse_args()
    if a.base and a.new:
        bp, np_ = a.base, a.new
    elif len(a.files) == 2:
        bp, np_ = [a.files[0]], [a.files[1]]
    else:
        ap.error("need BASE NEW (or --base ... --new ...)")
    base_s, new_s = load(bp), load(np_)
    base = {k: aggregate(v, a.agg) for k, v in base_s.items()}
    new = {k: aggregate(v, a.agg) for k, v in new_s.items()}
    th, met = a.threshold, a.metric
    rows, per_suite = [], defaultdict(list)
    nreg = nimp = nnoisy = 0
    for k in sorted(set(base) & set(new)):
        b, n = base[k], new[k]
        bv, nv = b.get(met), n.get(met)
        if not bv or nv is None or bv <= 0:
            continue
        ratio = nv / bv
        flags = []
        if met in ("median_ms", "min_ms"):
            rm = (n["median_ms"] / b["median_ms"]) if b.get("median_ms") else ratio
            rn = (n["min_ms"] / b["min_ms"]) if b.get("min_ms") else ratio
        else:
            rm = rn = ratio
        slower, faster = rm > 1 + th and rn > 1 + th, rm < 1 - th and rn < 1 - th
        if slower or faster:
            # Sample n of base against sample n of new (taken next to it in time), and the spread of the samples of a side.
            bs, ns = base_s[k], new_s[k]
            pm = "median_ms" if met in ("median_ms", "min_ms") else met
            by_n = {x["_n"]: x for x in bs}
            pairs = [y[pm] / by_n[y["_n"]][pm] for y in ns if y["_n"] in by_n and by_n[y["_n"]].get(pm) and y.get(pm) is not None]
            sp = max(spread(bs, pm), spread(ns, pm))
            if slower:
                agree = all(p > 1 for p in pairs)
                clear = rm - 1 > sp
            else:
                agree = all(p < 1 for p in pairs)
                clear = 1 - rm > sp
            if agree and clear:
                flags.append("REGRESSION" if slower else "IMPROVED")
                nreg += slower
                nimp += faster
            else:
                nnoisy += 1
                why = "rounds disagree" if not agree else f"spread {sp:.0%}"
                flags.append(f"NOISY x{rm:.2f} ({why})")
        bf, nf = b.get("first_ms"), n.get("first_ms")
        if bf and nf is not None and nf / bf > 1.25 and met != "first_ms":
            flags.append(f"FIRST x{nf / bf:.2f}")
        bm, nm = b.get("mem_mib"), n.get("mem_mib")
        if bm and nm is not None and nm / bm > 1.10 and met != "mem_mib":
            flags.append(f"MEM {bm:.0f}->{nm:.0f}MiB")
        bl, nl = b.get("launches"), n.get("launches")
        if bl is not None and nl is not None and nl > bl:
            flags.append(f"LAUNCHES {bl}->{nl}")
        per_suite[k[0]].append(ratio)
        if flags or a.all:
            rows.append((ratio, k, bv, nv, flags))
    rows.sort(key=lambda r: -r[0])
    print(f"metric: {met}, threshold {th:.0%}\n")
    print("| case | size | backend | base | new | ratio | flags |")
    print("|---|---|---|---:|---:|---:|---|")
    for ratio, k, bv, nv, flags in rows:
        print(f"| {k[1]} | {k[2]} | {k[3]} | {fmt(bv)} | {fmt(nv)} | {ratio:.3f} | {', '.join(flags)} |")
    print()
    for s, rs in sorted(per_suite.items()):
        gm = math.exp(sum(math.log(r) for r in rs) / len(rs))
        print(f"suite {s or '?'}: geomean ratio {gm:.3f} over {len(rs)} cases")
    only_b, only_n = set(base) - set(new), set(new) - set(base)
    total = sum(map(len, per_suite.values()))
    sizes = sorted({len(v) for v in base_s.values()}), sorted({len(v) for v in new_s.values()})
    print(f"samples per case: base {sizes[0]}, new {sizes[1]} ({a.agg} over them); {nnoisy} noisy (not counted)")
    print(f"\n{total} compared, "
          f"{nreg} regression(s), {nimp} improved, {len(only_b)} only in base, {len(only_n)} only in new")
    sys.exit(1 if nreg else 0)


if __name__ == "__main__":
    main()
