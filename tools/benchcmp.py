#!/usr/bin/env python3
"""Compare luma-bench JSON-lines results.

usage: benchcmp.py BASE.jsonl [BASE2 ...] NEW.jsonl [NEW2 ...] [--threshold 0.05]
                   [--metric median_ms|min_ms|gpu_ms|first_ms|mem_mib] [--all]

Several files per side: pass `--base A B C --new D E` or one argument per side
(each may be a glob). Cases are matched by (suite, case, size, backend); repeated
records of one case on a side keep the best (min) value of every metric.
Exit status 1 if any REGRESSION (median AND min both slower than the threshold).
"""
import argparse, glob, json, math, sys
from collections import defaultdict

METRICS = ("median_ms", "min_ms", "gpu_ms", "first_ms", "mem_mib")


def load(patterns):
    files = []
    for p in patterns:
        g = sorted(glob.glob(p))
        files += g if g else [p]
    out = {}
    for f in files:
        with open(f) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln.startswith("{"):
                    continue
                r = json.loads(ln)
                k = (r.get("suite", ""), r["case"], str(r.get("size", "")), r.get("backend", ""))
                cur = out.get(k)
                if cur is None:
                    out[k] = dict(r)
                else:
                    for m in METRICS:
                        a, b = cur.get(m), r.get(m)
                        if a is None or (b is not None and b < a):
                            cur[m] = b
                    cur["launches"] = min(cur.get("launches") or 0, r.get("launches") or 0) or r.get("launches")
    return out


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
    a = ap.parse_args()
    if a.base and a.new:
        bp, np_ = a.base, a.new
    elif len(a.files) == 2:
        bp, np_ = [a.files[0]], [a.files[1]]
    else:
        ap.error("need BASE NEW (or --base ... --new ...)")
    base, new = load(bp), load(np_)
    th, met = a.threshold, a.metric
    rows, per_suite = [], defaultdict(list)
    nreg = nimp = 0
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
            if rm > 1 + th and rn > 1 + th:
                flags.append("REGRESSION"); nreg += 1
            elif rm < 1 - th and rn < 1 - th:
                flags.append("IMPROVED"); nimp += 1
        else:
            if ratio > 1 + th:
                flags.append("REGRESSION"); nreg += 1
            elif ratio < 1 - th:
                flags.append("IMPROVED"); nimp += 1
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
    print(f"\n{total} compared, "
          f"{nreg} regression(s), {nimp} improved, {len(only_b)} only in base, {len(only_n)} only in new")
    sys.exit(1 if nreg else 0)


if __name__ == "__main__":
    main()
