#!/usr/bin/env python3
"""Env-var ledger audit (W0 part B).

Collects every LUMA_[A-Z0-9_]+ token in the repo's sources and docs and checks it against
tools/env-vars.txt (NAME<TAB>kind<TAB>owner<TAB>removal<TAB>default). Prints one offending name per
line and exits 1 if
  1. a token is not in the ledger,
  2. a ledger entry's token appears nowhere (stale), or
  3. a `dev` entry has no removal.
Usage: envaudit.py [--root DIR]   (DIR defaults to the repo root; used by the self-test)
"""
import os
import re
import sys

TOKEN = re.compile(rb"LUMA_[A-Z0-9_]+")
SCAN = ["src", "crates", "tests", "tools", "benches", "examples", "guide", "README.md", "CHANGELOG.md"]
SKIP_DIRS = {"target", ".git", "node_modules", "out", "book"}
LEDGER = os.path.join("tools", "env-vars.txt")
KINDS = {"user", "dev", "test", "internal"}


def tokens(root):
    found = set()
    for entry in SCAN:
        top = os.path.join(root, entry)
        if os.path.isfile(top):
            files = [top]
        else:
            files = []
            for d, dirs, names in os.walk(top):
                dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
                files += [os.path.join(d, n) for n in names]
        for f in files:
            if os.path.abspath(f) == os.path.abspath(os.path.join(root, LEDGER)):
                continue
            try:
                with open(f, "rb") as fh:
                    data = fh.read()
            except OSError:
                continue
            if b"\0" in data[:4096]:
                continue
            found.update(m.decode() for m in TOKEN.findall(data))
    return found


def ledger(root):
    entries = {}
    bad = []
    with open(os.path.join(root, LEDGER)) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.split("\t")
            name = cols[0]
            entries[name] = cols
            if len(cols) < 5 or cols[1] not in KINDS or not cols[2]:
                bad.append(name)
    return entries, bad


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if len(sys.argv) == 3 and sys.argv[1] == "--root":
        root = sys.argv[2]
    found = tokens(root)
    entries, malformed = ledger(root)
    offenders = set(malformed)
    offenders |= found - entries.keys()  # 1. unregistered
    offenders |= entries.keys() - found  # 2. stale
    offenders |= {n for n, c in entries.items() if c[1:2] == ["dev"] and (len(c) < 4 or not c[3].strip())}  # 3
    for name in sorted(offenders):
        print(name)
    return 1 if offenders else 0


if __name__ == "__main__":
    sys.exit(main())
