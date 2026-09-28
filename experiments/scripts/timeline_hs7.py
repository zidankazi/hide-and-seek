"""Summarize training logs in iteration buckets and print TSV.

Usage: python timeline_hs7.py <log> [more_logs...] [bucket_iters]"""
import re
import sys

logs = [a for a in sys.argv[1:] if not a.isdigit()]
digits = [a for a in sys.argv[1:] if a.isdigit()]
BUCKET = int(digits[0]) if digits else 200

FIELDS = ("hider", "seeker", "barr", "elev", "rlock")
print("run\tbucket_start_iter\tsteps\t" + "\t".join(FIELDS))
for log in logs:
    buckets = {}
    with open(log) as f:
        for line in f:
            if not line.startswith("Iter "):
                continue
            m = re.match(r"Iter (\d+) \| Steps (\d+)", line)
            if not m:
                continue
            it, steps = int(m.group(1)), int(m.group(2))
            vals = {}
            for field in FIELDS:
                fm = re.search(rf"{field}=([+-]?[\d.]+|nan)", line)
                if fm and fm.group(1) != "nan":
                    vals[field] = float(fm.group(1))
            k = it // BUCKET
            b = buckets.setdefault(k, {"steps": steps, "n": {}, "sum": {}})
            b["steps"] = max(b["steps"], steps)
            for field, v in vals.items():
                b["sum"][field] = b["sum"].get(field, 0.0) + v
                b["n"][field] = b["n"].get(field, 0) + 1
    for k in sorted(buckets):
        b = buckets[k]
        cells = []
        for field in FIELDS:
            n = b["n"].get(field, 0)
            cells.append(f"{b['sum'][field]/n:.4f}" if n else "")
        print(f"{log}\t{k*BUCKET}\t{b['steps']}\t" + "\t".join(cells))
