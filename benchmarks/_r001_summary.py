"""Summarize _round001_c1_results.json compactly."""
import json, os, sys

p = "/mnt/d/tc_sqd/benchmarks/_round001_c1_results.json"
if not os.path.exists(p):
    print("(no results yet)")
    sys.exit(0)
d = json.load(open(p))
if not d:
    print("(empty)")
    sys.exit(0)
for k in sorted(d):
    v = d[k]
    print(f"{k}: E={v['E']:.8f} err={v['err']:.3e} dim={v['dim']} "
          f"wall={v['wall_s']:.0f}s peak_rss={v.get('peak_rss_kb', 'n/a')}kB")
    for r in v["per_round"]:
        print(f"    r{r['round']:>2} dim={r['dim']:>7} n_str={r['n_str']:>4} "
              f"n_new={r['n_new_proxy']:>4} E={r['E']:.6f}")
