"""Print a compact summary of round_002 A/B/C results JSON (analysis helper)."""
import json
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_round002_c1_results.json")
if not os.path.exists(OUT):
    print("no results file yet:", OUT)
    sys.exit(0)

d = json.load(open(OUT))
print("== keys ==")
for k in sorted(d):
    v = d[k]
    print(f"  {k}: err={v['err']:.3e} dim={v['dim']} E={v['E']:.9f} "
          f"wall={v['wall_s']}s n_drawn_tot={v.get('n_drawn_total', '-')}")
print()
print("== per-round n_tgt/n_drawn (variant C cases) ==")
for k in sorted(d):
    v = d[k]
    if v.get("n_tgt_rounds"):
        print(f"  {k}: n_tgt={v['n_tgt_rounds']} n_drawn={v['n_drawn_rounds']}")
