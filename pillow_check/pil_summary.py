import json, sys, math
import numpy as np
from scipy import stats
R = [json.loads(l) for l in open(sys.argv[1])]
arms = {}
for r in R:
    arms.setdefault(r['arm'], {})[r['seed']] = r['correct'] / 100
def ci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    if n < 2: return m, float('nan'), float('nan')
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n); return m, m - h, m + h
for a, d in arms.items():
    m, lo, hi = ci(list(d.values())); print(f"{a:9s} seeds {sorted(d)}: {[d[s] for s in sorted(d)]}  mean {m:.2f} [{lo:.2f}, {hi:.2f}]")
for a in ('pillow', 'zeropad'):
    if a in arms:
        common = sorted(set(arms[a]) & set(arms['notebook']))
        diff = [arms[a][s] - arms['notebook'][s] for s in common]
        m, lo, hi = ci(diff); print(f"{a} - notebook, paired over seeds {common}: {np.round(diff, 2).tolist()} -> {m:+.2f} [{lo:+.2f}, {hi:+.2f}]")
print(f"Privacy Blur's reported 97.58; T4 10-epoch notebook-blur result 94.89 ± 0.13 (pop SD)")
