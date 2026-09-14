"""Direct comparison of two run dirs that differ only in physics backend."""
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd

A, B = Path(sys.argv[1]), Path(sys.argv[2])
def prov(d): return json.loads((d/'experiment_config.json').read_text())['provenance']
def secs(d):
    m = re.search(r'training done in ([\d.]+)s', (d/'run.log').read_text())
    return float(m.group(1)) if m else float('nan')

print("="*80)
print(f"A: {A.name}  [{prov(A)['backend']}]")
print(f"B: {B.name}  [{prov(B)['backend']}]")
print("="*80)

sa, sb = pd.read_csv(A/'steps.csv'), pd.read_csv(B/'steps.csv')
n = min(len(sa), len(sb))
print(f"\n### per-step trajectory ({n} steps)\n")
cols = ['reward','reward_objective','total_connected','fbs_connected','mbs_connected',
        'total_power','avg_rate','sum_rate','fbs0_x','fbs0_y','fbs0_height',
        'fbs0_power','fbs0_power_status']
worst = 0.0
for c in cols:
    if c not in sa or c not in sb: continue
    d = sa[c].values[:n] - sb[c].values[:n]
    worst = max(worst, float(np.nanmax(np.abs(d))))
    print(f"  {c:18s} identical {int((d==0).sum()):6d}/{n}   max|d|={np.nanmax(np.abs(d)):.3e}")
print(f"\n  worst deviation over every column: {worst:.3e}")

ea, eb = pd.read_csv(A/'eval_log.csv'), pd.read_csv(B/'eval_log.csv')
j = ea.set_index('timesteps').join(eb.set_index('timesteps'), lsuffix='_a', rsuffix='_b', how='inner')
print(f"\n### eval curve ({len(j)} checkpoints)\n")
for c in ['mean_objective','mean_fbs_connected','mean_total_connected']:
    if c+'_a' in j: print(f"  {c:22s} max|d| = {(j[c+'_a']-j[c+'_b']).abs().max():.3e}")

ta, tb = secs(A), secs(B)
print(f"\n### wall clock\n")
print(f"  A ({prov(A)['backend']:10s}) {ta:9.1f} s  ({ta/3600:.3f} h)  {n/ta:7.2f} steps/s")
print(f"  B ({prov(B)['backend']:10s}) {tb:9.1f} s  ({tb/3600:.3f} h)  {n/tb:7.2f} steps/s")
print(f"  A/B ratio: {ta/tb:.1f}x")
print(f"\n  MATLAB reference (same 25600 steps): 24463.1 s (6.80 h), 1.05 steps/s")
print(f"    vs A: {24463.1/ta:.2f}x    vs B: {24463.1/tb:.1f}x")
