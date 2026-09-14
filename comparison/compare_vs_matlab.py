"""Compare a pyqd-backend run against the archived MATLAB potrec10 run.

The pyqd run reproduces run_2026-08-19_23-46-16_1-1-1_1fbs1mbs_potrec10 exactly
except for (a) total_timesteps 25000 -> 12500 (the true halfway point: both
round up to 13 rollouts of n_steps=1024 = 13312 executed steps) and (b) the
physics backend. Everything else -- seed, reward, shaping, optimizer -- is identical,
so any divergence is attributable to the backend swap plus library-version drift.

Usage: python comparison/compare_vs_matlab.py <pyqd_run_dir>
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

OLD = Path('/Users/fadya/Documents/MATLAB/GA_github/genetic-algorithm-optimization'
           '/genetic-algorithm-optimization')
ML = OLD / 'ppo_runs/run_2026-08-19_23-46-16_1-1-1_1fbs1mbs_potrec10'
PY = Path(sys.argv[1] if len(sys.argv) > 1 else
          sorted(Path('ppo_runs').glob('*_1fbs1mbs_potrec10_half_fast'))[-1])
N = 13312  # the shared executed-step budget

def load(d):
    return (pd.read_csv(d/'steps.csv'), pd.read_csv(d/'eval_log.csv'),
            pd.read_csv(d/'progress.csv'))

ms, me, mp = load(ML)
ps, pe, pp = load(PY)
ms = ms[ms.timesteps <= N]

print("="*78)
print(f"MATLAB : {ML.name}")
print(f"pyqd   : {PY.name}")
print(f"window : {N} executed steps (MATLAB run continued to {len(pd.read_csv(ML/'steps.csv'))})")
print("="*78)

print("\n### 1. Seeded deterministic eval curve -- the headline learning signal\n")
m = me[me.timesteps <= N].set_index('timesteps')
p = pe.set_index('timesteps')
cols = ['mean_objective','mean_best_objective','mean_total_connected','mean_fbs_connected','mean_reward']
rows = []
for t in p.index:
    if t not in m.index: continue
    r = {'steps': t}
    for c in cols:
        r[f'{c}_ml'] = m.loc[t, c]; r[f'{c}_py'] = p.loc[t, c]
        r[f'{c}_d']  = p.loc[t, c] - m.loc[t, c]
    rows.append(r)
e = pd.DataFrame(rows)
for c in cols:
    print(f"  {c}")
    for _, r in e.iterrows():
        print(f"     {int(r['steps']):6d}   MATLAB {r[c+'_ml']:10.6f}   pyqd {r[c+'_py']:10.6f}   diff {r[c+'_d']:+.2e}")
    print(f"     -> max |diff| = {e[c+'_d'].abs().max():.3e}\n")

print("### 2. Per-step training trajectory (all 13312 steps)\n")
for c in ['reward_objective','total_connected','fbs_connected','total_power','reward']:
    if c not in ms or c not in ps: continue
    a, b = ms[c].values, ps[c].values[:len(ms)]
    d = b - a
    exact = int((d == 0).sum())
    print(f"  {c:18s} identical {exact:5d}/{len(a)} ({100*exact/len(a):5.1f}%)   "
          f"max|d|={np.abs(d).max():.3e}   mean|d|={np.abs(d).mean():.3e}")

print("\n### 3. FBS state trajectory (did the agent fly the same path?)\n")
for c in ['fbs0_x','fbs0_y','fbs0_height','fbs0_power','fbs0_power_status']:
    if c not in ms or c not in ps: continue
    a, b = ms[c].values, ps[c].values[:len(ms)]
    d = np.abs(b - a)
    print(f"  {c:20s} identical {int((d==0).sum()):5d}/{len(a)}   max|d|={d.max():.3e}")

print("\n### 4. Optimizer diagnostics\n")
key = 'time/total_timesteps'
mm = mp[mp[key] <= N].set_index(key); ppx = pp.set_index(key)
for c in ['train/explained_variance','train/approx_kl','train/value_loss','train/entropy_loss']:
    if c not in mm or c not in ppx: continue
    j = mm[[c]].join(ppx[[c]], lsuffix='_ml', rsuffix='_py').dropna()
    if len(j):
        d = (j[c+'_py'] - j[c+'_ml']).abs()
        print(f"  {c:28s} max|diff| over {len(j)} updates = {d.max():.3e}")

print("\n### 5. Wall clock\n")
mprov = json.loads((ML/'experiment_config.json').read_text())['provenance']
pprov = json.loads((PY/'experiment_config.json').read_text())['provenance']
import re
def secs(d):
    t = re.search(r'training done in ([\d.]+)s', (d/'run.log').read_text())
    return float(t.group(1)) if t else float('nan')
mt_full, pt = secs(ML), secs(PY)
mt = mt_full * N / 25600           # MATLAB's cost for the SAME 13312 steps
print(f"  MATLAB  25600 steps : {mt_full:10.1f} s  ({mt_full/3600:.2f} h)  "
      f"{25600/mt_full:.2f} steps/s")
print(f"  MATLAB  {N} steps (pro-rata): {mt:10.1f} s  ({mt/3600:.2f} h)")
print(f"  pyqd    {N} steps : {pt:10.1f} s  ({pt/3600:.4f} h)  {N/pt:.1f} steps/s")
print(f"  -> speedup on the matched budget: {mt/pt:.0f}x")
print(f"\n  MATLAB env setup : {mprov.get('env_setup_seconds')} s (MATLAB engine + QuaDRiGa)")
print(f"  pyqd   env setup : {pprov.get('env_setup_seconds')} s")
print(f"  backend recorded : {pprov.get('backend')} | pyqd_channel {pprov.get('versions',{}).get('pyqd_channel')}")
print(f"  torch  {mprov['versions']['torch']} -> {pprov['versions']['torch']} | "
      f"sb3 {mprov['versions']['stable_baselines3']} -> {pprov['versions']['stable_baselines3']}")
