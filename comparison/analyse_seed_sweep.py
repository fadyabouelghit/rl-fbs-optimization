"""Stress-test analysis of the potrec10 x 10-seed sweep on the pyqd backend.

Seed 0 reproduces the archived MATLAB run's seed, so it doubles as a full-length
regression check. Seeds 1-9 drive the FBS through different regions of state
space -- the actual robustness test of the ported physics.
"""
from pathlib import Path
import json, re, numpy as np, pandas as pd

OLD = Path('/Users/fadya/Documents/MATLAB/GA_github/genetic-algorithm-optimization'
           '/genetic-algorithm-optimization')
ML  = OLD/'ppo_runs/run_2026-08-19_23-46-16_1-1-1_1fbs1mbs_potrec10'
runs = {int(re.search(r'seed(\d+)$', p.name).group(1)): p
        for p in Path('ppo_runs').glob('*_potrec10_seed*') if p.is_dir()}
runs = dict(sorted(runs.items()))
print(f"analysing {len(runs)} seed runs\n")

print("="*86)
print("1. PHYSICS INTEGRITY  (every step of every seed)")
print("="*86)
bad = 0
for sd, d in runs.items():
    s = pd.read_csv(d/'steps.csv')
    tier = s.fbs_connected + s.mbs_coverage_connected + s.mbs_capacity_connected
    viol = int((tier != s.total_connected).sum())
    nan  = int(s[['reward_objective','total_connected','fbs_connected','total_power',
                  'avg_rate','sum_rate']].isna().sum().sum())
    inf  = int(np.isinf(s[['reward_objective','total_power','sum_rate']].values).sum())
    neg  = int((s.total_connected < 0).sum() + (s.fbs_connected < 0).sum())
    over = int((s.total_connected > 1000).sum())
    rng_ok = bool(s.fbs0_x.between(0,2000).all() and s.fbs0_y.between(0,1500).all()
                  and s.fbs0_height.between(20,150).all()
                  and s.fbs0_power.between(7,10.5).all())
    bad += viol+nan+inf+neg+over+(0 if rng_ok else 1)
    print(f"  seed {sd}: {len(s):5d} steps | tier-invariant viol {viol} | NaN {nan} | "
          f"inf {inf} | neg {neg} | >num_users {over} | genes in bounds {rng_ok}")
print(f"\n  => total integrity failures across {sum(len(pd.read_csv(d/'steps.csv')) for d in runs.values())} steps: {bad}")

print("\n"+"="*86)
print("2. SEED-0 REGRESSION vs MATLAB  (full 25600 steps, must be exact)")
print("="*86)
if 0 in runs:
    m = pd.read_csv(ML/'steps.csv'); p = pd.read_csv(runs[0]/'steps.csv')
    n = min(len(m), len(p)); print(f"  comparing {n} steps")
    for c in ['reward_objective','total_connected','fbs_connected','total_power',
              'fbs0_x','fbs0_y','fbs0_height','fbs0_power','fbs0_power_status']:
        d = p[c].values[:n]-m[c].values[:n]
        print(f"    {c:20s} identical {int((d==0).sum()):5d}/{n}  max|d|={np.abs(d).max():.3e}")
    me = pd.read_csv(ML/'eval_log.csv').set_index('timesteps')
    pe = pd.read_csv(runs[0]/'eval_log.csv').set_index('timesteps')
    j  = me.join(pe, lsuffix='_ml', rsuffix='_py', how='inner')
    print(f"    eval curve ({len(j)} checkpoints) max|diff| mean_objective = "
          f"{(j.mean_objective_py-j.mean_objective_ml).abs().max():.3e}")

print("\n"+"="*86)
print("3. CROSS-SEED OUTCOMES  (final eval, 25000 steps)")
print("="*86)
rows=[]
for sd, d in runs.items():
    e = pd.read_csv(d/'eval_log.csv'); s = pd.read_csv(d/'steps.csv')
    last = e.iloc[-1]
    t = re.search(r'training done in ([\d.]+)s', (d/'run.log').read_text())
    rows.append(dict(seed=sd, final_obj=last.mean_objective,
                     best_obj=e.mean_objective.max(),
                     fbs=last.mean_fbs_connected, conn=last.mean_total_connected,
                     peak_fbs=int(s.fbs_connected.max()),
                     sec=float(t.group(1)) if t else np.nan))
df=pd.DataFrame(rows)
print(df.to_string(index=False, float_format=lambda v:f"{v:9.4f}"))
print(f"\n  final objective : mean {df.final_obj.mean():.4f}  sd {df.final_obj.std(ddof=1):.4f}"
      f"  min {df.final_obj.min():.4f}  max {df.final_obj.max():.4f}")
print(f"  best  objective : mean {df.best_obj.mean():.4f}  sd {df.best_obj.std(ddof=1):.4f}")
print(f"  FBS-served      : mean {df.fbs.mean():.1f}  sd {df.fbs.std(ddof=1):.1f}"
      f"  range [{df.fbs.min():.1f}, {df.fbs.max():.1f}]")
print(f"  wall clock      : mean {df.sec.mean():.1f}s  sd {df.sec.std(ddof=1):.1f}s"
      f"  total {df.sec.sum():.0f}s")
print(f"\n  MATLAB seed-0 single run: final_obj 0.4733  fbs 90.7  (6.80 h)")
print(f"  do-nothing baseline J=0.4480 ; 1-FBS exhaustive optimum J=0.5220")
b,o=0.4480,0.5220
print(f"  headroom captured: mean {100*(df.final_obj.mean()-b)/(o-b):.1f}%  "
      f"range [{100*(df.final_obj.min()-b)/(o-b):.1f}%, {100*(df.final_obj.max()-b)/(o-b):.1f}%]")
df.to_csv('comparison/seed_sweep/summary.csv', index=False)
print("\n  -> comparison/seed_sweep/summary.csv")
