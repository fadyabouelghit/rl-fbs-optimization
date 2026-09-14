"""Compare our post-hoc evaluations against the archived MATLAB evaluation."""
from pathlib import Path
import numpy as np, pandas as pd

OLD = Path('/Users/fadya/Documents/MATLAB/GA_github/genetic-algorithm-optimization'
           '/genetic-algorithm-optimization')
ml = sorted((OLD/'ppo_runs/run_2026-08-19_23-46-16_1-1-1_1fbs1mbs_potrec10/evals').glob('*/summary.csv'))[-1]
M = pd.read_csv(ml)
print("="*84); print(f"MATLAB reference eval: {ml.parent.name}  (5 episodes, seed 1000, deterministic)"); print("="*84)
COLS = ['reward_sum','final_total_connected','final_fbs_connected','final_mbs_connected',
        'final_total_power','final_avg_rate','final_sum_rate','best_total_connected']
print(M[['episode']+COLS].to_string(index=False))

for tag, pat in [('full_seed0_faithful','*_potrec10_seed0_faithful'),
                 ('full_seed0_fast','*_potrec10_seed0')]:
    ds = [d for d in Path('ppo_runs').glob(pat)
          if ('faithful' in d.name) == ('faithful' in tag)]
    if not ds: print(f"\n[{tag}] no run found"); continue
    ev = sorted((ds[0]/'evals').glob('*/summary.csv'))
    if not ev: print(f"\n[{tag}] no evaluation found"); continue
    P = pd.read_csv(ev[-1])
    print(f"\n{'='*84}\n{tag}: {ds[0].name} / {ev[-1].parent.name}\n{'='*84}")
    n = min(len(M), len(P)); worst = 0.0
    for c in COLS:
        if c not in P: continue
        d = P[c].values[:n] - M[c].values[:n]
        worst = max(worst, float(np.max(np.abs(d))))
        print(f"  {c:24s} identical {int((d==0).sum())}/{n}   max|d|={np.max(np.abs(d)):.3e}")
    print(f"  -> worst deviation vs MATLAB across all eval columns: {worst:.3e}")
