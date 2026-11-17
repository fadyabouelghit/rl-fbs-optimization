# RL architectures for FBS on/off, placement and power

**Date:** 2026-11-08
**Repo:** `rl-fbs-pyqd`
**Scope:** every agent architecture built, considered or proposed so far, in one place. Results are for code 2-1-1 (2 FBSs, 1 MBS), fair2 protocol, unless stated.

Notation: b = on/off decision per FBS. c = continuous genes per FBS (x, y, height, power). s = observation. N = number of FBSs.

---

## 1. Shared environment (all agents)

All agents run on `ppo/env.py` (`FlyingBaseStationEnv`) with the `pyqd-fast` backend (QuaDRiGa-equivalent SINR, 1000 fixed users).

| | per FBS | 2-1-1 total |
|---|---|---|
| Observation | x, y, height, power, on/off (scaled to [-1, 1]) | 10 + 2 connection deltas + 4 tier fractions = 16 |
| Action | 4 continuous moves + 1 on/off value, all in [-1, 1] | 10 |

- **Continuous genes:** each moves by `a * 0.05 * range` per step and is clipped to bounds.
- **On/off gene (sticky rule):** it changes only when |a| >= 0.5; then on = (a >= 0).
- **Objective Φ:** 0.6 x connected-user share + 0.4 x share served by FBSs.
- **Reward:** change in Φ plus a bonus for beating the episode's best Φ (`potential_record`).
- **Episode:** 40 steps. **Training budget:** 100k steps.

Pinned for every algorithm by the fair protocol (`tuning/protocol.py`): learning rate 3e-4, γ 0.95, batch 256, hidden layers (64, 64). Each algorithm tunes exactly five knobs.

---

## 2. Baseline agents (Stable-Baselines3, unchanged)

### 2.1 PPO (`ppo/`)

```
s ──► actor MLP 64-64 (tanh) ──► mean μ(s) + learned spread ──► sample a ~ N(μ, σ²)
s ──► critic MLP 64-64 (tanh) ──► V(s)
```

- **Learning:** on-policy, from its latest batch of steps (rollout buffer).
- **Loss:** clipped surrogate + value loss, GAE.
- **On/off:** a random sample, so an FBS can always flip.

### 2.2 TD3 (`td3/`)

```
s ──► actor μ(s) 64-64 (ReLU, tanh out) ──► a = μ(s) + noise
[s, a] ──► critic Q1 64-64,  critic Q2 64-64      target = r + γ·min(Q1', Q2')
```

- **Learning:** off-policy, from a 1M-step replay memory.
- **TD3's three additions to DDPG:** two critics, delayed actor updates, smoothing noise on the target action.
- **Exploration:** Gaussian noise, with a separate size for the on/off dims (`binary_noise_std`).

### 2.3 DDPG (`ddpg/`)

Like TD3 but with one critic, no update delay and no target smoothing. Exploration is Gaussian or Ornstein-Uhlenbeck noise.

**Known weakness of TD3/DDPG:** the actor's tanh output on the on/off dim saturates near -1. Under the sticky rule a flip back to on then needs noise above 1.5, so a station can stay off for good (the stuck-off trap, first letter).

---

## 3. Hybrid actors: stochastic head on the on/off dim (`ddpg_hybrid/`, `td3_hybrid/`)

Same network as TD3/DDPG. Only the on/off output is read differently.

```
s ──► shared MLP 64-64 ──► z (pre-activation)
         continuous dims: a = tanh(z)
         on/off dim:      z = logit, P(on) = sigmoid(z)
                          training: Gumbel-sigmoid sample (straight-through), a in {-1, +1}
                          evaluation: on if z > 0
```

Two ways to train the on/off logit:

| Variant | Gradient on the on/off logit | Result (Run B, untuned) |
|---|---|---|
| `*_hybrid` | slope of the critic dQ/da at the sample | traps unchanged (11/40) |
| `*_hybrid_qd` | sigmoid(z) x [Q(on) − Q(off)], gap held fixed | DDPG: traps 0/40 (p = 0.0002) |

**Takeaway:** the update rule matters, not randomness. Comparing on vs off removes the trap, but stations are still left unused (both on in only 1 of 20 runs).

---

## 4. Switch + pilot: Double DQN + TD3/DDPG (`ddqn_td3/`), built and tuned

```
s ──► SWITCH: Double DQN, Q(s, b), b in {0,1}^N (2^N outputs) ──► b
[s, b] ──► PILOT: TD3 or DDPG actor μ(s, b) ──► c
env receives [c, b] ──► one shared reward r to both
```

- **Switch:** chooses on/off for all FBSs jointly by comparing Q-values (ε-greedy in training, greedy in evaluation). It is a Double DQN: the live net picks the next action, the target net scores it.
  - Settings: ε 1.0 → 0.05, target copy every 1000 steps, buffer 1M.
  - `exploration_fraction` is tuned (it takes `binary_noise_std`'s slot).
  - `hold_k`: decide every K steps (K = 1 in the tuned campaign).
- **Pilot:** an unchanged SB3 TD3/DDPG that sees the switch's decision as two extra inputs and outputs only the 4N continuous genes.
- **Memories:** each agent keeps its own. The switch trains on the same normalized reward as the pilot.
- **Limit:** N <= 3 (joint outputs grow as 2^N). Beyond that it needs one head per FBS.
- **Algorithm names:** `ddqn_td3`, `ddqn_ddpg`.

**Results** (fair2ddqn, 8 finalists x 35 seeds = 280 runs each):

| Algo | Winner J (35 seeds) | Both stations on | Failures | Gap to optimum closed* |
|---|---|---|---|---|
| PPO | 0.5282 | 233/280 | 0 | 85.3% |
| **DDQN+TD3** | **0.4944** | **134/280** | **0** | **59.5%** |
| **DDQN+DDPG** | **0.4729** | **86/280** | **7** | **43.1%** |
| TD3 | 0.4712 | 44/280 | 7 | 41.8% |
| DDPG | 0.4544 | 29/280 | 29 | 29.0% |

\*Best layout found by differential evolution, J 0.5474 (`headroom`): one FBS near (1442, 182), one near (320, 415), about 22 m high, full power.

**What we learned:**
- The switch roughly triples two-station use and removes switched-off runs.
- Holding the decision for K = 4 or 8 steps lengthens on-periods but changes neither station use nor J (`sp_hold`).
- PPO still wins mainly on placement: its two stations land on the optimum's two spots; the switch agents find the first spot but scatter the second.

---

## 5. Two-stage designs considered (before building section 4)

| # | Design | How on/off is chosen | Fixes the trap? | Cost | Status |
|---|---|---|---|---|---|
| 1 | **Two agents** (DQN switch + TD3 pilot) | DQN compares Q-values | yes | medium | built (section 4) |
| 2 | **One actor, two heads in sequence** (gate head, then mover head given b) | gate trained via critic | partly | low | not built |
| 3 | **Try both, keep the better** (critic scores c with b = off and b = on, pick max; P-DQN style) | critic comparison, no network outputs b | yes | medium | built as `on_off_td3/`, tuned in `fair2tb` (results below 5.1) |
| 4 | **Two substeps per env step** (set b, then set c) | same tanh actor | no, unless combined with 1 or 3 | lowest | not built |

### 5.1 Design 3 in detail: implementation spec ("try both")

**Added 2026-11-09.** Proposed names: package `tryboth/`, algorithms `tryboth_td3` and `tryboth_ddpg`.

#### Idea

```
s ──► actor μ(s) ──► c                      (continuous only, 4N values)
for every on/off pattern b in {-1,+1}^N:
      critic scores Q(s, c, b)              (2^N scores, one batched call)
b* = pattern with the highest score         (random pattern with prob. ε during training)
env receives [c, b*]
```

- **One agent, one memory.** No network outputs b.
- **The critic** is an ordinary TD3/DDPG critic on the full env action (5N values), so the replay memory stores the executed action unchanged.
- **The actor** outputs only the 4N continuous values.

#### Sizes (2 FBSs)

| Part | Input | Output | Hidden |
|---|---|---|---|
| Actor μ(s) | obs, 16 | c, 8 (tanh) | 64, 64 |
| Critic Q (TD3: twin pair; DDPG: one) | obs 16 + c 8 + b 2 = 26 | 1 | 64, 64 |

The pinned protocol values are unchanged: lr 3e-4, γ 0.95, batch 256, net (64, 64), 100k steps.

#### Acting

1. **Warm-up** (first `learning_starts` steps): c and b uniformly random. b is snapped to ±1, as in `ddpg_hybrid`.
2. **After warm-up:**
   - c = μ(s) + Gaussian (or OU) noise on the continuous dims only.
   - With probability ε, b is a random pattern. Otherwise b = argmax over all 2^N patterns of the critic score.
   - TD3 scores a pattern by min(Q1, Q2); DDPG uses Q.
3. **ε schedule:** linear from 1.0 to 0.05 over `exploration_fraction` x total steps, the same as the DQN switch.
4. **Evaluation and deployment:** ε = 0, no noise, argmax only.
5. **b is always sent as exactly ±1,** so the sticky rule always applies it.

#### Learning (per gradient step, batch of 256)

1. **Next continuous move:** c' = μ'(s') + clipped smoothing noise on the continuous dims (TD3 only; DDPG uses no noise).
2. **Next on/off:** b' = argmax over patterns of the online critic Q1(s', c', b'). This is the Double DQN style: the online critic picks, the target critic scores.
3. **Target:** y = r + γ (1 − done) x min(Q1'(s', c', b'), Q2'(s', c', b')). DDPG: Q'(s', c', b').
4. **Critic update:** standard MSE toward y on the stored (s, a), where a = [c, b] as executed.
5. **Actor update** (every `policy_delay` steps for TD3):
   - Pick b* = argmax over patterns of Q1(s, μ(s), b), with no gradient through this choice.
   - Loss = −mean Q1(s, μ(s), b*). Gradient flows into c only.
6. **Target networks:** Polyak update with τ, unchanged.

The enumeration expands the batch to 256 x 2^N rows (1,024 for 2 FBSs), which is cheap.

#### Code plan

Pin to SB3 2.9.0 with a version check, as `ddpg_hybrid` does.

| File | What |
|---|---|
| `tryboth/config.py` | `TryBothParams`: eps_start 1.0, eps_final 0.05, exploration_fraction (tuned), double_select True. Reuse `ddqn_td3.config.Layout` for index maps; N <= 3 |
| `tryboth/policy.py` | `TryBothActor`: output size 4N. `TryBothPolicy._predict`: actor then argmax over the critic, returns the full 5N action |
| `tryboth/algo.py` | `TryBothMixin`: overrides `_sample_action` (noise on c, ε-greedy on b) and `train()` (SB3 2.9.0 `TD3.train` with the target and actor-loss changes above). Classes `TryBothTD3(TryBothMixin, TD3)` and `TryBothDDPG(TryBothMixin, DDPG)` |
| `tryboth/train.py` | Calls `td3.train.train` / `ddpg.train.train` through the existing `model_factory` hook |
| `tryboth/callbacks.py` | Logs the chosen b per step and the per-station Q gap (best score with station j on minus best with it off). Writes `steps_compact.csv.gz` in the format `letterexp.pins` reads, with positions on |
| `tuning/` | Register `TRYBOTH_ALGOS`; spaces, worker, objective loader, study routing, CLI, as for the hybrids |
| `letterexp/` | Map the new algorithms to their parent package in `runner.score_one` and `launch.py` |

**Saving:** `model.zip` already holds actor and critic, so no extra file is needed. `best_model.zip` alone is a complete policy.

#### Five tuned knobs (same budget rule)

| Algo | Searched |
|---|---|
| `tryboth_td3` | action_noise_std · policy_delay · target_policy_noise · tau · exploration_fraction [0.05, 0.5] log |
| `tryboth_ddpg` | action_noise_std · noise_type · ou_dt · tau · exploration_fraction [0.05, 0.5] log |

**Pinned:** eps_final 0.05, double_select True.

#### Tests (`tests/test_tryboth.py`, analytic backend)

1. With a hand-set critic, the argmax returns the best of the 2^N patterns, for N = 1 and 2.
2. The actor's output size is 4N; on/off dims never come from the actor.
3. On a tiny batch, the target matches a hand computation (TD3 and DDPG).
4. The actor gradient does not flow through the choice of b.
5. A fixed seed reproduces final weights bit for bit.
6. `predict` on a plain env returns a 5N action with b in {−1, +1}^N; deterministic predict is repeatable.
7. Scoring through `tuning/objective.score_run` returns the same J on two loads.
8. The existing suite still passes.

#### Experiments

1. **Smoke run:** 10k steps on pyqd-fast, code 2-1-1, both pilots. Check the Q gap log and that both stations get switched on during training.
2. **Tuned campaign `fair2tb`:** full fair2 pipeline, same protocol as `fair2ddqn` (60 trials, top 8 x seeds 0-4, then 0-34).
3. **Comparison** against PPO, DDQN+TD3/DDPG and TD3/DDPG:
   - station use (both on), failures, J, gap closed to the 0.5474 layout
   - position maps (`lab/comparison/positions_rollout.py`)

#### Predictions (write down before results)

- **TB1:** both-on share >= DDQN+TD3 (134/280). Reason: b is judged on the exact moves about to be used.
- **TB2:** mean J >= DDQN+TD3 (0.4884 pooled).
- **TB3:** zero switched-off runs, as with the switch.

#### Risks

- **Everything hinges on the critic's ranking of on vs off.** Early mistakes are corrected only through ε-exploration.
- **Taking a max over b can overestimate.** This is mitigated by the twin minimum and the Double-style selection.
- **The actor learns c only under the currently best b,** so a pattern that rarely wins gets weak placement. If that shows up, test the P-DQN variant (separate c per pattern) or average the actor loss over all patterns.

#### Outcome (fair2tb, 2026-11-09)

Built as `on_off_td3/` (commit 64cc9a7), with two changes agreed before launch: the actor is trained under "all on", and selection uses the min of the online twin critics. Report: `lab/comparison/RESULTS_letter_2fbs_TB_tuned.txt`.

| Pooled, 280 runs | Bare | DQN switch | on_off |
|---|---|---|---|
| TD3 pilot: mean J / both on / failures | 0.4660 / 44 / 7 | **0.4884** / 134 / 0 | 0.4817 / **193** / 1 |
| DDPG pilot: mean J / both on / failures | 0.4443 / 29 / 29 | **0.4665** / **86** / **7** | 0.4428 / 79 / 40 |

- **Predictions:** TB1 (both on >= 134) supported, TB2 (J >= 0.4884) not supported, TB3 (no switched-off runs) supported.
- **Station use:** highest of the off-policy agents for the TD3 pilot.
- **Placement:** worse than the DQN switch. Both FBSs end within 150 m of the best layout in 1.6% of episodes (switch 23%, PPO 79%).
- **Likely cause:** placement is learned under the "all on" assumption, not the chosen pattern.

---

## 6. Proposed next architectures (target: placement of the second FBS)

### 6.1 Coarse-to-fine: the switch also picks a region

```
s ──► SWITCH (DQN): on/off + grid cell (e.g. 12 cells of 500 m) per FBS
[s, b, cell] ──► PILOT (TD3): fine position inside the cell, height, power
```

- **Why:** placement likely has several separate good spots. A gradient pilot climbs to the nearest one; choosing the cell by comparing Q-values searches the whole map.
- **Note:** needs one switch head per FBS (6.3), since joint options grow fast.

### 6.2 Evolution-guided pilot (GA + TD3 / CEM-RL)

```
GA population of pilot actors (e.g. 10) ──► episodes ──► shared replay memory ◄── TD3 actor + critic
best GA member and TD3 actor exchanged every few generations
```

- **GA members** do not learn by gradient. They are scored, selected and mutated, so they improve across generations.
- **What is evolved:** only the pilot's weights (about 5,900 for 2 FBSs). The critic, the switch and the hyperparameters stay fixed.
- **Mutation:** Gaussian noise on about 10% of the weights. Crossover swaps whole neurons.
- **CEM-RL variant:** half the population also gets gradient steps from the shared critic, so every member learns. This variant suits a fixed budget better.
- **Why:** broad search from the GA plus precise learning from TD3. It ties to the NSGA-II thesis work.
- **Cost:** more simulator steps per generation; the budget must be reported for fairness.

### 6.3 Shared network per FBS (decentralized actors, one central critic)

```
for each FBS i:  [own state_i, summary of other FBSs] ──► SAME actor ──► c_i
                 same input ──► SAME switch head ──► on/off_i
one central critic sees all FBS states and actions (shared reward)
```

- **Why:**
  - What is learned for FBS 1 transfers to FBS 2.
  - Outputs grow linearly with N, not as 2^N.
  - Each FBS sees the others, so they can spread out.
- **Basis:** the MADDPG/MATD3 pattern, plus parameter sharing.
- **At N = 1:** it reduces exactly to section 4. That case serves as a check; the real test is scaling to N = 2, 3, 4, where the advantage should grow.

### 6.4 Cheap variants, as ablations

- **Warm-up favouring "on":** the switch's early exploration chooses "on" more often, then decays. This builds in Run C's finding that early on-time helps.
- **Stochastic pilot (SAC):** tests whether PPO's edge comes from random exploration. SAC is already available in `ppo/`.

### 6.5 Hybrid A: DQN switch + PPO pilot (cheap first test)

**Added 2026-11-09.**

```
s ──► SWITCH: Double DQN Q(s, b) (unchanged, ddqn_td3/switch.py) ──► b
[s, b] ──► PILOT: PPO actor π(c | s, b), Gaussian, 64-64 tanh ──► c (sampled in training, mean at evaluation)
env receives [c, b];  shared reward
```

- **Question it answers:** does a value-based on/off decision help PPO, or is PPO's own sampled on/off already enough?
- **Target to beat:** PPO alone has both stations on in 233/280 runs (41 one-on runs remain), J 0.5282, 85% of the best layout. Success means both-on near 280/280 and J above 0.528.
- **Learning:**
  - The switch learns exactly as in `ddqn_td3` (Double DQN, own replay memory, ε-greedy, `hold_k` 1).
  - PPO learns on-policy from its own rollouts, with b as two extra inputs. From PPO's side, the switch is part of the environment.
- **Code plan:**
  - `ddqn_td3/wrapper.SwitchVecEnv` already does all the switch work inside `reset`/`step_wait` and does not depend on the pilot algorithm.
  - `ppo/train.py` has no wrapper hook, and `ppo/` must not be modified. So a new package `ddqn_ppo/` with its own `train.py` is needed: copy `ppo/train.py`'s setup (run logger, Monitor, VecNormalize order, `EnvInfoLoggingCallback`, `CsvEvalCallback`), insert `SwitchVecEnv`, and register the existing switch callbacks.
  - Evaluation and scoring reuse `ddqn_td3.policy.SwitchPilotPolicy` with a PPO loader branch.
  - The switch needs `n_envs == 1`, which matches the protocol.
- **Tuning (five knobs):** PPO's own four of `ent_coef`, `n_steps`, `n_epochs`, `gae_lambda`, `clip_range` (drop one, for example `clip_range` pinned at 0.2), plus `exploration_fraction`. State the dropped knob openly.
- **Pilot experiment before tuning:**
  - Run PPO winner t35's settings plus the default switch, seeds 0-19, against PPO t35's own seeds 0-19.
  - Measures: both-on, J, failures, positions.
  - Cost: about 20 runs, roughly 30 min at 8 workers.
- **Risks:**
  - PPO already switches well, so the gain may be small.
  - The switch learns from PPO's noisy placements early, which may bias it toward "off" until PPO improves.

### 6.6 Hybrid B: twin-critic on/off selection + PPO placement (full design)

**Added 2026-11-09.**

```
for each on/off pattern b in {-1,+1}^N:
      c_b = mean of PPO actor π(c | s, b)                  (PPO's best placement for that pattern)
      score_b = min(Q1, Q2)(s, c_b, b)                    (TD3-style twin critics)
b* = argmax score_b    (random pattern with prob. ε in training)
c  ~ π(c | s, b*)      (sample in training, mean at evaluation)
env receives [c, b*]
```

- **Why:**
  - **On/off is decided by value comparison**, as in on_off_td3, so there is no trap.
  - **Placement is learned for the chosen pattern.** In on_off_td3 the actor assumed "all on"; here it is conditioned on b, which targets on_off_td3's placement weakness.
  - **Placement uses PPO**, the best placer so far.
- **Networks (2 FBSs):**
  - PPO actor and value net: input obs 16 + b 2 = 18; actor output 8 continuous values; (64, 64), tanh.
  - Twin critics: input obs 16 + c 8 + b 2 = 26; (64, 64), ReLU; target copies with Polyak τ.
- **Learning:**
  1. **PPO:** standard clipped update on its rollouts (`n_steps`), with b* in its input. Advantages come from PPO's own value net and GAE, not from the Q critics.
  2. **Twin critics:** off-policy from a replay memory filled by the same rollouts (s, c, b*, r, s′).
     - Target: y = r + γ(1 − done) · min(Q1′, Q2′)(s′, c′, b′).
     - b′ = argmax over patterns of the online critics' min, with each pattern's c′ = PPO mean at (s′, b′).
     - One critic step per env step after warm-up.
  3. **ε for b:** linear 1.0 → 0.05 over `exploration_fraction`.
- **Why the two learners don't clash:** PPO never uses the Q critics for its gradient. The critics never update the actor. They meet only at action selection.
- **Cost:**
  - Per step: 2^N actor passes plus 2^N critic passes (4 each for 2 FBSs), batched.
  - Two update schedules: PPO every `n_steps`, critics every step.
  - Effort: medium-high, a custom SB3 PPO subclass that overrides `collect_rollouts` to insert the selection step.
- **Code plan:**
  - New package `ppo_onoff/`, pinned to SB3 2.9.0 with a version check.
  - `PPOOnOff(PPO)` overrides `collect_rollouts` (selection + store b* in obs) and adds a critic update hook.
  - The critic module is reused from `on_off_td3/`.
  - The saved model must hold actor, value net and both critics; deployment needs actor + critics.
- **Tuning (five knobs):** `ent_coef`, `n_steps`, `gae_lambda`, `exploration_fraction`, critic `tau`. Pin `n_epochs` and `clip_range` at the PPO winner's values and state it.
- **Tests:**
  1. With a hand-set critic, the argmax picks the right pattern.
  2. PPO's input contains b*, and the stored rollout action is c only.
  3. Critic target matches a hand computation.
  4. PPO's loss has no gradient path into the critics, and the critics' loss has none into the actor.
  5. A fixed seed reproduces weights bit for bit.
  6. Scoring returns the same J on two loads.
  7. The existing suite still passes.
- **Order:** build only if Hybrid A shows that a separate value-based on/off decision helps PPO. If A shows no gain, PPO's sampled on/off is already sufficient. The letter's message then becomes "off-policy agents need a value-based switch; PPO does not".

---

## 7. Where to find things

| Item | Path |
|---|---|
| Environment, reward | `ppo/env.py`, `ppo/config.py` |
| PPO / TD3 / DDPG training | `ppo/train.py`, `td3/train.py`, `ddpg/train.py` |
| Hybrid heads | `ddpg_hybrid/hybrid.py`, `td3_hybrid/train.py` |
| Switch + pilot | `ddqn_td3/` (brief: `prompt_switchpilot.md`) |
| Fair protocol, search spaces | `tuning/protocol.py`, `tuning/spaces.py` |
| Experiment history and results | `lab/EXPERIMENTS.md`, `lab/NOTES.md`, `lab/comparison/RESULTS_*.txt` |
| Figures | `lab/comparison/figures_letter/` (F_ddqn_*, F_onperiod_vs_steps, F_positions_*) |
