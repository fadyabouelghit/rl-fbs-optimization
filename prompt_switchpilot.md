# Prompt: two-agent remedy, switch + pilot (switchpilot)

**For:** Claude Code, in the repo root (`/Users/fadya/Documents/MATLAB/GA_github/rl-fbs-pyqd`), using `./venv/bin/python`.
**Prepared:** 2026-11-03
**Read first:** `HANDOVER_letter.md` (rules, fairness) and `letter_runs_2fbs/HANDOVER_runs_chat.md` (current 2-FBS state).

## Why

Run B (`m1_hybrid2`) showed that the update rule removes the trap: `ddpg_hybrid_qd` has 0/40 traps. But it still used both stations in only 1 of 20 runs. Trap and abandonment are separable.

The new method splits the agent in two:

- a **switch**: a DQN that decides on/off for each FBS
- a **pilot**: a TD3 or DDPG agent that decides x, y, height and power, given the switch's decision

The switch chooses by comparing Q-values, so no on/off gradient exists at all. The switch can also hold its decision for K steps. "On" is then judged after the pilot has had K steps to make it pay off, which is the suspected cause of abandonment.

## The method

Each env step:

1. The switch holds b, the on/off decision for every FBS, chosen at the last decision point. It decides again every K steps: ε-greedy over Q_sw(s, b), with b ∈ {0,1}^N as one joint output (2^N actions).
2. The pilot sees [obs, b] and outputs c, the continuous genes (4 per FBS), plus its normal exploration noise.
3. The env receives the full action [c, b]. b is passed as ±1, so the sticky rule always applies it.
4. The pilot stores ([obs, b], c, r, [obs′, b′]) and does its usual TD3/DDPG update.
5. At each decision point the switch stores (s, b, R_K, s_K, done), where R_K = Σ γ^i r_i over the hold. It does one Double-DQN update per env step after warm-up, with target r + γ^K · Q_target(s_K, argmax Q(s_K, ·)).
6. Both agents learn from the same reward: the shaped reward after VecNormalize, i.e. exactly what the pilot sees.

Evaluation is deterministic: ε = 0 and no pilot noise.

## Rules

- **Protected paths:** do not modify `ppo/`. Do not write to `ppo_runs/`, `ddpg_runs/`, `td3_runs/`, `tuning_runs/` or `training_log.csv`.
- **Outputs:** 2-FBS runs go to `letter_runs_2fbs/`, so set `LETTER_RUNS_DIR=letter_runs_2fbs`. 1-FBS runs go to `letter_runs/`.
- **Changes to existing packages:** additions to `td3/`, `ddpg/`, `tuning/` and `letterexp/` must be opt-in, with unchanged defaults. Run the full suite after any edit to them.
- **Compute:** one 8-worker job at a time. Launch detached and poll with separate short commands.
- **Git:** ask Dr. Fady before any commit.
- **Reporting:** counts and exact numbers. No italics, no em dashes.
- **Stop at every CHECKPOINT.**

## Step 1: build (no long runs)

**New package `switchpilot/`.** It imports `ppo/`, `td3/` and `ddpg/` and modifies none of them.

| File | What |
|---|---|
| `config.py` | `SwitchParams`: `hold_k` (1), `eps_start` (1.0), `eps_final` (0.05), `exploration_fraction` (0.2), `target_update` (1000 steps, hard copy), `double_q` (True), `buffer_size` (1M), `learning_starts` (= pilot's). Shared values come from the pilot's params: lr, γ, batch, net_arch. |
| `switch.py` | Q-network: MLP on the env obs, `net_arch` hidden layers (pinned 64,64, ReLU), 2^N outputs. Replay buffer, Double-DQN update with Huber loss, ε schedule. Uses its own torch and numpy RNG streams, seeded from the run seed. |
| `wrapper.py` | `SwitchWrapper(gym.Wrapper)`: holds the switch and runs steps 1, 3 and 5 above inside `reset()`/`step()`. The pilot's action space is Box(4N) (continuous dims only); its obs space is obs + N dims (b as ±1). The wrapper puts `b`, `switch_eps` and `switch_decided` into `info`. |
| `policy.py` | `SwitchPilotPolicy.load(run_dir, which="best"\|"final")` with `predict(obs, deterministic)`, which returns the full env action [c, b] on a plain env. This is the only interface that `assess_models`, `tuning/objective.py` and `letterexp` scoring use. |
| `train.py` | `train(exp, backend, run_dir, *, pilot="td3"\|"ddpg", switch=SwitchParams(), ...)`. Calls `td3.train.train` or `ddpg.train.train` with the wrapper (see the hook below). |

**Hook in `td3/train.py` and `ddpg/train.py`.** Add a keyword-only `env_wrapper: Optional[Callable] = None`, applied to the train env and the eval env. The default None means no change.

**Saving:**

- `model.zip` + `switch.pt` at the end of training
- `best_model.zip` + `best_switch.pt`, written together whenever `CsvEvalCallback` saves a new best. The best pair must always come from the same eval. Write a test for this.
- the switch's full state goes into `experiment_config.json` extra (`"switchpilot": {...}`)

**Tuning registration.** Add `SWITCHPILOT_ALGOS = ("sp_td3", "sp_ddpg")` to `tuning/protocol.py`; `ALGOS` and `HYBRID_ALGOS` stay unchanged. Register the new names in the same places the hybrids are registered:

- `spaces.py`
- `objective._load` (scores via `SwitchPilotPolicy`)
- `worker.py`
- `study._RUNS_ENV`
- `cli.py`
- `letterexp/runner.py` (`score_one`)
- `letterexp/launch.py`

**Search space: five knobs, as for every other algorithm.** The `binary_noise_std` slot becomes `exploration_fraction`, since the switch's ε is the binary exploration.

| algo | searched |
|---|---|
| `sp_td3` | `action_noise_std` · `policy_delay` · `target_policy_noise` · `tau` · `exploration_fraction` [0.05, 0.5] log |
| `sp_ddpg` | `action_noise_std` · `noise_type` · `ou_dt` · `tau` · `exploration_fraction` [0.05, 0.5] log |

- **Pinned, not searched:** `eps_final` 0.05, `target_update` 1000 and `double_q`, plus the shared protocol values (lr 3e-4, γ 0.95, batch 256, net 64,64). List them in `python -m tuning spaces`.
- **`hold_k` is a design arm, not a knob.** It is fixed per campaign (see Step 3).

**Probe instrumentation.** Add `SwitchProbeCallback` → `switch_probe.csv`. On the existing probe set, every 2k steps, record per station j:

- `gap_j` = (max Q over joint actions with j on) − (max Q over joint actions with j off)
- the greedy choice for j
- ε

It must be bitwise neutral to training, like `ProbeCallback`.

**Tests**, in `tests/test_switchpilot.py`, analytic backend:

1. The wrapper sends [c, b] to the env with b ∈ {−1, +1}^N. The obs and action shapes are right for N = 1 and N = 2.
2. With `hold_k` = 4: b changes only at steps 0, 4, 8, ...; one switch transition is stored per hold; R_K and γ^K are correct against a hand computation; an episode end inside a hold stores a truncated transition.
3. On a toy where "on" always pays more, Q(on) > Q(off) on every probe state after a short run.
4. A fixed seed reproduces final weights bit for bit, for both the pilot and the switch.
5. `best_model.zip` and `best_switch.pt` come from the same eval step.
6. `SwitchPilotPolicy` scoring through `tuning/objective.score_run` returns the same J on two loads.
7. The probe callback is bitwise neutral.
8. With `env_wrapper=None`, the td3/ddpg hook leaves existing runs bit for bit unchanged.
9. The full suite still passes (currently 198 passed, 3 skipped).

**CHECKPOINT 1.** Report:

- `git diff --stat`, showing `ppo/` untouched
- the new files
- the test results
- one `python -m tuning spaces` printout
- a 10k pyqd-fast smoke run on 2-1-1 for each pilot, with its `switch_probe.csv` head

## Step 2: pilot (transfer test, not tuned)

**Goal:** choose `hold_k` and check the method works, before spending a full tuning campaign.

**Plan `sp_pilot` in `letterexp/launch.py`:**

- **Scenario:** code 2-1-1, fair2 protocol, instrumented (compact steps with positions, switch probe)
- **Pilot knobs:** taken from the fair2 parents, read from `letter_runs_2fbs/tuning/fair2/*_confirmation.json`:
  - `sp_td3` ← TD3 t51 and TD3 t47
  - `sp_ddpg` ← DDPG t15
- **Switch:** default `SwitchParams`
- **Arms:** `hold_k` ∈ {1, 4, 8}
- **Seeds:** 0 to 19
- **Total:** 3 parents × 3 K × 20 seeds = 180 runs, about 4.5 h at 8 workers. Measure the DQN overhead on seed 0 first and correct the estimate.

**Pre-register before any result exists** (`analyze prereg sp_pilot`, hash in the results file):

- **SP1 (trap):** stations trapped at the end. The trap rule is `pins.py` MIN_RUN = 5, applied to the executed on/off gene. Prediction: 0 per arm, and fewer than the parent arms of `e2_2fbs` at the tuned s_b.
- **SP2 (abandonment):** the Step A class of the best model on the 20 tuner states. Prediction: both_on is higher than the parent and higher than `ddpg_hybrid_qd` in Run B, and it rises with K.
- **SP3 (score):** mean J_best ≥ the parent's fair2 mean J at the same seeds.

**Report per arm** (out of 20):

- both_on / one_on / none_on / mixed
- the trap counts, with Wilson 95% intervals
- mean J (sd)
- on-share and on-period mean, as in `m1_top3`
- the switch's median `gap_j` at the end of training

**Tests:**

- Fisher, arm vs parent and arm vs `ddpg_hybrid_qd`, for both_on and for traps
- Welch for J
- Cochran-Armitage across K for both_on

**CHECKPOINT 2.** Report the tables and a verdict per prediction (supported, not supported or inconclusive). Propose one K for Step 3 with the reason. Wait for Dr. Fady's choice.

## Step 3: tuned comparison (only after approval)

Run the fair2 pipeline with the chosen K fixed, using the same `Protocol` and the same five-knob budget: 60 trials at seed 0, top 8 at seeds 0 to 4, then all 8 at seeds 0 to 34, ranked on mean − sd.

```bash
export TUNING_DIR="$PWD/letter_runs_2fbs/tuning" LETTER_INSTRUMENT=1 LETTER_RUNS_DIR=letter_runs_2fbs
for ALGO in sp_td3 sp_ddpg; do
  ./venv/bin/python -m tuning search  --campaign fair2sp --algo $ALGO --code 2-1-1 --hold-k <K> --workers 8 --slim
  ./venv/bin/python -m tuning confirm --campaign fair2sp --algo $ALGO --code 2-1-1 --hold-k <K> --workers 8 --slim --top 8 --seeds 0 1 2 3 4
  ./venv/bin/python -m tuning confirm --campaign fair2sp --algo $ALGO --code 2-1-1 --hold-k <K> --workers 8 --slim --top 8 --seeds $(seq 0 34)
done
```

- **`--hold-k`:** an opt-in tuner flag that writes K into every trial spec.
- **Directory check:** before launching, check that `TUNING_DIR` is under `letter_runs_2fbs/` and never `tuning_runs/`.
- **Cost:** about 8 h per algorithm plus the DQN overhead measured in Step 2.

**Comparison** (pooled finalists, 8 × 35, against fair2's own records for PPO, TD3, DDPG and the `_qd` hybrids if tuned):

- both_on share, failures, switched-off count, traps
- mean J, winner vs winner

**CHECKPOINT 3.** Report the results.

## Not now

- **1-1-1 (fair1sp):** run the same pipeline on 1-1-1 only if Dr. Fady asks. Stations cannot be abandoned at M = 1, so only the trap and J are tested there.
- **More than 3 FBSs:** the switch would need factored per-FBS heads (branching DQN) instead of 2^N outputs. Scope only.

## Deliverables

- `switchpilot/`, the opt-in hooks, and `tests/test_switchpilot.py`
- `lab/comparison/RESULTS_letter_2fbs_SP_pilot.txt` (with the prereg block) and `RESULTS_letter_2fbs_SP_tuned.txt`
- `letter_runs_2fbs/analysis/sp_pilot_by_run.csv`, `sp_pilot_by_station.csv`
- a dated section in `lab/NOTES.md`: what ran, what it cost, what held, and every contradiction stated plainly
