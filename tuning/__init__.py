"""One tuner for three algorithms, kept apart from all of them.

``ppo/``, ``td3/`` and ``ddpg/`` are not modified by anything here. This
package imports them -- their configs, their ``train`` functions, their
evaluators -- and adds only what a fair comparison needs on top:

    protocol.py   the fairness contract: budget, objective, seed protocol
    spaces.py     per-algorithm flat search spaces + params -> experiment
    objective.py  one scorer: mean J on common start states
    worker.py     one trial = one subprocess (artifact routing, thread limits)
    study.py      GP-based ask/tell orchestrator + stage-2 confirmation
    paths.py      tuning_runs/<campaign>/ owns everything a campaign produces
    cli.py        ``python -m tuning spaces | search | confirm | report | rescore``

The motivation is that the three algorithms have had wildly unequal tuning
effort -- 56 hand-driven runs for DDPG, a learning-rate sweep for TD3, none
at all for PPO -- so comparing them compares search effort as much as it
compares algorithms. Equal budget, one objective, one seed protocol.

Quick start:

    python -m tuning spaces
    python -m tuning search --campaign fair1 --algo ddpg --workers 4
    python -m tuning confirm --campaign fair1 --algo ddpg --workers 4
    python -m tuning report --campaign fair1
"""
from .objective import score_run
from .protocol import ALGOS, DEFAULT, REPORT_SEED, SMOKE, Protocol
from .spaces import SEED_TRIALS, UNSUPPORTED, build_experiment, space, suggest
from .study import confirm, make_study, search

__all__ = [
    "ALGOS",
    "DEFAULT",
    "Protocol",
    "REPORT_SEED",
    "SEED_TRIALS",
    "SMOKE",
    "UNSUPPORTED",
    "build_experiment",
    "confirm",
    "make_study",
    "score_run",
    "search",
    "space",
    "suggest",
]
