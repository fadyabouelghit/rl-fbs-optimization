"""Command-line interface: ``python -m tuning <command>``.

    spaces    print the search space for each algorithm, gaps included
    search    stage 1: N trials of one algorithm at one seed
    confirm   stage 2: re-run the finalists across seeds, rank on mean - sd
    report    tabulate a campaign
    rescore   re-score a run on the held-out start states

Everything a campaign produces lives under ``tuning_runs/<campaign>/``.

Examples:
    python -m tuning spaces
    python -m tuning search  --campaign fair1 --algo ddpg --workers 4
    python -m tuning search  --campaign smoke --algo ddpg --smoke --workers 2
    python -m tuning confirm --campaign fair1 --algo ddpg --workers 4
    python -m tuning report  --campaign fair1
    python -m tuning rescore --campaign fair1 --algo ddpg --run <run dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

from .protocol import ALGOS, DEFAULT, HYBRID_ALGOS, REPORT_SEED, SMOKE


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tuning", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("spaces", help="print the per-algorithm search spaces")

    def _campaign(sp, algo_required=True):
        sp.add_argument("--campaign", required=True, help="campaign name (a directory)")
        if algo_required:
            sp.add_argument("--algo", required=True, choices=list(ALGOS) + list(HYBRID_ALGOS))
        sp.add_argument("--workers", type=int, default=4,
                        help="concurrent trial subprocesses (default 4). GPSampler "
                             "batches with qLogEI, but the GP still learns more per "
                             "trial at lower concurrency.")
        sp.add_argument("--slim", action="store_true",
                        help="delete steps.csv once a trial is scored (~26 MB of "
                             "every ~27 MB run, and nothing reads it)")
        sp.add_argument("--smoke", action="store_true",
                        help="tiny analytic-backend protocol: seconds, not hours")
        sp.add_argument("--backend", default=None)
        sp.add_argument("--code", default=None,
                        help="scenario code X-Y-Z (default 1-1-1). Changing it "
                             "changes the problem, so give the campaign its own "
                             "name and TUNING_DIR -- a campaign must not mix codes.")
        sp.add_argument("--timesteps", type=int, default=None)
        sp.add_argument("--quiet", action="store_true")

    s = sub.add_parser("search", help="stage 1: search one algorithm")
    _campaign(s)
    s.add_argument("--trials", type=int, default=None, help="default 60")
    s.add_argument("--startup-trials", type=int, default=None,
                   help="random warm-up before the GP leads (default 12)")
    s.add_argument("--seed", type=int, default=None, help="training seed for stage 1")
    s.add_argument("--sampler-seed", type=int, default=0)
    s.add_argument("--wide", action="store_true",
                   help="use the widened boundary-probe space for this algorithm. "
                        "NOT part of the equal-budget comparison -- put it in its "
                        "own campaign.")

    c = sub.add_parser("confirm", help="stage 2: re-run finalists across seeds")
    _campaign(c)
    c.add_argument("--top", type=int, default=None, help="finalists to confirm (default 5)")
    c.add_argument("--seeds", type=int, nargs="+", default=None, help="default 0 1 2 3 4")

    r = sub.add_parser("report", help="tabulate a campaign")
    r.add_argument("--campaign", required=True)
    r.add_argument("--algo", default=None, choices=list(ALGOS) + list(HYBRID_ALGOS))

    rs = sub.add_parser("rescore", help="re-score a run on the held-out start states")
    rs.add_argument("--campaign", required=True)
    rs.add_argument("--algo", required=True, choices=list(ALGOS) + list(HYBRID_ALGOS))
    rs.add_argument("--run", required=True, help="run directory")
    rs.add_argument("--states-seed", type=int, default=REPORT_SEED)
    rs.add_argument("--episodes", type=int, default=None)
    rs.add_argument("--backend", default=None)

    return p


def _protocol(args):
    proto = SMOKE if getattr(args, "smoke", False) else DEFAULT
    edits = {}
    if getattr(args, "slim", False):
        edits["keep_step_log"] = False
    for attr, field in (("trials", "n_trials"), ("startup_trials", "n_startup_trials"),
                        ("seed", "search_seed"), ("top", "confirm_top_k"),
                        ("backend", "backend"), ("timesteps", "total_timesteps"),
                        ("code", "code")):
        val = getattr(args, attr, None)
        if val is not None:
            edits[field] = val
    if getattr(args, "seeds", None):
        edits["confirm_seeds"] = tuple(args.seeds)
    return replace(proto, **edits) if edits else proto


def cmd_spaces(args) -> int:
    from .spaces import KNOBS_PER_ALGO, SHARED, SPECIFIC, UNSUPPORTED, space

    proto = DEFAULT
    print("FIXED -- one common value for all three, not searched")
    print(f"  {'learning_rate':22s} {proto.fixed_learning_rate:g}")
    print(f"  {'gamma':22s} {proto.fixed_gamma:g}")
    print(f"  {'batch_size':22s} {proto.fixed_batch_size}")
    print(f"  {'net_arch':22s} {','.join(map(str, proto.fixed_net_arch))}"
          f"   (SB3's PPO default; ppo/ cannot express any other)")
    for k in SHARED:                      # currently empty, kept for clarity
        print(f"  {k.name:22s} SEARCHED {k.range_str}")

    print(f"\nSEARCHED -- {KNOBS_PER_ALGO} knobs per algorithm")
    for algo in ALGOS:
        knobs = space(algo)
        assert len(knobs) == KNOBS_PER_ALGO
        print(f"\n  {algo.upper()}")
        for k in knobs:
            print(f"    {k.name:22s} {k.range_str:28s} {k.note}")

    print("\nNOT SEARCHABLE -- the package does not implement the knob.")
    if not UNSUPPORTED:
        print("  (none)")
    for (algo, knob), why in UNSUPPORTED.items():
        print(f"\n  {algo}.{knob}")
        for line in why.split(". "):
            if line.strip():
                print(f"      {line.strip().rstrip('.')}.")
    return 0


def cmd_search(args) -> int:
    from .study import search

    if getattr(args, "wide", False):
        print("*** WIDENED SPACE: boundary probe, not part of the equal-budget "
              "comparison ***")
    search(args.campaign, args.algo, _protocol(args), workers=args.workers,
           sampler_seed=args.sampler_seed, wide=getattr(args, "wide", False),
           quiet=args.quiet)
    return 0


def cmd_confirm(args) -> int:
    from .study import confirm

    summary = confirm(args.campaign, args.algo, _protocol(args),
                      workers=args.workers, quiet=args.quiet)
    win = summary["winner"]
    if win:
        print(f"\nwinner: trial {win['trial']}  mean {win['mean']:.4f} "
              f"sd {win['sd']:.4f}  score {win['score']:.4f}")
        print(json.dumps(win["params"], indent=2))
    return 0


def cmd_report(args) -> int:
    import pandas as pd

    from . import paths

    d = paths.study_dir(args.campaign)
    if not d.is_dir():
        raise SystemExit(f"no such campaign: {d}")
    algos = [args.algo] if args.algo else list(ALGOS)
    pd.set_option("display.width", 220)
    for algo in algos:
        csv = d / f"{algo}_trials.csv"
        if not csv.is_file():
            continue
        df = pd.read_csv(csv)
        ok = df[df["state"] == "COMPLETE"].sort_values("objective", ascending=False)
        print(f"\n=== {algo}: {len(ok)} complete / {len(df)} trials ===")
        cols = [c for c in ("trial", "objective", "objective_sd", "gap_closed_pct",
                            "train_seconds") if c in ok.columns]
        cols += [c for c in ok.columns if c.startswith("p_")]
        print(ok[cols].head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        conf = d / f"{algo}_confirmation.json"
        if conf.is_file():
            s = json.loads(conf.read_text())
            print(f"\n  stage 2 ({s['ranking_rule']}, seeds {s['seeds']}):")
            for row in s["finalists"]:
                print(f"    trial {row['trial']:4d}  mean {row['mean']:.4f}  "
                      f"sd {row['sd']:.4f}  score {row['score']:.4f}  "
                      f"n={row['n_ok']}")
    return 0


def cmd_rescore(args) -> int:
    from .objective import score_run

    proto = _protocol(args)
    out = score_run(args.algo, args.run, proto, states_seed=args.states_seed,
                    episodes=args.episodes)
    print(json.dumps(out, indent=2))
    return 0


def main(argv=None) -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = _build_parser().parse_args(argv)
    return {
        "spaces": cmd_spaces, "search": cmd_search, "confirm": cmd_confirm,
        "report": cmd_report, "rescore": cmd_rescore,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
