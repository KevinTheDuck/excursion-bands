"""Command-line backtesting with explicit, resumable run directories."""

import argparse
from dataclasses import replace
from pathlib import Path

from excursion_bands.backtesting.loader import load_backtest_config, load_core_data
from excursion_bands.backtesting.wfo import run_wfo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/strategies/donchian_1/backtest_robust.yaml"
    )
    parser.add_argument(
        "--run-dir", help="Checkpoint/output directory; required when resuming"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the newest valid checkpoint in --run-dir",
    )
    parser.add_argument(
        "--max-hours", type=float, help="Pause at a safe checkpoint after this budget"
    )
    parser.add_argument(
        "--trials", type=int, help="Trial budget (cannot change when resuming)"
    )
    parser.add_argument(
        "--workers", type=int, help="Independent robust variant workers"
    )
    args = parser.parse_args(argv)
    config = load_backtest_config(args.config)
    updates = {}
    if args.run_dir:
        updates["checkpoint_dir"] = str(Path(args.run_dir))
    if args.resume:
        updates["resume"] = True
    if args.max_hours is not None:
        if args.max_hours <= 0:
            parser.error("--max-hours must be positive")
        updates["max_run_seconds"] = args.max_hours * 3600
    if args.trials is not None:
        updates["n_trials"] = args.trials
    if args.workers is not None:
        updates["robust"] = replace(config.wfo.robust, workers=args.workers)
    config = replace(config, wfo=replace(config.wfo, **updates))
    if not config.wfo.enabled:
        parser.error("This entrypoint requires wfo.enabled: true")
    if config.wfo.resume and not config.wfo.checkpoint_dir:
        parser.error("--resume requires --run-dir or wfo.checkpoint_dir")
    intraday, bands = load_core_data(config.core_data)
    try:
        output = run_wfo(intraday.to_pandas(), bands.to_pandas(), config)
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Committed progress is safe; rerun with the same --run-dir and --resume."
        )
        return 130
    print(
        f"Run directory: {output.resolve()} (see manifest.json for completion status)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
