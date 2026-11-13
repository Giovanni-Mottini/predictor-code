"""Feature importance ranks: which Xsens features separate paretic from non-paretic arms, and
healthy from simulated movement.

Run from the project root:
    <python> -m feature_importance_code run --patients <run> --volunteers <features> [--volunteers ...] \
        --speed length --output reports/feature_importance/<new name>
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


def _versions() -> dict:
    import joblib, matplotlib, numpy, pandas, sklearn
    return {"numpy": numpy.__version__, "pandas": pandas.__version__, "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__, "matplotlib": matplotlib.__version__, "python": sys.version.split()[0]}


def main(argv=None) -> int:
    from .importance import COMPARISONS, REPEATS

    parser = argparse.ArgumentParser(prog="feature_importance_code", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Compute importance and ranks for one or both comparisons")
    run.add_argument("--output", type=Path, required=True, help="New folder (results hold person codes: keep it under reports/)")
    run.add_argument("--patients", type=Path, nargs="+", default=[], help="Patient feature run(s) or features.csv")
    run.add_argument("--volunteers", type=Path, nargs="+", default=[], help="Volunteer features.csv or folders (healthy and simulated)")
    run.add_argument("--speed", choices=["length", "mps", "already-normalized"], required=True,
                     help="length: m/s divided by lever (current tables); already-normalized: speeds divided upstream "
                          "(11/13 Nov speed-variant tables); mps: m/s")
    run.add_argument("--comparison", choices=list(COMPARISONS), action="append",
                     help="Default: every comparison whose inputs are given")
    run.add_argument("--repeats", type=int, default=REPEATS, help="Shuffles per feature and fold")
    run.add_argument("--jobs", type=int, default=-1)
    run.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    from .importance import comparison_rows, importance, summarize
    from .outputs import figures, rank_table
    from .tables import feature_files, labels, read_tables, sha256

    keys = args.comparison or [k for k, c in COMPARISONS.items() if (args.patients if c.cohort == "patients" else args.volunteers)]
    if not keys:
        parser.error("give --patients and/or --volunteers")
    if args.output.exists() and any(args.output.iterdir()):
        print(f"Use a new output folder: {args.output} is not empty", file=sys.stderr)
        return 2
    sources = {"patients": args.patients, "volunteers": args.volunteers}
    tables = {}
    for cohort in {COMPARISONS[k].cohort for k in keys}:
        if not sources[cohort]:
            parser.error(f"{cohort} comparison needs --{cohort}")
        tables[cohort] = read_tables(sources[cohort], args.speed)
    if len(tables) == 2 and set(tables["patients"]["person"]) & set(tables["volunteers"]["person"]):
        print("A person is both a patient and a volunteer", file=sys.stderr)
        return 2

    by_fold, folds, counts = [], [], {}
    for key in keys:
        comparison = COMPARISONS[key]
        rows = comparison_rows(tables[comparison.cohort], comparison)
        print(f"{comparison.title}: {len(rows)} repetitions, {rows['person'].nunique()} people", flush=True)
        fold_values, fold_lines = importance(rows, comparison, args.repeats, args.jobs)
        by_fold.append(fold_values)
        folds.append(fold_lines)
        counts[key] = {"repetitions": len(rows), "people": int(rows["person"].nunique()),
                       "sessions": int(rows[["person", "visit"]].drop_duplicates().shape[0]),
                       "rows_per_class": {c: int(n) for c, n in rows["condition"].value_counts().sort_index().items()},
                       "folds": len(fold_lines), "mean_fold_balanced_accuracy": float(fold_lines["balanced_accuracy"].mean())}

    import pandas as pd
    by_fold, folds = pd.concat(by_fold, ignore_index=True), pd.concat(folds, ignore_index=True)
    summary = summarize(by_fold)
    names = labels(args.speed)
    args.output.mkdir(parents=True, exist_ok=True)
    by_fold.to_csv(args.output / "importance_by_fold.csv", index=False)
    folds.to_csv(args.output / "folds.csv", index=False)
    summary.assign(label=summary["feature"].map(names)).to_csv(args.output / "importance.csv", index=False)
    (args.output / "table_ranks.tex").write_text(rank_table(summary, keys, names))
    if not args.no_figures:
        figures(summary, keys, names, args.output)
    inputs = {cohort: [{"path": str(f), "sha256": sha256(f)} for path in sources[cohort] for f in feature_files(path)]
              for cohort in tables}
    run_info = {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "command": sys.argv,
                "comparisons": keys, "speed": args.speed, "repeats": args.repeats, "counts": counts,
                "inputs": inputs, "packages": _versions()}
    (args.output / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
    for key in keys:
        top = summary[summary["comparison"] == key]
        print(f"{COMPARISONS[key].title}: Gini top 3 {[names[f] for f in top.nsmallest(3, 'gini_rank')['feature']]}, "
              f"held-out top 3 {[names[f] for f in top.nsmallest(3, 'held_out_rank')['feature']]}")
    print(f"Written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
