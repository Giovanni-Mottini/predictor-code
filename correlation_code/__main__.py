"""Correlation of movement measures with Fugl-Meyer and Box and Blocks, between and within patients.

Three studies, one command each (run from the project root):
    <python> -m correlation_code tasks         --features <run> --workbooks <dir> --speed length --output <dir>
    <python> -m correlation_code xsens-motore  --features <run> --workbooks <dir> --speed length --output <dir>
    <python> -m correlation_code robot         --features <run> --workbooks <dir> --exports <dir> [<dir> ...] --output <dir>
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

STUDIES = ("tasks", "xsens-motore", "robot")
SCALES = ("FMA", "BBT")


def _versions() -> dict:
    import numpy, openpyxl, pandas, scipy
    return {"numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
            "openpyxl": openpyxl.__version__, "python": sys.version.split()[0]}


def _order(path: Path | None, features: list[str]) -> list[str]:
    """Feature order: the paretic-vs-non-paretic Gini ranking of feature_importance_code, if given."""
    if path is None:
        return list(features)
    import pandas as pd
    ranks = pd.read_csv(path)
    ranks = ranks[ranks["comparison"] == "paretic_vs_non_paretic"].sort_values("gini", ascending=False, kind="stable")
    order = [f for f in ranks["feature"] if f in features]
    if set(order) != set(features):
        raise ValueError(f"{path}: the ranking does not cover every feature")
    return order


def main(argv=None) -> int:
    from .statistics import PERMUTATIONS, SEED

    parser = argparse.ArgumentParser(prog="correlation_code", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="study", required=True)
    for study in STUDIES:
        sub = commands.add_parser(study)
        sub.add_argument("--features", type=Path, nargs="+", required=True,
                         help="Patient feature run(s) or features.csv (paretic side, visits, Xsens dates)")
        sub.add_argument("--workbooks", type=Path, required=True, help="Folder with <ID>/<clinical workbook>.xlsx")
        sub.add_argument("--output", type=Path, required=True, help="New folder (results hold patient codes: keep it under reports/)")
        sub.add_argument("--seed", type=int, default=SEED)
        sub.add_argument("--permutations", type=int, default=PERMUTATIONS)
        sub.add_argument("--jobs", type=int, default=-1, help="Parallel tests; results do not depend on it")
        if study != "robot":
            sub.add_argument("--speed", choices=["length", "mps", "already-normalized"], required=True,
                             help="length: m/s divided by lever (current tables); already-normalized: speeds divided "
                                  "upstream (11/13 Nov speed-variant tables); mps: m/s")
            sub.add_argument("--order", type=Path, help="importance.csv of feature_importance_code: table rows in its "
                                                        "paretic-vs-non-paretic Gini order")
        else:
            sub.add_argument("--exports", type=Path, nargs="+", required=True,
                             help="Patients folder (<ID>/MOTORE/*_export) and/or single fgXX_export folders")
    args = parser.parse_args(argv)

    import pandas as pd
    from feature_importance_code.tables import ARM_FEATURES, feature_files, labels, paretic_sides, read_tables, sha256
    from .outputs import correlation_table, summary_block
    from .robot import KINDS, SCORES, read_sessions, robot_visits
    from .scales import read_scales, workbook_of
    from .sources import MOTORE_ACTIVITIES, pair, task_visits, visit_dates, whole_recording_visits
    from .statistics import correlate

    if args.output.exists() and any(args.output.iterdir()):
        print(f"Use a new output folder: {args.output} is not empty", file=sys.stderr)
        return 2
    speed = getattr(args, "speed", "mps")      # the robot study reads no speed
    table = read_tables(args.features, speed)
    table = table[table["condition"].isin(["paretic", "non_paretic"])]
    sides = paretic_sides(table)
    scales, notes = read_scales(args.workbooks, sides)

    families = []           # (name, title, caption, label, visits, features, labels, order, notes, extra tables)
    if args.study in ("tasks", "xsens-motore"):
        names = labels(speed)
        order = _order(args.order, list(ARM_FEATURES))
        if args.study == "tasks":
            families.append(("tasks", "Xsens tasks 1-4, paretic arm", "Correlation of the paretic-arm features with the clinical scales",
                             "tab:correlations", task_visits(table), names, order, [], {}))
        else:
            if not table["whole"].any():
                print("These feature tables have no mot1/mot2 rows: extract them with the current xsens_code", file=sys.stderr)
                return 2
            for activity, kind in MOTORE_ACTIVITIES.items():
                families.append((activity, f"Xsens {activity} ({kind}), paretic arm",
                                 f"Correlation of the paretic-arm features during the robot {kind} ({activity}) with the clinical scales",
                                 f"tab:correlations-{activity}", whole_recording_visits(table, activity), names, order, [], {}))
    else:
        sessions = read_sessions(args.exports)
        dates = visit_dates(table)
        for kind in KINDS:
            visits, pairing = robot_visits(sessions, dates, kind)
            own = sessions[sessions["kind"] == kind]
            off = int((own[["weight", "viscosity", "stiffness"]] != [2.0, 4.0, 2.0]).any(axis=1).sum())
            pair_notes = [f"{kind} sessions read: {len(own)} ({int(own['copies'].gt(1).sum())} found in two folders); "
                          f"not at weight 2 / viscosity 4 / stiffness 2: {off}",
                          f"Xsens visits: {len(pairing)}; " + ", ".join(f"{s}: {n}" for s, n in pairing["status"].value_counts().items())]
            families.append((f"robot_{kind.lower()}", f"MOTORE {kind} scores", f"Correlation of the robot {kind} scores with the clinical scales",
                             f"tab:correlations-robot-{kind.lower()}", visits, {s: s for s in SCORES}, list(SCORES), pair_notes,
                             {"pairing": pairing}))

    args.output.mkdir(parents=True, exist_ok=True)
    scales.to_csv(args.output / "scales.csv", index=False)
    summary = [f"CORRELATION WITH THE CLINICAL SCALES: {args.study}", "",
               f"scales: {len(scales)} visits of {scales['patient'].nunique()} patients (workbook T0 -> T1, T5+ same label)",
               *notes, "", ""]
    for name, title, caption, label, source, names, order, family_notes, extra in families:
        visits = pair(scales, source)
        result = correlate(visits, order, SCALES, args.seed, args.permutations, args.jobs)
        visits.to_csv(args.output / f"{name}_visits.csv", index=False)
        result.assign(label=result["feature"].map(lambda f: names.get(f, f))).to_csv(args.output / f"{name}_results.csv", index=False)
        (args.output / f"{name}_table.tex").write_text(correlation_table(result, order, names, caption, label, args.permutations))
        for key, frame in extra.items():
            frame.to_csv(args.output / f"{name}_{key}.csv", index=False)
        summary += summary_block(name, title, visits, result, order, names, family_notes)
        print(f"{title}: {len(visits)} visits, {visits['patient'].nunique()} patients", flush=True)
    (args.output / "summary.txt").write_text("\n".join(summary))
    inputs = {"features": [{"path": str(f), "sha256": sha256(f)} for path in args.features for f in feature_files(path)],
              "workbooks": [{"patient": p, "sha256": sha256(workbook_of(args.workbooks / p))} for p in sorted(sides)
                            if (args.workbooks / p).is_dir()]}
    if args.study == "robot":
        inputs["exports"] = [str(p) for p in args.exports]
    run_info = {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "command": sys.argv,
                "study": args.study, "speed": speed, "seed": args.seed, "permutations": args.permutations,
                "scales": list(SCALES), "inputs": inputs, "packages": _versions()}
    (args.output / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
    print(f"Written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
