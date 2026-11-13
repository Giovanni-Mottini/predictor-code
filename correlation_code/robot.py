"""MOTORE robot scores from the fgXX_export CSVs, one session per patient visit.

Each CSV is one robot session (csv_data_<Evaluation|TraIns>_<start UTC>.csv): 17 Name/Value rows,
copied from the session's session.xml. Kept here:
  scores     the 13 scores, with scores.params split into params_1 and params_2
  settings   weight, viscosity, stiffness (Evaluation always 2/4/2; TraIns changes, so its
             scores are not comparable across sessions without them)
Field meanings come from their names only; there is no documentation.

Pairing: the session of that type on the day of the Xsens visit (folder labels in the exports are
not exact around visit days). The exports hold one session per day (the first successful one);
if a day has more, the first is used and the pairing table says so. A copy of one session in two
folders counts once; two different files for one session stop the run.
"""
from __future__ import annotations

import csv
from pathlib import Path
import re

import pandas as pd

ROWS = ("scores.distanceTask", "scores.durationUser", "scores.workTanRobot", "scores.params", "scores.timeUser",
        "scores.workTanUserTotal", "weight", "scores.calories", "scores.workUserTotal", "scores.autonomyDuration",
        "scores.velocityNorm", "scores.precisionWork", "viscosity", "scores.workTanUserSelf", "scores.overall",
        "stiffness", "scores.userVelocityAvg")
SETTINGS = ("weight", "viscosity", "stiffness")
SCORES = ("distanceTask", "durationUser", "timeUser", "workTanRobot", "workTanUserTotal", "workTanUserSelf",
          "workUserTotal", "calories", "autonomyDuration", "velocityNorm", "userVelocityAvg", "precisionWork",
          "overall", "params_1", "params_2")
KINDS = ("Evaluation", "TraIns")
FILE = re.compile(r"csv_data_(Evaluation|TraIns)_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})-(\d{3})Z\.csv")
EXPORT_FOLDER = re.compile(r"(fg\d+)_export", re.I)


def export_files(paths) -> list[tuple[str, Path]]:
    """(patient, CSV) from a patients folder (<ID>/MOTORE/*_export/...) or from one fgXX_export folder."""
    found = []
    for base in map(Path, paths):
        own = EXPORT_FOLDER.fullmatch(base.name)
        if own:
            found += [(own.group(1).upper(), f) for f in sorted(base.rglob("csv_data_*.csv"))]
        else:
            found += [(f.relative_to(base).parts[0], f) for f in sorted(base.glob("*/MOTORE/*_export/**/csv_data_*.csv"))]
    if not found:
        raise ValueError("No MOTORE export CSV found")
    return found


def read_export(path: Path) -> dict:
    name = FILE.fullmatch(path.name)
    if not name:
        raise ValueError(f"{path.name}: not a MOTORE export file name")
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if rows[0] != ["Name", "Value"] or sorted(r[0] for r in rows[1:]) != sorted(ROWS):
        raise ValueError(f"{path.name}: unexpected rows")
    values = {r[0]: r[1] for r in rows[1:]}
    params = [float(v) for v in values["scores.params"].split(",")]
    if len(params) != 2:
        raise ValueError(f"{path.name}: scores.params does not hold two numbers")
    kind, day = name.group(1), name.group(2)
    record = {"kind": kind, "day": day, "start_utc": f"{day} {name.group(3)}:{name.group(4)}:{name.group(5)}.{name.group(6)}"}
    record |= {s: float(values[f"scores.{s}"]) for s in SCORES if not s.startswith("params_")}
    record |= {"params_1": params[0], "params_2": params[1]}
    record |= {s: float(values[s]) for s in SETTINGS}
    return record


def read_sessions(paths) -> pd.DataFrame:
    records = [read_export(f) | {"patient": patient, "file": str(f)} for patient, f in export_files(paths)]
    frame = pd.DataFrame(records)
    key = ["patient", "kind", "start_utc"]
    values = [*SCORES, *SETTINGS]
    conflicts = frame.groupby(key)[values].nunique().gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError(f"{int(conflicts.sum())} sessions have two different export files")
    copies = frame.groupby(key)["file"].transform("size")
    frame = frame.assign(copies=copies).sort_values([*key, "file"]).drop_duplicates(key).reset_index(drop=True)
    return frame[["patient", "kind", "day", "start_utc", *SCORES, *SETTINGS, "copies", "file"]]


def robot_visits(sessions: pd.DataFrame, dates: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores of the session on each Xsens visit day, and a pairing table for every visit."""
    if kind not in KINDS:
        raise ValueError(f"Unknown session type {kind}")
    own = sessions[sessions["kind"] == kind]
    visits, pairing = [], []
    for row in dates.itertuples(index=False):
        same_day = own[(own["patient"] == row.patient) & (own["day"] == row.xsens_date)].sort_values("start_utc")
        line = {"patient": row.patient, "visit": row.visit, "xsens_date": row.xsens_date, "kind": kind,
                "sessions_that_day": len(same_day)}
        if same_day.empty:
            pairing.append(line | {"status": "no session on the Xsens day"})
            continue
        chosen = same_day.iloc[0]
        pairing.append(line | {"status": "used" if len(same_day) == 1 else "used the first of the day",
                               "start_utc": chosen["start_utc"], "file": chosen["file"],
                               **{s: chosen[s] for s in SETTINGS}})
        visits.append({"patient": row.patient, "visit": row.visit, **{s: chosen[s] for s in (*SCORES, *SETTINGS)}})
    return (pd.DataFrame(visits, columns=["patient", "visit", *SCORES, *SETTINGS]), pd.DataFrame(pairing))
