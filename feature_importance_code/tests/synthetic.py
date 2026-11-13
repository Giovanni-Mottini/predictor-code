"""Synthetic feature tables in the xsens_code layout. Made-up codes only (FG91-FG96, SANO_BA091-...)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from feature_importance_code.tables import ARM_FEATURES, LEVERS


def task_table(people, cohort="PAZIENTI", visits=("T1", "T5"), conditions=("non_paretic", "paretic"), signal="rom_elbow_bending",
               effect=30.0, seed=0, levers=False, whole=False) -> pd.DataFrame:
    """Both arms (or healthy + simulated), 4 tasks x 4 repetitions per visit; `signal` differs by `effect` between them."""
    rng = np.random.default_rng(seed)
    rows, recording = [], 0
    for p_index, person in enumerate(people):
        sides = {"non_paretic": "left", "paretic": "right", "simulated": "right"}
        for visit_index, visit in enumerate(visits):
            for task in (1, 2, 3, 4):
                for condition in conditions:
                    recording += 1
                    for repetition in (1, 2, 3, 4):
                        values = {f: rng.normal(10, 1) for f in ARM_FEATURES}
                        values[signal] += effect if condition in ("paretic", "simulated") else 0.0
                        row = {"participant_key": f"{cohort}/{person}", "recording_id": f"r{recording}",
                               "source_sha256": f"sha{recording}", "timepoint": visit, "task": task, "repetition": repetition,
                               "anatomical_side": sides[condition], "limb_condition": condition,
                               "recorded_at_utc": f"2030-0{1 + visit_index}-1{p_index}T10:00:00+00:00",
                               "method_id": "synthetic", **values}
                        if levers:
                            row |= {lever: 0.5 for lever in LEVERS.values()}
                        if whole:
                            row |= {"activity": f"task{task}", "window_kind": "repetition"}
                        rows.append(row)
    return pd.DataFrame(rows)


def whole_rows(people, activity="mot1", visits=("T1", "T5"), seed=1, signal=None, levers=False) -> pd.DataFrame:
    """One whole recording per arm and visit (B&B, mot1, mot2 layout: no task, no repetition)."""
    rng = np.random.default_rng(seed)
    rows = []
    for p_index, person in enumerate(people):
        for v_index, visit in enumerate(visits):
            for condition, side in (("non_paretic", "left"), ("paretic", "right")):
                values = {f: rng.normal(10, 1) for f in ARM_FEATURES}
                if signal:
                    values |= signal(person, visit, condition)
                row = {"participant_key": f"PAZIENTI/{person}", "recording_id": f"w{person}{visit}{condition}{activity}",
                       "source_sha256": f"w{person}{visit}{condition}{activity}", "timepoint": visit, "task": None,
                       "repetition": None, "anatomical_side": side, "limb_condition": condition, "activity": activity,
                       "window_kind": "whole_recording", "recorded_at_utc": f"2030-0{1 + v_index}-1{p_index}T11:00:00+00:00",
                       "method_id": "synthetic", **values}
                if levers:
                    row |= {lever: 0.5 for lever in LEVERS.values()}
                rows.append(row)
    return pd.DataFrame(rows)


def write_run(root: Path, frame: pd.DataFrame, cohort="PAZIENTI") -> Path:
    """Write one features.csv per person in the run-folder layout."""
    for key, rows in frame.groupby("participant_key"):
        folder = root / "runs" / cohort / key.split("/")[-1] / "xsens"
        folder.mkdir(parents=True, exist_ok=True)
        rows.to_csv(folder / "features.csv", index=False)
    return root
