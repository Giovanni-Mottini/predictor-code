"""One value per patient, visit and feature, for each of the three studies.

tasks          Xsens tasks 1-4, paretic arm. Per visit, task and arm: mean of the repetitions (at least
               2). Per task: z-score over every patient visit of the paretic arm, so the four tasks
               share one scale. Visit value: mean over tasks (at least 2 tasks).
xsens_motore   Xsens recorded during the robot sessions (mot1 = Evaluation, mot2 = TraIns), paretic
               arm, one whole recording per visit, values as recorded. A z-score is not needed: one
               recording per visit, and both correlations ignore a common linear rescaling.
robot          see robot.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from feature_importance_code.tables import ARM_FEATURES, task_rows

MIN_REPETITIONS, MIN_TASKS = 2, 2
MOTORE_ACTIVITIES = {"mot1": "Evaluation", "mot2": "TraIns"}


def task_visits(table: pd.DataFrame, min_repetitions: int = MIN_REPETITIONS, min_tasks: int = MIN_TASKS) -> pd.DataFrame:
    features = list(ARM_FEATURES)
    rows = task_rows(table)
    groups = rows.groupby(["person", "visit", "task", "condition"])
    means = groups[features].mean()
    means["n_repetitions"] = groups.size()
    means = means[means["n_repetitions"] >= min_repetitions].reset_index()
    paretic = means[means["condition"] == "paretic"].set_index(["person", "visit", "task"])
    parts = []
    for task in sorted(paretic.index.get_level_values("task").unique()):
        values = paretic.xs(task, level="task")[features]
        parts.append(((values - values.mean()) / values.std()).assign(task=task).reset_index())
    tasks = pd.concat(parts, ignore_index=True)
    groups = tasks.groupby(["person", "visit"])
    visits = groups[features].mean()
    visits["n_tasks"] = groups.size()
    visits.loc[visits["n_tasks"] < min_tasks, features] = np.nan
    return visits.reset_index().rename(columns={"person": "patient"})


def whole_recording_visits(table: pd.DataFrame, activity: str) -> pd.DataFrame:
    rows = table[table["whole"] & (table["activity"] == activity) & (table["condition"] == "paretic")]
    if rows.duplicated(["person", "visit"]).any():
        raise ValueError(f"More than one paretic {activity} recording in a visit")
    return rows[["person", "visit", *ARM_FEATURES]].rename(columns={"person": "patient"}).reset_index(drop=True)


def visit_dates(table: pd.DataFrame) -> pd.DataFrame:
    """The Xsens day of each patient visit: the most common recording date (UTC) among its recordings."""
    recordings = table.drop_duplicates("recording_id")
    day = pd.to_datetime(recordings["recorded_at_utc"], utc=True, errors="coerce")
    recordings = recordings.assign(day=day.dt.date.astype(str))[day > pd.Timestamp("2000-01-01", tz="UTC")]
    dates = (recordings.groupby(["person", "visit"])["day"].agg(lambda d: d.value_counts().sort_index().idxmax())
             .reset_index().rename(columns={"person": "patient", "day": "xsens_date"}))
    return dates


def pair(scales: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    """Visits with a scale and a feature row, in the scale table's order (patient, then visit number)."""
    return scales.merge(visits, on=["patient", "visit"], how="inner")
