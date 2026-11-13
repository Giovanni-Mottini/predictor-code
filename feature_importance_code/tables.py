"""The feature tables written by xsens_code (features.csv), read the same way by both analyses.

A table has one row per task repetition (tasks 1-4) and, in tables made from 13 Nov 2026 on, one
row per whole recording (B&B, mot1, mot2; column window_kind). Older tables hold task rows only.

Speeds (--speed):
  length              m/s divided by the recording's lever length (columns lever_*_m), as in
                      OUTPUT_SPEED_LENGTH; a table without lever lengths is refused
  mps                 as stored, in m/s
  already-normalized  as stored, already divided upstream (the 11 and 13 Nov 2026 speed-variant
                      tables used for the paper)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

# The 15 xsens_code features without hand DTW, which compares two arms (not a feature of one arm).
ARM_FEATURES = ("vel_mean_trunk", "vel_mean_shoulder", "vel_mean_upper_arm", "vel_mean_forearm",
                "sparc_trunk", "sparc_shoulder", "sparc_upper_arm", "sparc_forearm",
                "rom_shoulder_flex_ext", "rom_shoulder_abd_add", "rom_shoulder_int_ext_rot", "rom_elbow_bending",
                "comp_trunk_lateral_bending", "comp_trunk_flex_ext")
LEVERS = {f"vel_mean_{role}": f"lever_{role}_m" for role in ("trunk", "shoulder", "upper_arm", "forearm")}
SPEED_MODES = ("length", "mps", "already-normalized")
TASK_ACTIVITIES = ("task1", "task2", "task3", "task4")
_LABELS = {"vel_mean_trunk": "Trunk speed", "vel_mean_shoulder": "Shoulder speed",
           "vel_mean_upper_arm": "Upper-arm speed", "vel_mean_forearm": "Forearm speed",
           "sparc_trunk": "Trunk SPARC", "sparc_shoulder": "Shoulder SPARC", "sparc_upper_arm": "Upper-arm SPARC",
           "sparc_forearm": "Forearm SPARC", "rom_shoulder_flex_ext": "Shoulder flexion range",
           "rom_shoulder_abd_add": "Shoulder abduction range", "rom_shoulder_int_ext_rot": "Shoulder rotation range",
           "rom_elbow_bending": "Elbow range", "comp_trunk_lateral_bending": "Trunk lateral range",
           "comp_trunk_flex_ext": "Trunk flexion range"}


def labels(speed: str) -> dict[str, str]:
    """Names used in the tables and figures; divided speeds are called 'Normalized ...'."""
    if speed not in SPEED_MODES:
        raise ValueError(f"Unknown speed mode {speed}")
    if speed == "mps":
        return dict(_LABELS)
    return {f: (f"Normalized {_LABELS[f][0].lower()}{_LABELS[f][1:]}" if f in LEVERS else _LABELS[f]) for f in ARM_FEATURES}


def feature_files(path: Path) -> list[Path]:
    """A features.csv, a folder holding one, or a run folder (runs/<cohort>/<ID>/xsens/features.csv)."""
    path = Path(path)
    if path.is_file():
        return [path]
    if (path / "features.csv").is_file():
        return [path / "features.csv"]
    files = sorted(path.glob("runs/*/*/xsens/features.csv"))
    if not files:
        raise ValueError(f"{path}: no features.csv found")
    return files


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_tables(paths, speed: str) -> pd.DataFrame:
    """All rows of the given tables, one shape for every analysis. Values are never imputed.

    Columns: person, visit, activity, task, repetition, whole, side, condition, recording_id,
    source_sha256, recorded_at_utc, method_id and the 14 arm features.
    """
    if speed not in SPEED_MODES:
        raise ValueError(f"Unknown speed mode {speed}")
    files = [file for path in paths for file in feature_files(path)]
    frame = pd.concat([pd.read_csv(file, low_memory=False) for file in files], ignore_index=True)
    missing = [c for c in ("participant_key", "timepoint", "task", "repetition", "anatomical_side", "limb_condition",
                           "recording_id", *ARM_FEATURES) if c not in frame]
    if missing:
        raise ValueError(f"Feature tables lack columns {missing}")
    if speed == "length":
        absent = [lever for lever in LEVERS.values() if lever not in frame]
        if absent:
            raise ValueError("--speed length needs the lever_*_m columns of the current xsens_code; these tables have "
                             "none. Use --speed already-normalized for tables whose speeds were divided upstream, "
                             "or --speed mps")
        for feature, lever in LEVERS.items():
            length = pd.to_numeric(frame[lever], errors="coerce")
            frame[feature] = frame[feature] / length.where(length > 0)
    whole = frame["window_kind"].eq("whole_recording") if "window_kind" in frame else pd.Series(False, index=frame.index)
    if "activity" in frame:
        activity = frame["activity"]
    else:
        activity = "task" + frame["task"].astype("Int64").astype(str)
    out = pd.DataFrame({
        "person": frame["participant_key"].astype(str).str.split("/").str[-1].str.removeprefix("SANO_"),
        "visit": frame["timepoint"].fillna("unlabelled").astype(str),
        "activity": activity.astype(str),
        "task": frame["task"],
        "repetition": frame["repetition"],
        "whole": whole.to_numpy(),
        "side": frame["anatomical_side"],
        "condition": frame["limb_condition"],
        "recording_id": frame["recording_id"],
        "source_sha256": frame["source_sha256"] if "source_sha256" in frame else None,
        "recorded_at_utc": frame["recorded_at_utc"] if "recorded_at_utc" in frame else None,
        "method_id": frame["method_id"] if "method_id" in frame else None,
    })
    for feature in ARM_FEATURES:
        out[feature] = pd.to_numeric(frame[feature], errors="coerce")
    tasks = ~out["whole"]
    if not out.loc[tasks, "activity"].isin(TASK_ACTIVITIES).all():
        raise ValueError("Task rows with an activity other than task1-task4")
    return out


def task_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Repetitions of tasks 1-4, with integer task and repetition."""
    rows = table[~table["whole"]].copy()
    if rows[["task", "repetition"]].isna().any().any():
        raise ValueError("Task rows without task or repetition number")
    rows["task"] = rows["task"].astype(int)
    rows["repetition"] = rows["repetition"].astype(int)
    if not (rows["task"].isin([1, 2, 3, 4]).all() and rows["repetition"].ge(1).all()):
        raise ValueError("Unexpected task or repetition")
    return rows


def paretic_sides(table: pd.DataFrame) -> dict[str, str]:
    """Patient -> 'left' or 'right', the arm the feature tables label paretic (side overrides included)."""
    sides = table[table["condition"] == "paretic"].groupby("person")["side"].unique()
    many = [person for person, found in sides.items() if len(found) != 1]
    if many:
        raise ValueError(f"{len(many)} patients have more than one paretic side")
    return {person: found[0] for person, found in sides.items()}
