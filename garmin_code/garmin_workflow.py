"""Atomic Garmin artifacts and callable notebook/CLI workflow."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy

from shared_code.provenance import implementation_hashes
from shared_code.io import atomic_json
from shared_code.artifacts import atomic_text, now, validate_output
from shared_code.reader import DataContractError
from shared_code.progress import ProgressReporter
from .garmin import GarminSubject, GarminConfig, build_windows, DEFERRED_HRV


def write_frame(path: Path, frame: pd.DataFrame):
    fd, temp = tempfile.mkstemp(prefix=".garmin-", dir=path.parent)
    os.close(fd)
    try:
        if path.suffix == ".parquet":
            pq.write_table(
                pa.Table.from_pandas(frame, preserve_index=False),
                temp,
                compression="zstd",
            )
        else:
            frame.to_csv(temp, index=False)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def output_directory(output, runs):
    if not runs or len({run.participant_key for run in runs}) != len(runs):
        raise DataContractError(
            "duplicate_or_empty_subjects", "Supply distinct explicit participants"
        )
    output = validate_output(Path(output), [run.reader for run in runs])
    output.mkdir(parents=True, exist_ok=True)
    return output


def method_document(config):
    return {
        **config.method(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "scipy": scipy.__version__,
        },
        "implementation_sha256": implementation_hashes(('garmin_code/garmin.py', 'garmin_code/garmin_hrv.py', 'garmin_code/garmin_hrv_notebook.py', 'garmin_code/garmin_reader.py', 'garmin_code/garmin_workflow.py', 'garmin_code/garmin_matrices.py', 'garmin_code/__main__.py', 'shared_code/reader.py', 'shared_code/artifacts.py', 'shared_code/io.py', 'shared_code/provenance.py', 'shared_code/progress.py')),
    }


def qc_frame(subjects):
    rows = []
    for subject in subjects:
        for name, stream in subject["streams"].items():
            for flag in stream["findings"]:
                rows.append(
                    {
                        "participant_key": subject["participant_key"],
                        "stream": name,
                        **{
                            k: flag[k]
                            for k in ("code", "feature", "severity", "count", "rule")
                        },
                        "examples_json": json.dumps(flag["examples"], allow_nan=False),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "participant_key",
            "stream",
            "code",
            "feature",
            "severity",
            "count",
            "rule",
            "examples_json",
        ],
    )


def export_preparation(runs: list[GarminSubject], stream: str, output: Path):
    output = output_directory(output, runs)
    frames = []
    for run in runs:
        prepared = run.prepare(stream)
        frame = prepared.data[
            [c for c in prepared.data if not c.startswith("_")]
        ].copy()
        frame["row_eligible"] = prepared.row_eligible
        for col in prepared.values:
            frame[f"{col}__numeric"] = prepared.values[col].where(
                np.isfinite(prepared.values[col])
            )
            frame[f"{col}__eligible"] = prepared.eligible[col]
            frame[f"{col}__review"] = prepared.review[col]
        frame["method_sha256"] = run.config.method()["method_sha256"]
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    write_frame(output / f"{stream}_prepared.parquet", combined)
    subjects = [r.report() for r in runs]
    atomic_json(
        output / "report.json",
        {
            "stream": stream,
            "created_at": now(),
            "subjects": subjects,
            "methods": [method_document(r.config) for r in runs],
            "ready_for_features": False,
        },
    )
    write_frame(output / "qc_flags.csv", qc_frame(subjects))
    return combined, pd.read_csv(output / "qc_flags.csv")


def run_windows(runs: list[GarminSubject], output: Path, *, progress: ProgressReporter | None = None):
    output = output_directory(output, runs)
    methods = {r.config.method()["method_sha256"] for r in runs}
    if len(methods) != 1:
        raise DataContractError(
            "inconsistent_methods",
            "One artifact batch requires one Garmin configuration",
        )
    started = now()
    frames = []
    normalized = []
    scaling = []
    features = None
    hrv_blocks = []
    hrv_epochs = []
    for index, run in enumerate(runs, 1):
        if progress:
            progress.update(f"Starting participant {run.participant_key}: {run.start} to {run.end} (exclusive)",
                            completed=index - 1, total=len(runs))
        raw, norm, names = build_windows(run, progress=progress)
        if features is not None and features != names:
            raise DataContractError(
                "feature_schema_mismatch",
                "Participants have different source feature semantics; run separate batches",
            )
        features = names
        frames.append(raw)
        normalized.append(norm)
        scaling.extend(norm.attrs["scaling"])
        hrv_blocks.append(run.hrv_blocks)
        hrv_epochs.append(run.hrv_epochs)
        if progress:
            progress.update(f"Participant windows processed: {run.participant_key}; {len(raw)} windows",
                            completed=index, total=len(runs))
    if progress:
        progress.update(f"Writing Garmin feature/HRV/QC artifacts to {output}")
    frame = pd.concat(frames, ignore_index=True)
    norm = pd.concat(normalized, ignore_index=True)
    label = f"{runs[0].config.window_minutes}min"
    files = {}
    for suffix, subset in (
        ("", frame),
        ("_day", frame.loc[frame["circadian_cycle"].ne(0)]),
        ("_night", frame.loc[frame["circadian_cycle"].eq(0)]),
        ("_normalized", norm),
        ("_normalized_day", norm.loc[norm["circadian_cycle"].ne(0)]),
        ("_normalized_night", norm.loc[norm["circadian_cycle"].eq(0)]),
    ):
        stem = f"{label}_wearable{suffix}"
        if progress:
            progress.update(f"Writing {stem}.parquet and {stem}.csv")
        files[suffix or "full"] = f"{stem}.parquet"
        write_frame(output / f"{stem}.parquet", subset)
        write_frame(output / f"{stem}.csv", subset)
    for stem, parts in (("hrv_blocks", hrv_blocks), ("hrv_epochs", hrv_epochs)):
        combined = pd.concat(parts, ignore_index=True)
        write_frame(output / f"{stem}.parquet", combined)
        write_frame(output / f"{stem}.csv", combined)
        files[stem] = f"{stem}.parquet"
    subjects = [r.report() for r in runs]
    qc = qc_frame(subjects)
    write_frame(output / "qc_flags.csv", qc)
    atomic_json(
        output / "scaling.json",
        {
            "fit_scope": "each participant's explicit selected period; descriptive only",
            "parameters": scaling,
        },
    )
    atomic_json(output / "method.json", method_document(runs[0].config))
    report = {
        "report_version": 2,
        "started_at": started,
        "finished_at": now(),
        "method": runs[0].config.method(),
        "subjects": subjects,
        "feature_columns": features,
        "deferred_hrv_fields": list(DEFERRED_HRV),
        "windows": len(frame),
        "files": files,
        "ready_for_features": False,
        "scientifically_validated": False,
        "ema_joined": False,
        "output_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted([*output.glob(f"{label}_wearable*"), *output.glob("hrv_*.csv"), *output.glob("hrv_*.parquet")])
        },
    }
    atomic_json(output / "report.json", report)
    lines = [
        "# Garmin standardized window pilot",
        "",
        f"Method: `{report['method']['method_id']}`.",
        "",
        "| Participant | Local start | Local end (exclusive) | Windows | Health coverage >0 |",
        "|---|---|---|---:|---:|",
    ]
    for run in runs:
        part = frame.loc[frame["participant_key"].eq(run.participant_key)]
        lines.append(
            f"| {run.participant_key} | {run.start} | {run.end} | {len(part)} | {int(part['health_coverage_pct'].gt(0).sum())} |"
        )
    lines += [
        "",
        "Window features are means of eligible measurements whose start falls in [start,end). Total, active and legacy unspecified calories stay separate. High-rate calorie means remain provisional in the feature table and become null in normalization. No source values are repaired.",
        "",
        "Six-hour HRV is repeated across smaller windows. Time-domain pairs respect original rows, UTC gaps and declared gap markers. Advanced HRV uses continuous source-row/UTC segments: complete 300s LF/HF epochs, separate 3600s VLF epochs, exact contiguous beat epochs for ApEn/SampEn, and Poincare statistics on accepted pairs. Block summaries are medians of finite epochs, with counts/statuses. Raw BBI only; pairs whose UTC step differs from the following BBI by more than the clock tolerance are broken. Missing epochs stay null. hrv_blocks contains one row per block; hrv_epochs retains source slices and numerical evidence. These descriptors do not certify stationarity or sinus rhythm.",
        "",
        "Coverage is measured health-interval coverage. Wearing percentage is duration-weighted within observed intervals (uniform within each source interval); a window with no observed intervals has null wear, never an assumed 100%. Local labels are wall-clock labels; ambiguous local observations and transition health intervals are excluded with flags.",
        "",
        "Sleep uses one consistent session summary per local end date and UTC session duration, including awake time. Repeated phase summaries collapse; multiple daily sessions remain ambiguous and produce null daily values. RHR uses its local start date. Daily and six-hour measures repeat across windows; they are not independent measurements.",
        "",
        "Normalization fits the requested participant period only and preserves nulls/review exclusions. This descriptive fit must not be reused as a training split scaler. EMA/questionnaires and AlexNet do not run, and first-wave eligibility sheets are not applied to new cohorts.",
        "",
        "SHA-256 verifies standardized tables and metadata snapshots. Raw source hashes are inherited provenance, not a fresh source read. Local conversion and this bounded run do not establish remote completeness or scientific validity. ready_for_features remains false.",
        "",
        f"Rows in QC summary: {len(qc)}. Counts may overlap between rules; do not add them as unique excluded measurements.",
        "",
        "[Features]("
        + files["full"]
        + "), [QC flags](qc_flags.csv), [provenance](report.json), [method](method.json), [scaling](scaling.json).",
        "",
    ]
    atomic_text(output / "report.md", "\n".join(lines))
    if progress:
        progress.update(f"Window artifacts written: {output / 'report.md'}")
    return frame, norm, report


def runs_from_ranges(paths, ranges, config=None):
    readers = []
    from shared_code.reader import SubjectReader

    for path in paths:
        readers.append(SubjectReader(path))
    keys = {r.participant_key for r in readers}
    if set(ranges) != keys:
        raise DataContractError(
            "range_identity_mismatch",
            "Date ranges must exactly match supplied cohort-qualified participants",
        )
    return [
        GarminSubject(
            r.root,
            ranges[r.participant_key]["start"],
            ranges[r.participant_key]["end_exclusive"],
            config,
        )
        for r in readers
    ]
