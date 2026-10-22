"""Feature-by-time previews from a verified standardized Garmin window run."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from shared_code.io import atomic_json
from shared_code.reader import contained_path, document, DataContractError, SubjectReader
from shared_code.artifacts import validate_output
from .garmin_workflow import write_frame


def export_matrices(window_output: Path, output: Path, *, max_groups: int = 32):
    """Export bounded day-block matrices and PNGs; preserve missing values."""
    from matplotlib import pyplot as plt

    root = Path(window_output).resolve()
    report, digest = document(root / "report.json")
    readers = [
        SubjectReader(Path(s["provenance"][0]["subject_path"]))
        for s in report["subjects"]
    ]
    output = validate_output(Path(output), readers)
    if output == root:
        raise DataContractError(
            "unsafe_output", "Matrix output needs its own directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    source = contained_path(root, report["files"]["_normalized_day"])
    payload = source.read_bytes()
    expected = report["output_sha256"].get(source.name)
    if not expected or hashlib.sha256(payload).hexdigest() != expected:
        raise DataContractError(
            "window_checksum_mismatch", "Window artifact differs from recorded revision"
        )
    import pyarrow as pa
    import pyarrow.parquet as pq

    frame = pq.read_table(pa.BufferReader(payload)).to_pandas()
    features = report["feature_columns"]
    keys = ["participant_key", "date", "circadian_cycle"]
    if frame.duplicated(keys + ["window_start"]).any():
        raise DataContractError("duplicate_window", "Repeated participant/time windows")
    groups = list(frame.groupby(keys, sort=True))
    if len(groups) > max_groups:
        raise DataContractError(
            "too_many_matrices",
            f"{len(groups)} groups exceeds {max_groups}; narrow the run or explicitly increase the limit",
        )
    index = []
    for (participant, day, cycle), group in groups:
        group = group.sort_values("window_start")
        values = group[features].to_numpy(dtype=float).T
        times = pd.to_datetime(group["window_start"]).dt.strftime("%H:%M").tolist()
        stem = f"{hashlib.sha256(participant.encode()).hexdigest()[:12]}_{day}_cycle{int(cycle)}"
        matrix = pd.DataFrame(values, index=features, columns=times)
        write_frame(output / f"{stem}.csv", matrix.reset_index(names="feature"))
        fig, ax = plt.subplots(
            figsize=(9, max(5, len(features) * 0.23)), layout="constrained"
        )
        cmap = plt.get_cmap("cividis").copy()
        cmap.set_bad("#e6e6e6")
        plot = ax.imshow(
            np.ma.masked_invalid(values),
            vmin=0,
            vmax=1,
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set_yticks(range(len(features)), features, fontsize=9)
        step = max(1, len(times) // 6)
        ticks = sorted(set([0, *range(0, len(times), step), len(times) - 1]))
        ax.set_xticks(ticks, [times[t] for t in ticks], fontsize=10)
        ax.set_xlabel("Source local window start")
        ax.set_title(
            f"{participant} · {day} · {int(cycle)*6:02d}:00–{(int(cycle)+1)*6:02d}:00\nWithin-period normalized features; grey = missing or review excluded",
            fontsize=11,
        )
        fig.colorbar(plot, ax=ax, label="Within-participant scale (0–1)", shrink=0.7)
        fig.savefig(output / f"{stem}.png", dpi=130)
        plt.close(fig)
        index.append(
            {
                "participant_key": participant,
                "date": day,
                "circadian_cycle": int(cycle),
                "windows": len(group),
                "features": len(features),
                "missing_cells": int(np.isnan(values).sum()),
                "csv": f"{stem}.csv",
                "png": f"{stem}.png",
            }
        )
    summary = pd.DataFrame(index)
    write_frame(output / "matrix_index.csv", summary)
    atomic_json(
        output / "manifest.json",
        {
            "source_report_sha256": digest,
            "source_window_sha256": expected,
            "source_window_file": str(source),
            "method_sha256": report["method"]["method_sha256"],
            "groups": index,
            "scientifically_validated": False,
            "training_ready": False,
        },
    )
    return summary
