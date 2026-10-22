"""Explicit local-calendar monitoring periods over verified Garmin artifacts."""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from shared_code.artifacts import atomic_text, now, validate_output
from shared_code.io import atomic_json
from shared_code.provenance import implementation_hashes
from shared_code.reader import DataContractError, SubjectReader, contained_path, document
from .garmin import ADVANCED_HRV, HEALTH_FEATURES, HRV_FIELDS, METHOD_ID
from .garmin_workflow import write_frame

ALIGNMENT_METHOD = "garmin-explicit-local-periods-v1"
DAILY_FEATURES = ("rhr_avg", "sleep_score", "sleep_session_hours")
SAMPLE_FEATURES = tuple(f"{f}_mean" for f in (*HEALTH_FEATURES, "stress_level_value", "body_battery", "respiration_breaths_min"))


def _fail(code, message):
    raise DataContractError(code, message)


def _readers(paths):
    readers = [SubjectReader(p) for p in paths]
    if not readers or len({r.participant_key for r in readers}) != len(readers):
        _fail("duplicate_or_empty_subjects", "Supply distinct cohort-qualified participants")
    return {r.participant_key: r for r in readers}


def _labels(reader):
    labels = {r["timepoint"] for r in reader.select(modality="xsens") if r.get("timepoint")}
    timeline = reader.subject.get("clinical_metadata", {}).get("timeline_dates", {})
    labels.update(re.sub(r"_D\d+$", "", key) for key in timeline)
    return labels, timeline


def period_template(paths):
    """Preserve clinical date evidence; leave monitoring boundaries unset."""
    participants = {}
    for key, reader in _readers(paths).items():
        labels, timeline = _labels(reader)
        participants[key] = [
            {"period_id": label, "timepoint": label, "protocol_day": None,
             "start": None, "end_exclusive": None, "definition_source": None,
             "clinical_dates": {k: v for k, v in timeline.items() if k == label or k.startswith(label + "_D")}}
            for label in sorted(labels, key=lambda v: (int(v[1:]) if re.fullmatch(r"T\d+", v) else 10**9, v))
        ]
    return {"format_version": 1, "time_basis": "source_local_wall_clock", "participants": participants}


def validate_periods(value, readers):
    if (not isinstance(value, dict) or set(value) != {"format_version", "time_basis", "participants"}
            or type(value.get("format_version")) is not int or value.get("format_version") != 1
            or value.get("time_basis") != "source_local_wall_clock"):
        _fail("invalid_period_document", "Use format_version 1 and source_local_wall_clock")
    participants = value.get("participants")
    if not isinstance(participants, dict) or set(participants) != set(readers):
        _fail("period_identity_mismatch", "Periods must exactly match supplied cohort-qualified participants")
    periods = []
    allowed = {"period_id", "timepoint", "protocol_day", "start", "end_exclusive", "definition_source", "clinical_dates"}
    for key, items in participants.items():
        if not isinstance(items, list) or not items:
            _fail("empty_periods", f"Supply at least one period for {key}")
        labels, timeline = _labels(readers[key])
        seen, subject_periods = set(), []
        for item in items:
            if not isinstance(item, dict) or set(item) - allowed:
                _fail("invalid_period", "Unknown period fields or non-object period")
            pid, label, day = item.get("period_id"), item.get("timepoint"), item.get("protocol_day")
            if not isinstance(pid, str) or not pid.strip() or pid in seen:
                _fail("invalid_period_id", f"Period IDs must be nonempty and unique within {key}")
            if not isinstance(label, str) or label not in labels:
                _fail("unknown_timepoint", f"Timepoint {label!r} is absent from canonical metadata for {key}")
            if day is not None and (type(day) is not int or not 1 <= day <= 5):
                _fail("invalid_protocol_day", "protocol_day must be null or an integer in 1..5")
            if day is not None and f"{label}_D{day}" not in timeline and not any(
                r.get("timepoint") == label and r.get("protocol_day") == day for r in readers[key].manifest["records"]
            ):
                _fail("unknown_protocol_day", f"No canonical day {day} for {key}/{label}")
            try:
                start, end = date.fromisoformat(item["start"]), date.fromisoformat(item["end_exclusive"])
            except (KeyError, ValueError, TypeError) as exc:
                raise DataContractError("period_boundaries_required", "Supply explicit YYYY-MM-DD start/end_exclusive for every period") from exc
            if (start.isoformat() != item["start"] or end.isoformat() != item["end_exclusive"] or start >= end):
                _fail("invalid_period_dates", "Require canonical local dates with start < end_exclusive")
            source = item.get("definition_source")
            if not isinstance(source, str) or not source.strip():
                _fail("period_source_required", "Record the protocol or decision defining each monitoring interval")
            evidence = {k: v for k, v in timeline.items() if k == label or k.startswith(label + "_D")}
            if "clinical_dates" in item and item["clinical_dates"] != evidence:
                _fail("clinical_evidence_mismatch", "Template clinical_dates differs from current canonical metadata")
            subject_periods.append({"participant_key": key, "period_id": pid, "timepoint": label,
                                    "protocol_day": day, "start": start.isoformat(), "end_exclusive": end.isoformat(),
                                    "definition_source": source, "clinical_dates_json": json.dumps(evidence, sort_keys=True)})
            seen.add(pid)
        ordered = sorted(subject_periods, key=lambda p: p["start"])
        if any(a["end_exclusive"] > b["start"] for a, b in zip(ordered, ordered[1:])):
            _fail("overlapping_periods", f"Monitoring periods overlap for {key}; define a separate overlap policy first")
        periods.extend(ordered)
    return periods


def _artifact(root, report, name):
    path = contained_path(root, report["files"][name])
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != report["output_sha256"].get(path.name):
        _fail("window_checksum_mismatch", f"Changed artifact: {path}")
    frame = pq.read_table(pa.BufferReader(payload)).to_pandas()
    return frame, {"path": str(path), "sha256": digest}


def load_windows(paths, readers):
    """Read only declared raw features, enforcing artifact and subject revisions."""
    if not paths:
        _fail("windows_required", "Supply at least one Garmin window output directory")
    frames, blocks, inputs, features, method = [], [], [], None, None
    for supplied in paths:
        root = Path(supplied).resolve()
        report, digest = document(root / "report.json")
        if report["method"]["method_id"] != METHOD_ID:
            _fail("unsupported_window_method", "Alignment requires the current continuous-HRV window method")
        if method is not None and method != report["method"]:
            _fail("inconsistent_methods", "All window artifacts must use the same Garmin configuration")
        method = report["method"]
        names = report["feature_columns"]
        if (not isinstance(names, list) or len(names) != len(set(names))
                or not set(names) <= set((*SAMPLE_FEATURES, *HRV_FIELDS, *ADVANCED_HRV, *DAILY_FEATURES))):
            _fail("unsupported_feature_schema", "Unknown or duplicate feature semantics")
        if features is not None and features != names:
            _fail("feature_schema_mismatch", "Source feature meanings differ; align separate batches")
        features = names
        declared = {s["participant_key"] for s in report["subjects"]}
        for subject in report["subjects"]:
            key = subject["participant_key"]
            if key in readers:
                if not subject["provenance"] or any(p["subject_metadata_sha256"] != readers[key].metadata_sha256 for p in subject["provenance"]):
                    _fail("alignment_snapshot_mismatch", f"Window inputs use a different subject/manifest revision: {key}")
        raw, raw_ref = _artifact(root, report, "full")
        hrv, hrv_ref = _artifact(root, report, "hrv_blocks")
        if len(raw) != report["windows"] or set(raw["participant_key"]) != declared or set(hrv["participant_key"]) != declared:
            _fail("window_identity_mismatch", "Artifact participants/row count disagree with report")
        if not raw["method_sha256"].eq(method["method_sha256"]).all() or not raw["method_id"].eq(METHOD_ID).all():
            _fail("window_method_mismatch", "Artifact rows disagree with method provenance")
        for frame in (raw, hrv):
            frame["window_artifact"] = str(root)
        frames.append(raw.loc[raw["participant_key"].isin(readers)].copy())
        blocks.append(hrv.loc[hrv["participant_key"].isin(readers)].copy())
        inputs.append({"root": str(root), "report_sha256": digest, "windows": raw_ref, "hrv_blocks": hrv_ref,
                       "source_subjects": report["subjects"]})
    frame, hrv = pd.concat(frames, ignore_index=True), pd.concat(blocks, ignore_index=True)
    if set(frame["participant_key"]) != set(readers):
        _fail("missing_window_participant", "Each supplied participant must have a declared window artifact")
    minutes = method["config"]["window_minutes"]
    for col in ("window_start", "window_end", "hrv_block_start"):
        frame[col] = pd.to_datetime(frame[col], errors="raise")
        if frame[col].dt.tz is not None or frame[col].isna().any():
            _fail("invalid_local_time", "Expected complete naive source-local window labels")
    hrv["hrv_block_start"] = pd.to_datetime(hrv["hrv_block_start"], errors="raise")
    if hrv["hrv_block_start"].dt.tz is not None or hrv["hrv_block_start"].isna().any():
        _fail("invalid_local_time", "Expected source-local HRV labels")
    if (not frame["time_basis"].eq("source_local_wall_clock").all()
            or not (frame["window_end"] - frame["window_start"]).eq(pd.Timedelta(minutes=minutes)).all()
            or not frame["window_start"].eq(frame["window_start"].dt.floor(f"{minutes}min")).all()
            or not frame["hrv_block_start"].eq(frame["window_start"].dt.floor("6h")).all()
            or not frame["date"].eq(frame["window_start"].dt.strftime("%Y-%m-%d")).all()
            or not hrv["hrv_block_start"].eq(hrv["hrv_block_start"].dt.floor("6h")).all()):
        _fail("invalid_window_grid", "Window/HRV labels do not follow the declared local grid")
    if frame.duplicated(["participant_key", "window_start"]).any() or hrv.duplicated(["participant_key", "hrv_block_start"]).any():
        _fail("duplicate_window", "Input runs overlap; each window/HRV block must occur once")
    # HRV block evidence must agree with the values repeated on window rows.
    hrv_keys = ["participant_key", "hrv_block_start"]
    if set(map(tuple, frame[hrv_keys].drop_duplicates().to_numpy())) != set(map(tuple, hrv[hrv_keys].to_numpy())):
        _fail("hrv_block_mismatch", "HRV evidence and window blocks disagree")
    hrv_fields = [c for c in hrv if c not in {*hrv_keys, "window_artifact"}]
    joined = frame.merge(hrv[hrv_keys + hrv_fields], on=hrv_keys, suffixes=("", "__block"), validate="many_to_one")
    for col in hrv_fields:
        if col not in frame or not (joined[col].eq(joined[col + "__block"]) | (joined[col].isna() & joined[col + "__block"].isna())).all():
            _fail("hrv_block_mismatch", f"Window and block evidence differs: {col}")
    return frame, hrv, features, method, inputs


def _assign(frame, periods, date_column):
    out = frame.copy()
    for column in ("period_id", "timepoint", "protocol_day"):
        out[column] = pd.Series(None, index=out.index, dtype=object)
    for period in periods:
        mask = (out["participant_key"].eq(period["participant_key"])
                & out[date_column].ge(pd.Timestamp(period["start"]))
                & out[date_column].lt(pd.Timestamp(period["end_exclusive"])))
        for column in ("period_id", "timepoint", "protocol_day"):
            out.loc[mask, column] = period[column]
    out["alignment_status"] = np.where(out["period_id"].notna(), "assigned", "outside_defined_periods")
    return out


def _daily(frame):
    fields = [c for c in (*DAILY_FEATURES, "rhr_avg_review", "sleep_score_review", "rhr_session_count", "sleep_session_count") if c in frame]
    keys = ["participant_key", "date"]
    if frame.groupby(keys)[fields].nunique(dropna=False).gt(1).any().any():
        _fail("inconsistent_daily_values", "Repeated daily values/review masks are inconsistent")
    daily = frame[keys + fields].drop_duplicates(keys).copy()
    daily["local_date"] = pd.to_datetime(daily["date"])
    return daily


def _review(frame, feature):
    col = "sleep_score_review" if feature == "sleep_session_hours" else f"{feature}_review"
    if col not in frame:
        if feature in HRV_FIELDS:
            return pd.Series(False, index=frame.index)
        _fail("review_mask_missing", f"Missing review mask for {feature}")
    if frame[col].isna().any() or not pd.api.types.is_bool_dtype(frame[col]):
        _fail("invalid_review_mask", f"Require explicit boolean review mask: {col}")
    return frame[col]


def summarize(windows, blocks, daily, periods, features, minutes):
    """Means of source measurements/days; medians of distinct HRV blocks."""
    period_rows, feature_rows = [], []
    for period in periods:
        def selected(frame):
            return frame.loc[frame["participant_key"].eq(period["participant_key"]) & frame["period_id"].eq(period["period_id"])]
        win, hrv, days = selected(windows), selected(blocks), selected(daily)
        seconds = period.get("included_wall_clock_seconds", (pd.Timestamp(period["end_exclusive"]) - pd.Timestamp(period["start"])).total_seconds())
        coverage = pd.to_numeric(win["health_coverage_pct"], errors="raise")
        if (~np.isfinite(coverage) | coverage.lt(0) | coverage.gt(100 + 1e-8)).any():
            _fail("invalid_coverage", "Window health coverage must be finite in [0,100]")
        observed = float((coverage / 100 * minutes * 60).sum())
        wearing = pd.to_numeric(win["wearing_time_pct_observed"], errors="raise")
        if (coverage.gt(0) & (~np.isfinite(wearing) | wearing.lt(0) | wearing.gt(100))).any():
            _fail("invalid_wear", "Observed intervals require wearing percentage in [0,100]")
        worn = float((coverage / 100 * minutes * 60 * wearing.fillna(0) / 100).sum())
        expected = int(seconds / (minutes * 60))
        period_rows.append({**period, "expected_windows": expected, "available_windows": len(win),
                            "missing_windows": expected - len(win), "days_in_period": int(seconds / 86400),
                            "available_days": len(days), "available_hrv_blocks": len(hrv),
                            "calendar_wall_clock_seconds": seconds,
                            "observed_health_seconds": observed, "estimated_worn_seconds": worn,
                            "health_coverage_pct": observed / seconds * 100,
                            "wearing_time_pct_observed": worn / observed * 100 if observed else np.nan,
                            "coverage_status": "no_observed_health" if not observed else "partial_artifact" if len(win) < expected else "complete_grid",
                            "ready_for_features": False})
        for feature in features:
            units = win if feature in SAMPLE_FEATURES else days if feature in DAILY_FEATURES else hrv
            values = pd.to_numeric(units[feature], errors="raise")
            finite, review = np.isfinite(values), _review(units, feature)
            usable = finite & ~review
            weights = None
            aggregation, support = ("source_observation_weighted_mean", "window") if feature in SAMPLE_FEATURES else ("mean", "local_day") if feature in DAILY_FEATURES else ("median", "six_hour_block")
            if feature in SAMPLE_FEATURES:
                weights = pd.to_numeric(units[f"{feature}_n"], errors="raise")
                if (~np.isfinite(weights) | weights.lt(0) | weights.ne(np.floor(weights)) | (finite & weights.eq(0))).any():
                    _fail("invalid_observation_counts", f"Invalid source counts for {feature}")
                usable &= weights.gt(0)
            value = (float(np.average(values[usable], weights=weights[usable])) if weights is not None else
                     float(values[usable].median() if aggregation == "median" else values[usable].mean())) if usable.any() else np.nan
            feature_rows.append({"participant_key": period["participant_key"], "period_id": period["period_id"],
                                 "timepoint": period["timepoint"], "protocol_day": period["protocol_day"],
                                 "feature": feature, "value": value, "aggregation": aggregation, "support_unit": support,
                                 "available_units": len(units), "finite_units": int(finite.sum()),
                                 "review_excluded_units": int((finite & review).sum()), "used_units": int(usable.sum()),
                                 "used_source_observations": int(weights[usable].sum()) if weights is not None else None,
                                 "status": "ok" if usable.any() else "review_excluded" if (finite & review).any() else "no_finite_support"})
    return pd.DataFrame(period_rows), pd.DataFrame(feature_rows)


def _timepoint_summaries(windows, blocks, daily, periods, features, minutes):
    """Combine a union of explicit periods; gaps between them add no exposure."""
    groups = {}
    for period in periods:
        groups.setdefault((period["participant_key"], period["timepoint"]), []).append(period)
    combined = []
    for (key, label), members in groups.items():
        combined.append({"participant_key": key, "period_id": label, "timepoint": label, "protocol_day": None,
                         "start": min(p["start"] for p in members), "end_exclusive": max(p["end_exclusive"] for p in members),
                         "definition_source": "union of explicit nonoverlapping member periods",
                         "clinical_dates_json": members[0]["clinical_dates_json"],
                         "member_periods_json": json.dumps([{k: p[k] for k in ("period_id", "start", "end_exclusive", "protocol_day")} for p in members]),
                         "included_wall_clock_seconds": sum((date.fromisoformat(p["end_exclusive"]) - date.fromisoformat(p["start"])).days * 86400 for p in members)})
    tables = []
    for frame in (windows, blocks, daily):
        frame = frame.copy()
        frame["period_id"] = frame["timepoint"]
        tables.append(frame)
    coverage, values = summarize(*tables, combined, features, minutes)
    coverage = coverage.drop(columns=["period_id", "protocol_day"]).rename(columns={"start": "first_included_date", "end_exclusive": "end_last_included_date"})
    values = values.drop(columns=["period_id", "protocol_day"])
    return coverage, values


def _xsens_links(periods, readers):
    rows = []
    fields = ("recording_id", "source_id", "session_id", "timepoint", "protocol_day", "acquisition_date", "task", "anatomical_side", "limb_condition", "formatter_status")
    for period in periods:
        records = readers[period["participant_key"]].select(modality="xsens", timepoints=[period["timepoint"]], tasks=[1, 2, 3, 4])
        for record in records:
            if record.get("representation") != "full_recording" or (period["protocol_day"] is not None and record.get("protocol_day") != period["protocol_day"]):
                continue
            rows.append({"participant_key": period["participant_key"], "period_id": period["period_id"],
                         "association_rule": "exact_canonical_timepoint_and_optional_protocol_day",
                         **{f: record.get(f) for f in fields}})
    return pd.DataFrame(rows, columns=["participant_key", "period_id", "association_rule", *fields])


def _xsens_timepoint_links(periods, readers):
    by_label = {}
    for period in periods:
        by_label.setdefault((period["participant_key"], period["timepoint"]), []).append(period)
    union = [{**members[0], "period_id": label, "protocol_day": None}
             for (key, label), members in by_label.items()]
    links = _xsens_links(union, readers).drop(columns="period_id")
    statuses = []
    for row in links.to_dict("records"):
        members = by_label[(row["participant_key"], row["timepoint"])]
        value = row["acquisition_date"]
        statuses.append("acquisition_date_unresolved" if pd.isna(value) or not value else
                        "acquisition_date_in_defined_days" if any(p["start"] <= value < p["end_exclusive"] for p in members) else
                        "acquisition_date_outside_defined_days")
    links["date_alignment_status"] = statuses
    return links


def align_periods(subject_paths, window_paths, period_path, output, *, progress=None):
    if progress:
        progress.update("Validating period definitions and loading window artifacts")
    readers = _readers(subject_paths)
    spec, spec_hash = document(Path(period_path))
    periods = validate_periods(spec, readers)
    frame, hrv, features, method, inputs = load_windows(window_paths, readers)
    output = validate_output(Path(output), list(readers.values()))
    if Path(period_path).resolve().is_relative_to(output):
        _fail("unsafe_output", "Period-definition input must stay outside the alignment output directory")
    for root in [Path(i["root"]) for i in inputs]:
        if output.is_relative_to(root) or root.is_relative_to(output):
            _fail("unsafe_output", "Alignment output must be separate from input artifact directories")
    if progress:
        progress.update(f"Aligning {len(frame)} windows to {len(periods)} explicit period(s)")
    daily = _assign(_daily(frame), periods, "local_date")
    windows, blocks = _assign(frame, periods, "window_start"), _assign(hrv, periods, "hrv_block_start")
    if progress:
        progress.update("Calculating period/timepoint summaries and Xsens links")
    summary, values = summarize(windows, blocks, daily, periods, features, method["config"]["window_minutes"])
    timepoints, timepoint_values = _timepoint_summaries(windows, blocks, daily, periods, features, method["config"]["window_minutes"])
    links = _xsens_links(periods, readers)
    timepoint_links = _xsens_timepoint_links(periods, readers)
    for reader in readers.values():
        for name, digest in reader.metadata_sha256.items():
            if hashlib.sha256((reader.root / name).read_bytes()).hexdigest() != digest:
                _fail("metadata_changed_during_read", f"Changed {reader.participant_key}/{name}")
    if hashlib.sha256(Path(period_path).read_bytes()).hexdigest() != spec_hash:
        _fail("metadata_changed_during_read", "Period definitions changed during alignment")
    output.mkdir(parents=True, exist_ok=True)
    tables = {"aligned_windows": windows, "aligned_hrv_blocks": blocks, "aligned_daily": daily,
              "periods": summary, "period_features": values, "timepoints": timepoints,
              "timepoint_features": timepoint_values, "xsens_links": links, "xsens_timepoint_links": timepoint_links}
    for index, (name, table) in enumerate(tables.items(), 1):
        if progress:
            progress.update(f"Writing alignment table {name}", completed=index - 1, total=len(tables))
        write_frame(output / f"{name}.csv", table)
        write_frame(output / f"{name}.parquet", table)
        if progress:
            progress.update(f"Alignment table written: {name}", completed=index, total=len(tables))
    report = {"method_id": ALIGNMENT_METHOD, "created_at": now(), "period_definition_sha256": spec_hash,
              "period_definition_path": str(Path(period_path).resolve()), "periods": periods,
              "window_method": method, "inputs": inputs, "feature_columns": features,
              "time_basis": "source_local_wall_clock", "boundary_rule": "explicit full local days [start,end_exclusive)",
              "sample_summary": "source-count weighted mean; exclude whole reviewed windows; steps/calories remain means, not totals",
              "daily_summary": "one consistent value per local date; sleep uses local end date and retains UTC duration",
              "hrv_summary": "median of distinct six-hour estimates; no pooling of beats, epochs or percentages",
              "xsens_association": "exact canonical timepoint/day; does not assert simultaneous acquisition",
              "overlap_policy": "reject", "missing_policy": "no interpolation; incomplete grids retained with counts; no wear/coverage acceptance threshold assumed",
              "timepoint_summary": "union of explicitly defined member periods; intervening gaps contribute no days or observations",
              "unassigned_windows": int(windows["period_id"].isna().sum()),
              "review_required": bool(summary["missing_windows"].gt(0).any() or summary["health_coverage_pct"].lt(100).any()
                                      or values["review_excluded_units"].gt(0).any() or values["used_units"].eq(0).any()
                                      or windows["period_id"].isna().any()),
              "ready_for_features": False, "scientifically_validated": False,
              "subject_metadata_sha256": {key: r.metadata_sha256 for key, r in readers.items()},
              "implementation_sha256": implementation_hashes(("garmin_code/alignment.py", "garmin_code/__main__.py", "shared_code/reader.py", "shared_code/progress.py")),
              "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.stem in tables and p.suffix in {".csv", ".parquet"}}}
    atomic_json(output / "report.json", report)
    atomic_json(output / "period_definitions.json", spec)
    lines = ["# Garmin monitoring-period alignment", "", f"Method: `{ALIGNMENT_METHOD}`.", "",
             "Explicit source-local calendar days use [start,end_exclusive). Period definitions and input artifacts are checksummed. Clinical dates are evidence, not inferred boundaries.", "",
             "| Participant | Period | Timepoint | Start | End (exclusive) | Windows / expected | Health coverage |",
             "|---|---|---|---|---|---:|---:|"]
    for row in summary.to_dict("records"):
        lines.append(f"| {row['participant_key']} | {row['period_id']} | {row['timepoint']} | {row['start']} | {row['end_exclusive']} | {row['available_windows']} / {row['expected_windows']} | {row['health_coverage_pct']:.2f}% |")
    lines += ["", "[Period features](period_features.csv) gives source-observation weighted means, daily means counted once per date, and medians counted once per six-hour HRV block. Reviewed values are excluded; reviewed sample windows are excluded in full. Steps/calories remain means. No pooled HRV or period totals are claimed.", "",
              "[Timepoint features](timepoint_features.csv) combines the union of explicit periods sharing a canonical timepoint. [Timepoint coverage](timepoints.csv) records each member interval; gaps between member intervals add no exposure. These are descriptive observed-data summaries.", "",
              "[Aligned windows](aligned_windows.csv), [daily evidence](aligned_daily.csv), [HRV blocks](aligned_hrv_blocks.csv), [coverage](periods.csv), [day-specific Xsens links](xsens_links.csv), [Xsens timepoint links and date status](xsens_timepoint_links.csv), [provenance](report.json).", "",
              "Missing windows, null features, review exclusions and unassigned observations remain visible. Health/wear exposure uses the existing local-wall-clock method; DST transition days require review. No coverage acceptance threshold or scientific readiness is inferred. Xsens links use canonical labels and do not establish simultaneous recording.", "",
              f"Unassigned windows: {report['unassigned_windows']}. Review required: {report['review_required']}. ready_for_features=false.", ""]
    atomic_text(output / "report.md", "\n".join(lines))
    if progress:
        progress.update(f"Alignment reports written: {output / 'report.md'}")
    return summary, values, report


def clinical_day_periods(paths):
    """T0–T10 on exact imported clinical days; no fixed-length visit windows."""
    participants, plan = {}, []
    for key, reader in _readers(paths).items():
        clinical = reader.subject.get("clinical_metadata", {})
        if reader.subject["cohort"] != "PAZIENTI" or clinical.get("status") != "available":
            _fail("clinical_dates_unavailable", f"Exact-day T0–T10 alignment requires available patient clinical metadata: {key}")
        _, timeline = _labels(reader)
        items = []
        for n in range(11):
            label = f"T{n}"
            dates = {k: v for k, v in timeline.items() if k == label or re.fullmatch(label + r"_D[1-5]", k)}
            for source_label, value in sorted(dates.items()):
                try:
                    day = date.fromisoformat(value)
                except (ValueError, TypeError) as exc:
                    raise DataContractError("invalid_clinical_date", f"Invalid {key}/{source_label}") from exc
                items.append({"period_id": source_label, "timepoint": label,
                              "protocol_day": int(source_label.rsplit("_D", 1)[1]) if "_D" in source_label else None,
                              "start": day.isoformat(), "end_exclusive": (day + timedelta(days=1)).isoformat(),
                              "definition_source": "clinical-exact-local-day-v1: subject.json clinical_metadata.timeline_dates; full recorded day only"})
            plan.append({"participant_key": key, "timepoint": label, "dated_clinical_days": len(dates),
                         "clinical_dates_json": json.dumps(dates, sort_keys=True),
                         "date_status": "dated" if dates else "no_recorded_clinical_date"})
        if not items:
            _fail("clinical_dates_unavailable", f"No recorded T0–T10 dates for {key}")
        participants[key] = items
    spec = {"format_version": 1, "time_basis": "source_local_wall_clock", "participants": participants}
    validate_periods(spec, _readers(paths))
    return spec, pd.DataFrame(plan)


def run_clinical_alignment(subject_paths, output, config=None, *, max_days=64, progress=None):
    """Bounded daily extraction and exact-date alignment for supplied patients."""
    from .garmin import GarminConfig, GarminSubject
    from .garmin_workflow import run_windows

    if progress:
        progress.update("Resolving recorded clinical days and Garmin exports")
    readers = _readers(subject_paths)
    output = validate_output(Path(output), list(readers.values()))
    spec, plan = clinical_day_periods(subject_paths)
    count = sum(len(items) for items in spec["participants"].values())
    if type(max_days) is not int or max_days < 1 or count > max_days:
        _fail("too_many_clinical_days", f"{count} clinical days exceeds limit {max_days}; select fewer patients or increase --max-clinical-days")
    config = config or GarminConfig()
    # Resolve input exports before any output: missing/unconverted Garmin cannot
    # silently become a scientifically meaningful all-null participant.
    for key, items in spec["participants"].items():
        GarminSubject(readers[key].root, items[0]["start"], items[0]["end_exclusive"], config)
    output.mkdir(parents=True, exist_ok=True)
    period_path = output / "clinical_day_periods.json"
    atomic_json(period_path, spec)
    paths = []
    for key, items in spec["participants"].items():
        for item in sorted(items, key=lambda p: p["start"]):
            folder = output / "windows" / hashlib.sha256(key.encode()).hexdigest()[:12] / item["start"]
            if progress:
                progress.update(f"Starting clinical day {len(paths)+1}/{count}: {key} {item['period_id']} {item['start']}",
                                completed=len(paths), total=count)
            run_windows([GarminSubject(readers[key].root, item["start"], item["end_exclusive"], config)], folder,
                        progress=progress)
            paths.append(folder)
            if progress:
                progress.update(f"Clinical day processed: {key} {item['start']}", completed=len(paths), total=count)
    summary, values, report = align_periods(subject_paths, paths, period_path, output / "aligned", progress=progress)
    tp = pd.read_parquet(output / "aligned" / "timepoints.parquet")
    if progress:
        progress.update(f"Writing clinical alignment plan and reports to {output}")
    plan = plan.merge(tp[["participant_key", "timepoint", "health_coverage_pct", "available_windows"]], on=["participant_key", "timepoint"], how="left", validate="one_to_one")
    write_frame(output / "clinical_alignment_plan.csv", plan)
    atomic_json(output / "report.json", {"method_id": "clinical-exact-local-day-v1", "clinical_days": count,
                                        "alignment_report": "aligned/report.json", "date_rule": "actual imported dates only; T1–T4 retain all recorded D1–D5 days",
                                        "clinical_plan_sha256": hashlib.sha256((output / "clinical_alignment_plan.csv").read_bytes()).hexdigest(),
                                        "review_required": report["review_required"], "ready_for_features": False})
    lines = ["# Garmin T0–T10 exact clinical-day alignment", "",
             "Rule: `clinical-exact-local-day-v1`. Each imported assessment/training date selects that complete source-local day. T1–T4 combine only their actual D1–D5 dates; gaps between them add no observations. No seven-day or between-visit monitoring interval is inferred.", "",
             f"Selected clinical days: {count}. [Alignment and coverage](aligned/report.md), [timepoint features](aligned/timepoint_features.csv), [T0–T10 date plan](clinical_alignment_plan.csv), [definitions](clinical_day_periods.json).", "",
             "Missing clinical dates remain visible in the plan. Missing Garmin observations remain null with coverage/count evidence. Exact-day association is a descriptive comparison rule, not confirmation of the patient's full monitoring protocol. Numerical/scientific readiness remains false.", ""]
    atomic_text(output / "report.md", "\n".join(lines))
    return summary, values, report
