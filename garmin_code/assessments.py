"""Assessment-centered Garmin features with explicit source and day support.

Clinical sources are evidence, never correction targets. T0 is exploratory only.
The existing six-hour extractor owns stream eligibility and numerical HRV.
"""
from __future__ import annotations

from datetime import date, timedelta
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from predictor_import.clinical import extract_workbook
from shared_code.artifacts import now, validate_output
from shared_code.io import atomic_json
from shared_code.reader import DataContractError, SubjectReader, document
from shared_code.provenance import implementation_hashes
from .garmin import GarminConfig, GarminSubject, build_windows, numeric
from .garmin_workflow import method_document

METHOD_ID = "garmin-xsens-centered-five-day-equal-day-t1-first-worn-v3"
CYCLES = ("night", "morning", "afternoon", "evening")
MIN_WORN_PCT, MIN_DAYS = 50.0, 3  # protocol: >=50% wear per six-hour cycle; >=3 valid days per T
BLOCK_FEATURES = {
    "rmssd": "RMSSD", "sdnn": "SDNN", "lf_hf_ratio": "LF_HF_ratio",
    "sd2_sd1_ratio": "SD2_SD1_ratio", "sample_entropy": "SampEn",
    "steps": "steps_mean", "met": "met_avg_mean",
    "total_calories_kcal": "total_calories_kcal_mean",
    "respiration_breaths_min": "respiration_breaths_min_mean",
    "physiological_stress": "stress_level_value_mean",
}
DAILY_FEATURES = {"rhr_bpm": "rhr_avg", "sleep_quality": "sleep_score", "sleep_session_hours": "sleep_session_hours"}
# Fitrockr's own 5-minute HRV averaged per block; Fitrockr provides no sample entropy.
FITROCKR_FEATURES = {**BLOCK_FEATURES, "rmssd": "rmssd_mean", "sdnn": "sdnn_mean", "lf_hf_ratio": "lf_hf_mean",
                     "sd2_sd1_ratio": "sd2_sd1_mean", "sample_entropy": None}
UNITS = {"rmssd": "ms", "sdnn": "ms", "lf_hf_ratio": "ratio", "sd2_sd1_ratio": "ratio",
         "sample_entropy": "dimensionless", "steps": "steps per source interval", "met": "MET",
         "total_calories_kcal": "kcal per source interval", "respiration_breaths_min": "breaths/min",
         "physiological_stress": "source scale 0-100", "rhr_bpm": "bpm", "sleep_quality": "source score 0-100",
         "sleep_session_hours": "h including awake time"}


def fail(code, message):
    raise DataContractError(code, message)


def load_overrides(path, readers):
    """Anchor overrides plus optional explicit windows: {"overrides": [...], "windows": [...]}."""
    if path is None:
        return {}, None
    spec, digest = document(Path(path))
    if not spec or not set(spec) <= {"overrides", "windows"} or not all(isinstance(v, list) for v in spec.values()):
        fail("invalid_anchor_override", "Expected overrides and/or windows lists")
    result = {}
    for item in spec.get("windows", []):
        if set(item) != {"participant_key", "timepoint", "start", "end", "reason"} or item["participant_key"] not in readers or not item["reason"].strip():
            fail("invalid_window_override", "Window requires an explicitly selected identity, start, end and reason")
        start, end = date.fromisoformat(item["start"]), date.fromisoformat(item["end"])
        if item["timepoint"] not in {f"T{i}" for i in range(1, 11)} or start.isoformat() != item["start"] or end.isoformat() != item["end"] or not 0 <= (end - start).days < 7:
            fail("invalid_window_override", "Window needs T1-T10 and canonical dates spanning 1-7 days")
        key = item["participant_key"], item["timepoint"], "window"
        if key in result:
            fail("duplicate_window_override", "Repeated participant/timepoint window")
        result[key] = item
    for item in spec.get("overrides", []):
        required = {"participant_key", "timepoint", "source_timepoint", "date", "reason"}
        if set(item) != required or item["participant_key"] not in readers or not item["reason"].strip():
            fail("invalid_anchor_override", "Override requires an explicitly selected identity, source/date and reason")
        if item["timepoint"] not in {f"T{i}" for i in range(1, 11)}:
            fail("invalid_anchor_override", "Only T1-T10 anchor overrides are supported")
        if date.fromisoformat(item["date"]).isoformat() != item["date"]:
            fail("invalid_anchor_override", "Override date must be canonical ISO")
        key = item["participant_key"], item["timepoint"]
        if key in result:
            fail("duplicate_anchor_override", "Repeated participant/timepoint override")
        result[key] = item
    return result, {"path": str(Path(path).resolve()), "sha256": digest}


def first_worn_day(root, start, end, config):
    """First worn health interval in [start, end) and the first local day wholly after it."""
    try:
        health = GarminSubject(root, start.isoformat(), end.isoformat(), config).prepare("health").data
    except DataContractError as exc:
        if exc.code == "empty_garmin_selection":
            return None, None
        raise
    worn = health.loc[numeric(health["wearing_time"]).gt(0), "_local"].dropna()
    if worn.empty:
        return None, None
    first = worn.min()
    return first, first.date() if first == first.normalize() else first.date()+timedelta(days=1)


def t1_window(root, anchor, t2_anchor, config):
    """The watch is handed over at T0 and paired days later, so Xsens day -2 often precedes any data.

    T1 keeps up to five days, starting at the later of Xsens day -2 and the first wholly worn day,
    and ending before T2's Xsens day. None when this is the standard window.
    """
    day = date.fromisoformat(anchor)
    t2 = date.fromisoformat(t2_anchor) if t2_anchor else None
    limit = t2 if t2 and t2 > day else day+timedelta(days=7)
    first, full = first_worn_day(root, day-timedelta(days=3), limit, config)
    start = max(day-timedelta(days=2), full) if full else day-timedelta(days=2)
    end = min(start+timedelta(days=4), limit-timedelta(days=1))
    if start == day-timedelta(days=2) and end == day+timedelta(days=2):
        return None
    return {"rule": "t1_first_worn_day", "start": start.isoformat(), "end": end.isoformat(),
            "first_worn_local": first.isoformat() if first is not None else None,
            "first_full_worn_day": full.isoformat() if full else None,
            "t2_anchor_date": t2_anchor,
            "reason": "Xsens day -2 to +2, moved to start on the first wholly worn day and stopped before T2's Xsens day"}


def assessment_plan(reader, overrides, config=None):
    events, sources, issues = [], [], []
    for record in reader.select(modality="clinical"):
        _, calendar, found, source = extract_workbook(reader, record)
        events.extend(calendar)
        sources.append(source)
        issues.extend(found)
    if any(i["code"] == "clinical_metadata_date_disagrees" for i in issues):
        fail("clinical_metadata_date_disagrees", "Clinical workbook and imported dates disagree")
    plans = []
    for n in range(11):
        label = f"T{n}"
        override = overrides.get((reader.participant_key, label))
        window = overrides.get((reader.participant_key, label, "window"))
        source_label = override["source_timepoint"] if override else label
        matches = [e for e in events if e.get("timepoint") == source_label and (
            e.get("event_kind") == "clinical_day" if n == 0 else
            e.get("event_kind") == "protocol_activity" and "xsens" in e.get("measure", "").lower()
            and str(json.loads(e["raw_value_json"])).strip().lower() == "x")]
        dates = {e["clinical_date"] for e in matches}
        anchor = next(iter(dates)) if len(dates) == 1 and None not in dates else None
        if override and anchor != override["date"]:
            fail("anchor_override_source_disagrees", "Override must match its original X-marked Xsens source date")
        day = date.fromisoformat(anchor) if anchor else None
        if window and day:
            first = date.fromisoformat(window["start"])
            dates = [(first+timedelta(days=i)).isoformat() for i in range((date.fromisoformat(window["end"])-first).days+1)]
        else:
            dates = [(day+timedelta(days=i)).isoformat() for i in range(-2, 3)] if day else []
        plans.append({"participant_key": reader.participant_key, "timepoint": label,
                      "anchor_date": anchor, "date_status": "anchored" if anchor else "unresolved_xsens_date",
                      "anchor_basis": "explicit_user_override" if override else "clinical_T0_date" if n == 0 else "clinical_xsens_activity_mark",
                      "override": override, "window": window if window and day else None, "dates": dates,
                      "window_basis": "explicit_window" if window and day else "anchor_plus_minus_2",
                      "source_cells": [{k: e.get(k) for k in ("source_path", "source_sha256", "sheet", "source_cell", "source_label", "clinical_date", "raw_value_json")} for e in matches]})
    t1 = plans[1]
    if t1["anchor_date"] and not t1["window"]:
        rule = t1_window(reader.root, t1["anchor_date"], plans[2]["anchor_date"], config or GarminConfig(window_minutes=360))
        if rule:
            first = date.fromisoformat(rule["start"])
            t1.update(window=rule, window_basis=rule["rule"],
                      dates=[(first+timedelta(days=i)).isoformat() for i in range((date.fromisoformat(rule["end"])-first).days+1)])
    return plans, sources, issues


def empty_blocks(plan):
    return pd.DataFrame([{
        "participant_key": plan["participant_key"], "timepoint": plan["timepoint"],
        "anchor_date": plan["anchor_date"], "date_status": plan["date_status"],
        "relative_day": day, "date": None, "circadian_cycle": cycle, "cycle": CYCLES[cycle],
        **{c: np.nan for c in (*BLOCK_FEATURES.values(), *DAILY_FEATURES.values())},
    } for day in range(-2, 3) for cycle in range(4)])


def daily_table(blocks):
    """Count the repeated daily fields once, validating consistency first."""
    rows = []
    for (relative_day,), group in blocks.groupby(["relative_day"], sort=True):
        row = {c: group.iloc[0][c] for c in ("participant_key", "timepoint", "anchor_date", "date_status", "date")}
        row["relative_day"] = relative_day
        for source in DAILY_FEATURES.values():
            values = pd.to_numeric(group[source], errors="coerce").dropna().unique()
            if len(values) > 1:
                fail("inconsistent_daily_values", f"Daily {source} differs across repeated six-hour rows")
            row[source] = values[0] if len(values) else np.nan
            review = f"{source}_review"
            row[review] = bool(group[review].fillna(False).any()) if review in group else False
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(plan, blocks, daily, variant, block_features=BLOCK_FEATURES):
    """Mean of finite daily features; advisory reviews remain with counts."""
    result = {c: plan[c] for c in ("participant_key", "timepoint", "anchor_date", "date_status", "anchor_basis")}
    support = []
    for cycle, features, frame in [(name, block_features, blocks.loc[blocks.cycle.eq(name)]) for name in CYCLES] + [("daily", DAILY_FEATURES, daily)]:
        for feature, source in features.items():
            if source not in frame:
                frame = frame.assign(**{str(source): np.nan})
            values = pd.to_numeric(frame[str(source)], errors="coerce").replace([np.inf, -np.inf], np.nan)
            usable = values.notna()
            count = int(usable.sum()) if plan["anchor_date"] else None
            col = f"avg_{feature}" if cycle == "daily" else f"avg_{cycle}_{feature}"
            enough = plan["anchor_date"] and count >= MIN_DAYS
            result[col] = float(values.mean()) if enough else None
            recomputed = frame.get(f"{source}_recomputed_n", pd.Series(0, index=frame.index)).fillna(0).gt(0)
            review = frame.get(f"{source}_review", pd.Series(False, index=frame.index)).fillna(False)
            support.append({"participant_key": plan["participant_key"], "timepoint": plan["timepoint"], "variant": variant,
                            "cycle": cycle, "feature": feature, "summary_column": col, "unit": UNITS[feature],
                            "mean": result[col], "contributing_days": count,
                            "relative_days_json": json.dumps(frame.loc[usable, "relative_day"].astype(int).tolist()),
                            "advisory_review_days": int((usable & review).sum()) if count is not None else None,
                            "recomputed_days": int((usable & recomputed).sum()) if count is not None else None,
                            "status": "unresolved_anchor" if count is None else "missing" if count == 0 else "below_minimum_days" if count < MIN_DAYS else "available"})
    result["cells_below_three_days"] = sum(s["contributing_days"] < MIN_DAYS for s in support) if plan["anchor_date"] else None
    result["advisory_review_cells"] = sum(bool(s["advisory_review_days"]) for s in support) if plan["anchor_date"] else None
    return result, support


def compare_t0_t1(plans, summaries, support):
    p0, p1 = plans[0], plans[1]
    rows = []
    for col in summaries["T1"]:
        if not col.startswith("avg_"):
            continue
        a, b = summaries["T0"][col], summaries["T1"][col]
        rows.append({"participant_key": p1["participant_key"], "feature": col,
                     "T0_anchor": p0["anchor_date"], "T1_anchor": p1["anchor_date"],
                     "T0_mean": a, "T1_mean": b, "T1_minus_T0": b-a if a is not None and b is not None else None,
                     "T0_days": support["T0"][col]["contributing_days"], "T1_days": support["T1"][col]["contributing_days"],
                     "shared_calendar_days": len(set(p0["dates"]) & set(p1["dates"])) if p0["dates"] and p1["dates"] else None,
                     "interpretation": "same anchor after explicit mapping; not independent assessments" if p0["anchor_date"] and p0["anchor_date"] == p1["anchor_date"] else "exploratory overlapping five-day windows; no independent baseline inference"})
    return rows


def extract_assessment(root, plan, config, progress):
    variant = "raw"
    if not plan["anchor_date"] or not plan["dates"]:
        blocks = empty_blocks(plan)
        return blocks, daily_table(blocks), pd.DataFrame(), None
    first, last = plan["dates"][0], (date.fromisoformat(plan["dates"][-1])+timedelta(days=1)).isoformat()
    run = GarminSubject(root, first, last, config)
    if progress:
        progress.update(f"{run.participant_key} {plan['timepoint']} {variant}: {first} to {last}")
    frame, _, _ = build_windows(run, progress=progress)
    required = {*BLOCK_FEATURES.values(), *DAILY_FEATURES.values()}
    if not required <= set(frame):
        fail("selected_feature_missing", f"Required source semantics missing: {sorted(required-set(frame))}")
    frame["timepoint"], frame["anchor_date"], frame["date_status"] = plan["timepoint"], plan["anchor_date"], plan["date_status"]
    frame["relative_day"] = (pd.to_datetime(frame.date)-pd.Timestamp(plan["anchor_date"])).dt.days
    frame["cycle"] = frame.circadian_cycle.map(dict(enumerate(CYCLES)))
    assert len(frame) == 4*len(plan["dates"])
    keep = ["participant_key", "timepoint", "anchor_date", "date_status", "relative_day", "date", "cycle", "circadian_cycle", "window_start", "window_end"]
    keep += [c for c in frame if c in required or c in FITROCKR_FEATURES.values() or c.endswith(("_n", "_review", "_n_epochs", "_status")) or c in {
        "n_beats", "n_adjacent_pairs", "health_coverage_pct", "wearing_time_pct_observed", "hrv_segments", "hrv_longest_segment_s", "hrv_pairs_excluded", "hrv_clock_mismatch_pairs"}]
    blocks = frame[list(dict.fromkeys(keep))].copy()
    blocks["worn_pct"] = blocks.health_coverage_pct * blocks.wearing_time_pct_observed.fillna(0) / 100
    thin = blocks.worn_pct.lt(MIN_WORN_PCT)
    for source in (*BLOCK_FEATURES.values(), *filter(None, FITROCKR_FEATURES.values())):
        if source in blocks:
            blocks.loc[thin, source] = np.nan
    epochs = run.hrv_epochs.copy()
    epochs.insert(1, "timepoint", plan["timepoint"])
    epochs.insert(2, "variant", variant)
    report = run.report()
    report["timepoint"], report["variant"] = plan["timepoint"], variant
    daily = daily_table(blocks)
    del run, frame
    gc.collect()
    return blocks, daily, epochs, report


def run_assessments(subject_paths, output, *, override_path=None, config=None, progress=None, xlsx_writer=None):
    config = config or GarminConfig(window_minutes=360)
    if config.window_minutes != 360:
        fail("assessment_config", "Assessment method requires six-hour windows")
    readers = {r.participant_key: r for r in (SubjectReader(p) for p in subject_paths)}
    if not readers or len(readers) != len(subject_paths):
        fail("duplicate_or_empty_subjects", "Supply distinct explicit subjects")
    output = validate_output(Path(output), list(readers.values()))
    if output.exists():
        fail("output_exists", "Assessment output must be a fresh directory")
    overrides, override_source = load_overrides(override_path, readers)
    if override_source and Path(override_source["path"]).is_relative_to(output):
        fail("unsafe_output", "Anchor override input must be outside the output directory")
    all_plans = {}
    for key, reader in readers.items():
        all_plans[key] = assessment_plan(reader, overrides, config)
    # Complete plan validation before reserving a fresh output or calculating.
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"method_id": METHOD_ID, "started_at": now(), "state": "running", "anchor_override_source": override_source,
                "subjects": {}, "output_sha256": {}}
    atomic_json(output/"metadata.json", metadata)
    method = {"assessment_method_id": METHOD_ID, "primary": method_document(config),
              "hrv_sources": {"Summary": "all HRV from raw BBI",
                              "Summary_Fitrockr_HRV": "RMSSD, SDNN, LF/HF, SD2/SD1 = means of Fitrockr 5-min detailed-health-log values (SD2/SD1 = 1/SD1_SD2); no SampEn; other features identical"},
              "aggregation": "T1-T10; Xsens date +/-2 local calendar days unless an explicit window is given; T1 starts no earlier than the first wholly worn day and ends before T2's Xsens day; six-hour block-days need >=50% wear; equal mean of valid daily values; RHR/sleep once per date; fewer than 3 valid days -> blank",
              "calories": "total calories source interval means; existing high-rate warning retained; upper-limit decision deferred",
              "T0": "separate exploratory clinical-date +/-2 window; shared days reported; excluded from primary T1-T10 table",
              "scientifically_validated": False,
              "implementation_sha256": implementation_hashes(("garmin_code/assessments.py", "garmin_code/assessment_export.py", "predictor_import/clinical.py"))}
    atomic_json(output/"method.json", method)
    from .assessment_export import export_assessment
    for key, reader in readers.items():
        plans, clinical_sources, clinical_issues = all_plans[key]
        summaries, fitrockr, by_support, support_rows, block_tables, daily_tables, epoch_tables, reports = {}, {}, {}, [], [], [], [], []
        for plan in plans:
            b, d, e, report = extract_assessment(reader.root, plan, config, progress)
            summary, supports = summarize(plan, b, d, "in_house")
            summaries[plan["timepoint"]] = summary
            fitrockr[plan["timepoint"]], _ = summarize(plan, b, d, "fitrockr_hrv", FITROCKR_FEATURES)
            by_support[plan["timepoint"]] = {s["summary_column"]: s for s in supports}
            if plan["timepoint"] != "T0":
                support_rows.extend(supports)
                block_tables.append(b)
                daily_tables.append(d)
                if len(e): epoch_tables.append(e)
            if report: reports.append(report)
        payload = {
            "Summary": pd.DataFrame([summaries[f"T{i}"] for i in range(1, 11)]),
            "Summary_Fitrockr_HRV": pd.DataFrame([fitrockr[f"T{i}"] for i in range(1, 11)]),
            "Blocks_20_per_T": pd.concat(block_tables, ignore_index=True),
            "Daily": pd.concat(daily_tables, ignore_index=True),
            "Support": pd.DataFrame(support_rows),
            "T0_T1": pd.DataFrame(compare_t0_t1(plans, summaries, by_support)),
            "HRV_epochs": pd.concat(epoch_tables, ignore_index=True) if epoch_tables else pd.DataFrame(columns=["participant_key", "timepoint", "variant"]),
        }
        destination = output/key.split("/")[1]/"Garmin_features.xlsx"
        verification = export_assessment(destination, payload, plans, reports, clinical_issues, method, writer=xlsx_writer)
        metadata["subjects"][key] = {"plans": plans, "clinical_sources": clinical_sources, "clinical_issues": clinical_issues,
                                     "runs": reports, "workbook": str(destination.relative_to(output)), "verification": verification}
        metadata["output_sha256"][str(destination.relative_to(output))] = verification["sha256"]
        atomic_json(output/"metadata.json", metadata)
        if progress: progress.update(f"Verified workbook: {destination}")
    if override_source and hashlib.sha256(Path(override_source["path"]).read_bytes()).hexdigest() != override_source["sha256"]:
        fail("anchor_override_changed", "Anchor override changed during extraction")
    for key, reader in readers.items():
        for name, digest in reader.metadata_sha256.items():
            if hashlib.sha256((reader.root/name).read_bytes()).hexdigest() != digest:
                fail("metadata_changed_during_read", f"Changed {key}/{name}")
        for source in all_plans[key][1]:
            if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
                fail("clinical_source_changed", "Clinical source changed during extraction")
    metadata.update(state="complete", finished_at=now())
    atomic_json(output/"metadata.json", metadata)
    return metadata
