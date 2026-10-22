"""Consequential method, date-mapping and source-to-XLSX regressions."""
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from garmin_code.assessments import (BLOCK_FEATURES, DAILY_FEATURES, FITROCKR_FEATURES, assessment_plan, load_overrides,
                                     daily_table, empty_blocks, summarize, run_assessments)
from garmin_code.garmin import GarminConfig, GarminSubject, build_windows
from garmin_code.tests.test_garmin import factory, times
from shared_code.reader import DataContractError, SubjectReader


def plan(anchor="2030-04-09"):
    return {"participant_key": "PAZIENTI/FG91", "timepoint": "T1", "anchor_date": anchor,
            "date_status": "anchored" if anchor else "unresolved_xsens_date", "anchor_basis": "clinical_xsens_activity_mark"}


def test_equal_day_weighting_and_three_day_minimum():
    p = plan()
    b = empty_blocks(p)
    for day, value in ((-2, 100.), (-1, 300.), (0, 200.)):
        b.loc[b.relative_day.eq(day), "steps_mean"] = value
        b.loc[b.relative_day.eq(day), "steps_mean_n"] = 100 if day == -2 else 1
    for day, value in ((-2, 60.), (-1, 80.)):
        b.loc[b.relative_day.eq(day), "rhr_avg"] = value
    s, support = summarize(p, b, daily_table(b), "in_house")
    assert s["avg_night_steps"] == 200  # equal day weights, not observation weights
    assert s["avg_rhr_bpm"] is None  # two days only
    rhr = next(r for r in support if r["feature"] == "rhr_bpm")
    assert rhr["contributing_days"] == 2 and rhr["status"] == "below_minimum_days"
    assert next(r for r in support if r["feature"] == "rmssd")["status"] == "missing"
    b.loc[b.index[1], "rhr_avg"] = 90
    with pytest.raises(DataContractError, match="differs across"):
        daily_table(b)


def test_undated_assessment_has_blank_counts_instead_of_zero_measured_days():
    p = plan(None)
    b = empty_blocks(p)
    s, support = summarize(p, b, daily_table(b), "in_house")
    assert len(b) == 20 and s["avg_night_rmssd"] is None
    assert all(r["contributing_days"] is None and r["status"] == "unresolved_anchor" for r in support)


def calendar(reader, _):
    event = {"timepoint": "T0", "event_kind": "clinical_day", "clinical_date": "2030-04-09", "source_label": "T0", "source_cell": "B5"}
    activity = {**event, "event_kind": "protocol_activity", "measure": "Task n4 Xsens", "raw_value_json": '"X"', "source_cell": "B33"}
    return [], [event, activity], [], {"path": str(reader.root/"subject.json"), "sha256": reader.metadata_sha256["subject.json"]}


def test_override_is_explicit_and_source_checked(factory, monkeypatch):
    reader = SubjectReader(factory())
    monkeypatch.setattr(reader, "select", lambda **_: [{}])
    monkeypatch.setattr("garmin_code.assessments.extract_workbook", calendar)
    plans, _, _ = assessment_plan(reader, {})
    assert plans[1]["anchor_date"] is None
    override = {"source_timepoint": "T0", "date": "2030-04-09", "reason": "explicit test authorization"}
    plans, _, _ = assessment_plan(reader, {(reader.participant_key, "T1"): override})
    assert plans[1]["anchor_date"] == "2030-04-09" and plans[1]["source_cells"][0]["source_label"] == "T0"
    override["date"] = "2030-04-10"
    with pytest.raises(DataContractError, match="original X-marked"):
        assessment_plan(reader, {(reader.participant_key, "T1"): override})


def test_complete_workflow_writes_verified_xlsx_and_refuses_existing_output(factory, monkeypatch, tmp_path):
    subject = factory()
    # A bounded synthetic calendar isolates the full numerical/export route;
    # clinical-workbook parsing and preservation are checked on the real run.
    reader = SubjectReader(subject)
    original = SubjectReader.select
    monkeypatch.setattr(SubjectReader, "select", lambda self, **kw: [{}] if kw.get("modality") == "clinical" else original(self, **kw))
    monkeypatch.setattr("garmin_code.assessments.extract_workbook", calendar)
    overrides = tmp_path/"anchors.json"
    overrides.write_text(json.dumps({"overrides": [{"participant_key": reader.participant_key, "timepoint": "T1", "source_timepoint": "T0", "date": "2030-04-09", "reason": "explicit test authorization"}]}))
    output = tmp_path/"fresh"
    result = run_assessments([subject], output, override_path=overrides)
    assert result["state"] == "complete"
    info = result["subjects"][reader.participant_key]
    assert info["verification"]["numerical_readback_passed"]
    w = load_workbook(output/info["workbook"], read_only=True, data_only=True)
    assert w["Summary"].max_row == 11 and w["Blocks_20_per_T"].max_row == 201
    assert w["Daily"].max_row == 51 and w["Summary_Fitrockr_HRV"].max_row == 11
    summary = list(w["Summary"].values)
    t1 = dict(zip(summary[0], summary[1]))
    assert t1["avg_night_steps"] is None and t1["avg_sleep_session_hours"] is None  # 2 of 3 required days
    rows = list(w["Support"].values)
    support = [dict(zip(rows[0], r)) for r in rows[1:]]
    steps = next(r for r in support if r["timepoint"] == "T1" and r["summary_column"] == "avg_night_steps")
    assert steps["contributing_days"] == 0 and steps["status"] == "missing"  # night worn ~30 min < 50%
    rows = list(w["Blocks_20_per_T"].values)
    blocks = [dict(zip(rows[0], r)) for r in rows[1:]]
    night = next(r for r in blocks if r["timepoint"] == "T1" and r["date"] and r["date"].date() == date(2030, 4, 9) and r["cycle"] == "night")
    assert night["worn_pct"] < 50 and night["steps_mean"] is None
    assert t1["anchor_date"].date() == date(2030, 4, 9)
    assert list(w["T0_T1"].values)[1][-2] == 3  # T1 starts on the first worn day, 9 April
    anchors = list(w["Anchors"].values)
    assert dict(zip(anchors[0], anchors[2]))["window_basis"] == "t1_first_worn_day"
    w.close()
    with pytest.raises(DataContractError, match="fresh directory"):
        run_assessments([subject], output, override_path=overrides)


def test_fitrockr_summary_has_no_entropy_and_raw_is_the_only_bbi_source(factory):
    b = empty_blocks(plan())
    b["rmssd_mean"], b["RMSSD"] = 70.0, 40.0
    s, _ = summarize(plan(), b, daily_table(b), "fitrockr_hrv", FITROCKR_FEATURES)
    assert s["avg_night_rmssd"] == 70 and s["avg_night_sample_entropy"] is None
    assert "bbi_source" not in GarminConfig().method()["config"]


def test_explicit_window_replaces_anchor_window(factory, monkeypatch, tmp_path):
    reader = SubjectReader(factory())
    monkeypatch.setattr(reader, "select", lambda **_: [{}])
    monkeypatch.setattr("garmin_code.assessments.extract_workbook", calendar)
    spec = tmp_path/"anchors.json"
    spec.write_text(json.dumps({"windows": [{"participant_key": reader.participant_key, "timepoint": "T1",
                                             "start": "2030-04-12", "end": "2030-04-15", "reason": "test"}]}))
    overrides, _ = load_overrides(spec, {reader.participant_key: reader})
    plans, _, _ = assessment_plan(reader, {**overrides, (reader.participant_key, "T1"): {
        "source_timepoint": "T0", "date": "2030-04-09", "reason": "test"}})
    assert plans[1]["dates"] == ["2030-04-12", "2030-04-13", "2030-04-14", "2030-04-15"]
    assert plans[1]["window"]["reason"] == "test" and plans[0]["window"] is None
    assert plans[1]["window_basis"] == "explicit_window"


def xsens_calendar(t1, t2=None):
    def extract(reader, _):
        events = [{"timepoint": label, "event_kind": "protocol_activity", "measure": "Task n4 Xsens", "raw_value_json": '"X"',
                   "clinical_date": day, "source_label": label, "source_cell": "B33"} for label, day in (("T1", t1), ("T2", t2)) if day]
        return [], events, [], {"path": str(reader.root/"subject.json"), "sha256": reader.metadata_sha256["subject.json"]}
    return extract


@pytest.mark.parametrize("paired,expected", [
    ("2030-04-10T16:55:00", ["2030-04-11", "2030-04-12"]),  # first wholly worn day, stopped before T2 on 13 April
    ("2030-04-12T16:55:00", []),  # no wholly worn day before T2: T1 stays blank
])
def test_t1_starts_on_first_wholly_worn_day_and_stops_before_t2(factory, monkeypatch, paired, expected):
    health = lambda rows: [times("2030-04-09T00:00:00", 300) + [0, 2, 0, 0, 0], times(paired, 300) + [100, 2, 10, 10, 1]]
    reader = SubjectReader(factory(health_change=health))
    monkeypatch.setattr(reader, "select", lambda **_: [{}])
    monkeypatch.setattr("garmin_code.assessments.extract_workbook", xsens_calendar("2030-04-09", "2030-04-13"))
    t1 = assessment_plan(reader, {})[0][1]
    assert t1["anchor_date"] == "2030-04-09" and t1["dates"] == expected
    assert t1["window_basis"] == "t1_first_worn_day" and t1["window"]["first_worn_local"] == paired


def test_t1_keeps_standard_window_when_worn_from_its_first_day(factory, monkeypatch):
    reader = SubjectReader(factory())  # worn from 9 April 00:00, so 9 April already counts
    monkeypatch.setattr(reader, "select", lambda **_: [{}])
    monkeypatch.setattr("garmin_code.assessments.extract_workbook", xsens_calendar("2030-04-11"))
    t1 = assessment_plan(reader, {})[0][1]
    assert t1["dates"] == ["2030-04-09", "2030-04-10", "2030-04-11", "2030-04-12", "2030-04-13"]
    assert t1["window"] is None and t1["window_basis"] == "anchor_plus_minus_2"
