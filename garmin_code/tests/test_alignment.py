from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from garmin_code.alignment import align_periods, clinical_day_periods, period_template, run_clinical_alignment, validate_periods
from garmin_code.garmin import GarminSubject
from garmin_code.garmin_workflow import run_windows
from garmin_code.__main__ import main
from garmin_code.tests.test_garmin import factory  # shared raw CSV/ZIP fixture
from shared_code.io import atomic_json
from shared_code.reader import DataContractError, SubjectReader


def metadata(subject):
    doc = json.loads((subject / "subject.json").read_text())
    doc["clinical_metadata"]["timeline_dates"] = {"T0": "2030-04-08", "T1_D1": "2030-04-09",
                                                 "T1_D2": "2030-04-10", "T2": "2030-04-10", "T3": "2030-04-11"}
    atomic_json(subject / "subject.json", doc)
    manifest = json.loads((subject / "manifest.json").read_text())
    for sid, timepoint, day in (("xs1", "T1", None), ("xs2", "T1", 2), ("xs3", "T2", None)):
        manifest["records"].append({"source_id": sid, "recording_id": sid, "modality": "xsens", "disposition": "mapped",
                                    "timepoint": timepoint, "protocol_day": day, "task": 1, "representation": "full_recording",
                                    "session_id": timepoint, "formatter_status": "not_converted",
                                    "acquisition_date": "2030-04-10" if sid == "xs3" else None})
    atomic_json(subject / "manifest.json", manifest)


def spec(key):
    return {"format_version": 1, "time_basis": "source_local_wall_clock", "participants": {
        key: [{"period_id": label, "timepoint": label, "protocol_day": None,
               "start": f"2030-04-{day:02d}", "end_exclusive": f"2030-04-{day+1:02d}",
               "definition_source": "synthetic test intervals; not a patient protocol"}
              for label, day in (("T0", 8), ("T1", 9), ("T2", 10), ("T3", 11))]}}


@pytest.fixture
def pilot(factory, tmp_path):
    subject = factory()
    metadata(subject)
    windows = tmp_path / "windows"
    run_windows([GarminSubject(subject, "2030-04-09", "2030-04-11")], windows)
    period_path = tmp_path / "period-definitions.json"
    atomic_json(period_path, spec("PAZIENTI/FG91"))
    return subject, windows, period_path, tmp_path / "alignment"


def test_half_open_periods_weighted_means_daily_and_hrv_counts(pilot):
    subject, windows, period_path, output = pilot
    periods, features, report = align_periods([subject], [windows], period_path, output)
    p = periods.set_index("period_id")
    assert p.loc["T0", "available_windows"] == 0
    assert p.loc["T1", "available_windows"] == 48
    assert p.loc["T2", "available_windows"] == 48
    assert p.loc["T3", "available_windows"] == 0
    assert p.loc["T0", "missing_windows"] == 48
    assert report["unassigned_windows"] == 0
    f = features.set_index(["period_id", "feature"])
    # Independent source calculation: zero-wear observation (30) is excluded.
    assert f.loc[("T1", "steps_mean"), "value"] == pytest.approx((10 + 20 + 40 + 50) / 4)
    assert f.loc[("T1", "steps_mean"), "used_source_observations"] == 4
    # The reviewed 600-kcal window is excluded without modifying the source value.
    assert f.loc[("T1", "total_calories_kcal_mean"), "value"] == pytest.approx((10 + 20 + 40) / 3)
    assert f.loc[("T1", "total_calories_kcal_mean"), "review_excluded_units"] == 1
    assert f.loc[("T1", "rhr_avg"), "value"] == 60
    assert f.loc[("T1", "rhr_avg"), "used_units"] == 1
    assert f.loc[("T1", "sleep_score"), "used_units"] == 1
    assert f.loc[("T1", "sleep_session_hours"), "value"] == 4
    assert f.loc[("T1", "SDNN"), "used_units"] == 1
    assert f.loc[("T1", "SDNN"), "available_units"] == 4
    assert f.loc[("T1", "SDNN"), "aggregation"] == "median"
    assert pd.isna(f.loc[("T0", "SDNN"), "value"])
    assigned = pd.read_parquet(output / "aligned_windows.parquet")
    assert assigned.loc[assigned["window_start"].eq(pd.Timestamp("2030-04-10")), "period_id"].item() == "T2"
    assert assigned.loc[assigned["window_start"].eq(pd.Timestamp("2030-04-09 00:30")), "total_calories_kcal_mean"].item() == 600
    links = pd.read_csv(output / "xsens_links.csv")
    assert set(links.loc[links["period_id"].eq("T1"), "recording_id"]) == {"xs1", "xs2"}
    assert links.loc[links["period_id"].eq("T2"), "recording_id"].tolist() == ["xs3"]
    tp_links = pd.read_csv(output / "xsens_timepoint_links.csv")
    assert set(tp_links.loc[tp_links["timepoint"].eq("T1"), "recording_id"]) == {"xs1", "xs2"}
    assert tp_links.loc[tp_links["timepoint"].eq("T1"), "date_alignment_status"].eq("acquisition_date_unresolved").all()
    assert tp_links.loc[tp_links["timepoint"].eq("T2"), "date_alignment_status"].item() == "acquisition_date_in_defined_days"
    assert not report["ready_for_features"] and report["review_required"]
    for name, digest in report["output_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def test_exact_day_link_does_not_infer_unlabelled_acquisition(pilot):
    subject, windows, period_path, output = pilot
    value = spec("PAZIENTI/FG91")
    value["participants"]["PAZIENTI/FG91"] = [{**value["participants"]["PAZIENTI/FG91"][2], "timepoint": "T1", "protocol_day": 2}]
    atomic_json(period_path, value)
    _, _, report = align_periods([subject], [windows], period_path, output)
    links = pd.read_csv(output / "xsens_links.csv")
    assert links["recording_id"].tolist() == ["xs2"]
    assert set(pd.read_csv(output / "xsens_timepoint_links.csv")["recording_id"]) == {"xs1", "xs2"}
    assert report["unassigned_windows"] == 48
    assigned = pd.read_parquet(output / "aligned_windows.parquet")
    assert assigned.loc[assigned["window_start"].lt(pd.Timestamp("2030-04-10")), "period_id"].isna().all()


@pytest.mark.parametrize("change,code", [
    (lambda v: v["participants"].update({"FG91": v["participants"].pop("PAZIENTI/FG91")}), "period_identity_mismatch"),
    (lambda v: v["participants"]["PAZIENTI/FG91"][0].update(start=None), "period_boundaries_required"),
    (lambda v: v["participants"]["PAZIENTI/FG91"][0].update(end_exclusive="2030-04-10"), "overlapping_periods"),
    (lambda v: v["participants"]["PAZIENTI/FG91"][0].update(timepoint="T99"), "unknown_timepoint"),
    (lambda v: v["participants"]["PAZIENTI/FG91"][0].update(definition_source=""), "period_source_required"),
    (lambda v: v["participants"]["PAZIENTI/FG91"][0].update(start="20300408"), "invalid_period_dates"),
    (lambda v: v["participants"]["PAZIENTI/FG91"][0].update(protocol_day=True), "invalid_protocol_day"),
])
def test_period_contract(factory, change, code):
    subject = factory()
    metadata(subject)
    reader = SubjectReader(subject)
    value = spec(reader.participant_key)
    change(value)
    with pytest.raises(DataContractError, match=".") as e:
        validate_periods(value, {reader.participant_key: reader})
    assert e.value.code == code


def test_template_preserves_multiple_dates_without_inventing_boundaries(factory, tmp_path):
    subject = factory()
    metadata(subject)
    template = period_template([subject])
    t1 = next(p for p in template["participants"]["PAZIENTI/FG91"] if p["timepoint"] == "T1")
    assert t1["start"] is None and t1["end_exclusive"] is None
    assert t1["clinical_dates"] == {"T1_D1": "2030-04-09", "T1_D2": "2030-04-10"}
    path = tmp_path / "template.json"
    assert main(["garmin-period-template", str(subject), "--output", str(path)]) == 0
    assert json.loads(path.read_text()) == template
    with pytest.raises(SystemExit) as e:
        main(["garmin-period-template", str(subject), "--output", str(path)])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        main(["garmin-period-template", str(subject), "--output", str(subject / "bad.json")])
    assert e.value.code == 2


def test_checksums_snapshot_overlap_and_output_protection(pilot):
    subject, windows, period_path, output = pilot
    with pytest.raises(DataContractError) as e:
        align_periods([subject], [windows, windows], period_path, output)
    assert e.value.code == "duplicate_window"
    with pytest.raises(DataContractError) as e:
        align_periods([subject], [windows], period_path, windows)
    assert e.value.code == "unsafe_output"
    with pytest.raises(DataContractError) as e:
        align_periods([subject], [windows], period_path, subject / "alignment")
    assert e.value.code == "unsafe_output"
    output.mkdir()
    internal_spec = output / "report.json"
    internal_spec.write_bytes(period_path.read_bytes())
    with pytest.raises(DataContractError) as e:
        align_periods([subject], [windows], internal_spec, output)
    assert e.value.code == "unsafe_output"
    artifact = windows / "30min_wearable.parquet"
    artifact.write_bytes(artifact.read_bytes() + b"changed")
    with pytest.raises(DataContractError) as e:
        align_periods([subject], [windows], period_path, output)
    assert e.value.code == "window_checksum_mismatch"


def test_stale_subject_metadata_is_not_silently_relabelled(pilot):
    subject, windows, period_path, output = pilot
    doc = json.loads((subject / "subject.json").read_text())
    doc["clinical_metadata"]["timeline_dates"]["T0"] = "2030-04-07"
    atomic_json(subject / "subject.json", doc)
    with pytest.raises(DataContractError) as e:
        align_periods([subject], [windows], period_path, output)
    assert e.value.code == "alignment_snapshot_mismatch"


def test_cli_writes_review_required_results(pilot):
    subject, windows, period_path, output = pilot
    assert main(["garmin-align", str(subject), "--windows", str(windows), "--periods", str(period_path), "--output", str(output)]) == 1
    assert (output / "report.md").exists()


def test_clinical_dates_and_disjoint_rehab_days_are_aggregated_once(factory, tmp_path):
    subject = factory()
    metadata(subject)
    doc = json.loads((subject / "subject.json").read_text())
    doc["clinical_metadata"].update(status="available", timeline_dates={"T0": "2030-04-08", "T1_D1": "2030-04-09", "T1_D2": "2030-04-11", "T5": "2030-04-12"})
    atomic_json(subject / "subject.json", doc)
    config, plan = clinical_day_periods([subject])
    assert len(config["participants"]["PAZIENTI/FG91"]) == 4
    assert plan.set_index("timepoint").loc["T1", "dated_clinical_days"] == 2
    assert plan.set_index("timepoint").loc["T6", "date_status"] == "no_recorded_clinical_date"
    output = tmp_path / "clinical"
    periods, values, report = run_clinical_alignment([subject], output)
    assert len(periods) == 4
    tp = pd.read_parquet(output / "aligned" / "timepoints.parquet").set_index("timepoint")
    assert tp.loc["T1", "days_in_period"] == 2
    assert tp.loc["T1", "expected_windows"] == 96
    assert tp.loc["T1", "calendar_wall_clock_seconds"] == 2 * 86400
    features = pd.read_parquet(output / "aligned" / "timepoint_features.parquet").set_index(["timepoint", "feature"])
    assert features.loc[("T1", "rhr_avg"), "used_units"] == 1
    assert features.loc[("T1", "rhr_avg"), "value"] == 60
    assert len(pd.read_csv(output / "clinical_alignment_plan.csv")) == 11
    assert report["ready_for_features"] is False
    # Data on April 10 is deliberately outside the recorded rehab-day union.
    raw = pd.read_parquet(output / "aligned" / "aligned_windows.parquet")
    assert not raw["date"].eq("2030-04-10").any()


def test_clinical_cli_progress_is_on_stderr_and_can_be_disabled(factory, tmp_path, capsys):
    subject = factory()
    metadata(subject)
    doc = json.loads((subject / "subject.json").read_text())
    doc["clinical_metadata"].update(status="available", timeline_dates={"T0": "2030-04-09", "T1_D1": "2030-04-10"})
    atomic_json(subject / "subject.json", doc)
    output, quiet = tmp_path / "with-progress", tmp_path / "without-progress"
    args = ["garmin-clinical", str(subject)]
    assert main([*args, "--output", str(output)]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["clinical_days"] == 2
    assert "Starting clinical day 1/2" in captured.err and "Starting clinical day 2/2" in captured.err
    assert "Clinical day processed" in captured.err and "2/2 (100%)" in captured.err
    assert "Calculating period/timepoint summaries" in captured.err
    assert "Writing clinical alignment plan" in captured.err and "Finished; exit status 1" in captured.err

    assert main([*args, "--output", str(quiet), "--no-progress"]) == 1
    captured = capsys.readouterr()
    assert captured.err == "" and json.loads(captured.out)["clinical_days"] == 2
    for name in ("timepoint_features", "timepoints", "aligned_windows"):
        active = pd.read_parquet(output / f"aligned/{name}.parquet")
        silent = pd.read_parquet(quiet / f"aligned/{name}.parquet")
        if "window_artifact" in active:
            # Absolute provenance roots differ because each run has its own destination.
            active["window_artifact"] = active["window_artifact"].map(lambda p: str(Path(p).relative_to(output)))
            silent["window_artifact"] = silent["window_artifact"].map(lambda p: str(Path(p).relative_to(quiet)))
        pd.testing.assert_frame_equal(active, silent)


def test_clinical_extraction_requires_dates_patient_identity_and_bounded_batch(factory, tmp_path):
    subject = factory(cohort="SANI", subject_id="SANO_FG91")
    metadata(subject)
    with pytest.raises(DataContractError) as e:
        clinical_day_periods([subject])
    assert e.value.code == "clinical_dates_unavailable"
    patient = factory()
    metadata(patient)
    doc = json.loads((patient / "subject.json").read_text())
    doc["clinical_metadata"].update(status="available", timeline_dates={"T0": "2030-04-08", "T1_D1": "2030-04-09"})
    atomic_json(patient / "subject.json", doc)
    with pytest.raises(DataContractError) as e:
        run_clinical_alignment([patient], tmp_path / "too-large", max_days=1)
    assert e.value.code == "too_many_clinical_days"
    assert not (tmp_path / "too-large").exists()
