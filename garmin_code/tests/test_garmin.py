from __future__ import annotations

import csv
from datetime import datetime, timedelta
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from predictor_import.adapter import make_plan, write_plan, atomic_json
from predictor_import.garmin_formatter import format_subject
from predictor_import.inventory import scan
from shared_code.reader import DataContractError
from garmin_code.garmin import (
    GarminSubject,
    GarminConfig,
    build_windows,
    normalize,
    within_intervals,
)
from garmin_code.garmin_workflow import (
    run_windows,
    export_preparation,
    runs_from_ranges,
)
from garmin_code.__main__ import main


def source_csv(headers, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(headers)
    writer.writerows(rows)
    return stream.getvalue()


def times(local, seconds=None):
    utc = datetime.fromisoformat(local) - timedelta(hours=2)
    fields = [local, utc.isoformat(timespec="milliseconds"), 7200]
    if seconds is not None:
        fields += [
            (datetime.fromisoformat(local) + timedelta(seconds=seconds)).isoformat(),
            (utc + timedelta(seconds=seconds)).isoformat(),
        ]
    return fields


@pytest.fixture
def factory(tmp_path):
    def create(
        *,
        health_change=None,
        bbi_change=None,
        sleep_change=None,
        stress_change=None,
        cohort="PAZIENTI",
        subject_id="FG91",
        missing=None,
    ):
        timestamp = ["Start Time (Local)", "Start Time (UTC)", "Time Zone (s)"]
        tables = {}
        health = [
            times(f"2030-04-09T{t}", 300) + [wear, 2, steps, calories, 1]
            for t, wear, steps, calories in [
                ("00:00:00", 100, 10, 10),
                ("00:05:00", 100, 20, 20),
                ("00:10:00", 0, 30, 30),
                ("00:15:00", 50, 40, 40),
                ("00:30:00", 100, 50, 600),
            ]
        ]
        health += [times("2030-04-10T00:00:00", 300) + [100, 2, 60, 6, 1]]
        if health_change:
            health = health_change(health)
        tables["detailed-health-log.csv"] = (
            timestamp
            + [
                "End Time (Local)",
                "End Time (UTC)",
                "Wearing Time (%)",
                "MET (avg)",
                "Steps",
                "Total Calories (kcal)",
                "Active Calories (kcal)",
            ],
            health,
        )
        stress = [
            times("2030-04-09T00:00:00") + [10, 60, 1],
            times("2030-04-09T00:01:00") + [20, 20, 1],
        ]
        if stress_change:
            stress = stress_change(stress)
        tables["stress.csv"] = (
            timestamp
            + ["Stress Level Value", "Body Battery (%)", "Body Battery Status"],
            stress,
        )
        tables["heart-rate.csv"] = (
            timestamp + ["Heart Rate (bpm)"],
            [times("2030-04-09T00:00:00") + [70]],
        )
        tables["respiration.csv"] = (
            timestamp + ["Respiration (breaths/min)", "Status"],
            [
                times("2030-04-09T00:00:00") + [12, 0],
                times("2030-04-09T00:01:00") + [13, 1],
                times("2030-04-09T00:10:00") + [14, 0],
            ],
        )
        bbi = [
            times("2030-04-09T00:00:" + t) + [v]
            # Raw BBI timing: each time step equals the following beat's interval.
            for t, v in [
                ("00.000", 800),
                ("00.900", 900),
                ("01.900", 1000),
                ("07.900", 1100),
                ("09.100", 1200),
                ("10.000", 9999),
                ("11.000", 1300),
            ]
        ]
        if bbi_change:
            bbi = bbi_change(bbi)
        tables["bbi-(raw).csv"] = (timestamp + ["BBI (ms)"], bbi)
        sleep = [
            times("2030-04-08T21:00:00", 14400) + [80, "deep"],
            times("2030-04-08T21:00:00", 14400) + [80, "light"],
        ]
        if sleep_change:
            sleep = sleep_change(sleep)
        tables["sleep.csv"] = (
            timestamp
            + [
                "End Time (Local)",
                "End Time (UTC)",
                "Sleep Score Value",
                "Sleep Level",
            ],
            sleep,
        )
        tables["daily-health-log.csv"] = (
            timestamp + ["Resting Heart Rate (avg)"],
            [times("2030-04-09T00:00:00") + [60]],
        )
        if missing:
            tables.pop(missing)
        raw = tmp_path / "raw"
        archive = raw / cohort / subject_id / "Garmin" / "export.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(archive, "w") as zipped:
            for filename, (columns, rows) in tables.items():
                zipped.writestr("nested/" + filename, source_csv(columns, rows))
        output = tmp_path / "standardized"
        write_plan(make_plan(scan(raw)), output)
        subject = output / cohort / subject_id
        report = format_subject(subject)
        assert report["converted"] == 1, report
        return subject

    return create


def run(subject):
    return GarminSubject(subject, "2030-04-09", "2030-04-10")


def test_local_dates_half_open_means_separate_calories_and_provenance(factory):
    pipeline = run(factory())
    p = pipeline.prepare("health")
    assert len(p.data) == 5 and p.data.loc[0, "start_time_utc"].startswith("2030-04-08")
    assert p.data["source_row"].tolist() == list(range(5))
    frame, norm, names = build_windows(pipeline)
    assert len(frame) == 48
    assert frame.loc[0, "steps_mean"] == pytest.approx((10 + 20 + 40) / 3)
    assert frame.loc[0, "total_calories_kcal_mean"] == pytest.approx((10 + 20 + 40) / 3)
    assert frame.loc[0, "active_calories_kcal_mean"] == 1
    assert "calories_kcal_mean" not in names
    assert not pipeline.report()["provenance"][0]["raw_source_rechecked"]


def test_high_calories_preserved_reviewed_and_not_normalized(factory):
    pipeline = run(factory())
    frame, norm, _ = build_windows(pipeline)
    assert pipeline.prepare("health").data.loc[4, "total_calories_kcal"] == 600
    assert frame.loc[1, "total_calories_kcal_mean"] == 600
    assert frame.loc[1, "total_calories_kcal_mean_review"]
    assert pd.isna(norm.loc[1, "total_calories_kcal_mean"])
    assert norm.loc[0, "total_calories_kcal_mean"] == 0
    flag = next(
        f
        for f in pipeline.prepare("health").findings
        if f["code"] == "high_calorie_rate"
    )
    assert flag["examples"][0]["source_csv_record"] == 5
    assert flag["examples"][0]["raw_value"] == "600"


def test_coverage_partial_wearing_and_empty_window_never_assumed_100(factory):
    frame, _, _ = build_windows(run(factory()))
    assert frame.loc[0, "health_coverage_pct"] == pytest.approx(20 / 30 * 100)
    assert frame.loc[0, "wearing_time_pct_observed"] == 62.5
    assert frame.loc[2, "health_coverage_pct"] == 0
    assert pd.isna(frame.loc[2, "wearing_time_pct_observed"])
    assert pd.isna(frame.loc[2, "steps_mean"])


def test_bbi_differences_never_bridge_gaps_or_excluded_source_rows(factory):
    config = GarminConfig(window_minutes=30, hrv_method="segments", bbi_screens=True)
    frame, _, _ = build_windows(GarminSubject(factory(), "2030-04-09", "2030-04-10", config))
    row = frame.iloc[0]
    assert (
        row["n_beats"] == 6
        and row["n_adjacent_pairs"] == 3
        and row["hrv_pairs_excluded"] == 2
    )
    assert row["RMSSD"] == 100 and row["NN50"] == 3 and row["pNN50"] == 100
    assert row["SDNN"] == pytest.approx(
        np.std([800, 900, 1000, 1100, 1200, 1300], ddof=1)
    )
    assert frame.loc[11, "SDNN"] == row["SDNN"] and pd.isna(frame.loc[12, "SDNN"])
    assert frame["VLF"].isna().all() and frame["ApEn"].isna().all()


def test_status_qc_is_feature_specific_and_hr_reference_is_explicit(factory):
    pipeline = run(factory())
    stress = pipeline.prepare("stress")
    resp = pipeline.prepare("respiration")
    assert stress.eligible["stress_level_value"].tolist() == [True, False]
    assert stress.eligible["body_battery"].tolist() == [True, True]
    assert resp.eligible["respiration_breaths_min"].tolist() == [True, False, False]


def test_sleep_phase_summaries_collapse_without_losing_source_rows(factory):
    pipeline = run(factory())
    p = pipeline.prepare("sleep")
    assert len(p.data) == 2 and p.row_eligible.tolist() == [True, False]
    frame, _, _ = build_windows(pipeline)
    assert frame["sleep_session_count"].eq(1).all()
    assert (
        frame["sleep_score"].eq(80).all() and frame["sleep_session_hours"].eq(4).all()
    )


def test_sleep_merges_sessions_under_two_hours_and_drops_naps(factory):
    nap = factory(sleep_change=lambda rows: rows + [times("2030-04-09T13:00:00", 3600) + [60, "deep"]])
    frame, _, _ = build_windows(run(nap))
    assert frame["sleep_session_count"].eq(2).all()
    assert frame["sleep_score"].eq(80).all() and frame["sleep_session_hours"].eq(4).all()
    split = factory(sleep_change=lambda rows: rows + [times("2030-04-09T01:30:00", 5400) + [60, "deep"]])
    frame, _, _ = build_windows(run(split))
    assert frame["sleep_score"].eq(70).all() and frame["sleep_session_hours"].eq(6).all()


def test_conflicting_duplicates_excluded_instead_of_arbitrarily_selected(factory):
    def conflict(rows):
        copy = rows[0].copy()
        copy[-3] = 999
        return rows + [copy]

    pipeline = run(factory(health_change=conflict))
    p = pipeline.prepare("health")
    assert not p.row_eligible.iloc[0] and not p.row_eligible.iloc[-1]
    assert any(
        f["code"] == "conflicting_utc_observation" and f["count"] == 2
        for f in p.findings
    )


def test_exact_duplicates_collapsed_with_explicit_flag(factory):
    pipeline = run(factory(health_change=lambda rows: rows + [rows[0].copy()]))
    p = pipeline.prepare("health")
    assert p.row_eligible.iloc[0] and not p.row_eligible.iloc[-1]
    assert any(f["code"] == "exact_observation_duplicate" for f in p.findings)


def test_timezone_conflicts_not_forced_or_repaired(factory):
    def change(rows):
        rows[0][2] = 3600
        return rows

    pipeline = run(factory(health_change=change))
    p = pipeline.prepare("health")
    assert p.data.iloc[0]["time_zone_s"] == 3600 and not p.row_eligible.iloc[0]
    assert any(f["code"] == "invalid_or_inconsistent_timestamp" for f in p.findings)


def test_negative_nonfinite_and_literal_na_are_preserved_but_excluded(factory):
    def change(rows):
        rows[0][-3] = "NA"
        rows[1][-3] = "Infinity"
        rows[3][-3] = -4
        return rows

    p = run(factory(health_change=change)).prepare("health")
    assert p.data["steps"].tolist()[:2] == ["NA", "Infinity"]
    assert not p.eligible.loc[[0, 1, 3], "steps"].any()


def test_header_only_and_no_observed_period_are_still_rectangular(factory):
    pipeline = GarminSubject(factory(), "2030-05-01", "2030-05-02")
    frame, norm, _ = build_windows(pipeline)
    assert len(frame) == 48 and frame["steps_mean"].isna().all()
    assert frame["wearing_time_pct_observed"].isna().all()
    assert frame["health_coverage_pct"].eq(0).all() and frame["n_beats"].eq(0).all()


def test_reader_rejects_corruption_and_incomplete_conversion(factory):
    pipeline = run(factory())
    export = pipeline.exports[0]
    table = next(
        t
        for t in export.descriptor["tables"]
        if t["source_table"].endswith("detailed-health-log.csv")
    )
    (export.descriptor_path.parent / table["file"]).write_bytes(b"corrupt")
    with pytest.raises(DataContractError, match="Changed table"):
        pipeline.prepare("health")
    export.descriptor["complete_selected_tables"] = False
    atomic_json(export.descriptor_path, export.descriptor)
    with pytest.raises(DataContractError, match="incomplete"):
        run(pipeline.reader.root)


def test_descriptor_metadata_change_and_missing_required_stream_are_errors(factory):
    pipeline = run(factory(missing="stress.csv"))
    with pytest.raises(DataContractError, match="found 0"):
        pipeline.prepare("stress")
    export = pipeline.exports[0]
    export.descriptor["source_sha256"] = "a" * 64
    atomic_json(export.descriptor_path, export.descriptor)
    with pytest.raises(DataContractError, match="revision"):
        run(pipeline.reader.root)


@pytest.mark.parametrize("minutes", [0, -1, 50, 90.5, 361])
def test_incompatible_window_grid_is_rejected(minutes):
    with pytest.raises(DataContractError):
        GarminConfig(window_minutes=minutes)


@pytest.mark.parametrize(
    "start,end",
    [
        ("2030-04-10", "2030-04-09"),
        ("2030-04-09", "2030-04-09"),
        ("20300409", "2030-04-10"),
    ],
)
def test_explicit_canonical_dates_required(factory, start, end):
    with pytest.raises(DataContractError):
        GarminSubject(factory(), start, end)


def test_normalization_constant_missing_and_review_values():
    frame = pd.DataFrame(
        {
            "participant_key": ["PAZIENTI/A"] * 3,
            "x": [5.0, 5.0, np.nan],
            "x_review": [False, True, False],
        }
    )
    normalized, scaling = normalize(frame, ["x"])
    assert normalized.loc[0, "x"] == 0 and normalized.loc[1:, "x"].isna().all()
    assert scaling[0]["observations"] == 1


def test_overlapping_interval_union_membership_is_half_open():
    assert within_intervals(
        np.array([0, 1, 2, 8, 9]), np.array([1, 2]), np.array([9, 3])
    ).tolist() == [False, True, True, True, False]


def test_batch_outputs_hashes_and_protected_inputs(factory, tmp_path):
    pipeline = run(factory())
    inputs = [p for p in pipeline.reader.root.rglob("*") if p.is_file()]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    frame, norm, report = run_windows([pipeline], tmp_path / "outputs")
    assert (
        report["windows"] == 48
        and not report["ema_joined"]
        and not report["ready_for_features"]
    )
    assert pq.read_table(tmp_path / "outputs/30min_wearable.parquet").num_rows == 48
    blocks = pq.read_table(tmp_path / "outputs/hrv_blocks.parquet").to_pandas()
    assert len(blocks) == 4 and blocks["participant_key"].eq(pipeline.participant_key).all()
    assert pq.read_table(tmp_path / "outputs/hrv_epochs.parquet").num_rows == 0
    assert report["deferred_hrv_fields"] == [] and "SD1" in report["feature_columns"]
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    with pytest.raises(DataContractError, match="outside"):
        run_windows([pipeline], pipeline.reader.root / "features")
    prepared, qc = export_preparation([pipeline], "health", tmp_path / "prep")
    assert len(prepared) == 5 and prepared.loc[4, "total_calories_kcal"] == 600
    assert prepared.loc[4, "total_calories_kcal__review"]


def test_cohort_qualified_ranges_and_duplicate_subjects_are_enforced(factory, tmp_path):
    subject = factory()
    key = "PAZIENTI/FG91"
    with pytest.raises(DataContractError, match="exactly match"):
        runs_from_ranges(
            [subject], {"FG91": {"start": "2030-04-09", "end_exclusive": "2030-04-10"}}
        )
    with pytest.raises(DataContractError, match="distinct"):
        run_windows([run(subject), run(subject)], tmp_path / "duplicate")
    with pytest.raises(SystemExit) as error:
        main(["garmin-windows", str(subject), "--output", str(tmp_path / "cli")])
    assert error.value.code == 2
    assert (
        main(
            [
                "garmin-windows",
                str(subject),
                "--start",
                "2030-04-09",
                "--end-exclusive",
                "2030-04-10",
                "--output",
                str(tmp_path / "cli"),
            ]
        )
        == 1
    )


def test_window_cli_progress_preserves_features_and_clean_json(factory, tmp_path, capsys):
    subject = factory()
    args = ["garmin-windows", str(subject), "--start", "2030-04-09", "--end-exclusive", "2030-04-10"]
    output = tmp_path / "with-progress"
    assert main([*args, "--output", str(output)]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["windows"] == 48
    for stage in ("health stream", "stress stream", "respiration stream", "health coverage",
                  "BBI stream", "HRV block", "daily sleep", "normalizing", "Writing", "Finished; exit status 1"):
        assert stage in captured.err
    assert "4/4 (100%)" in captured.err and "elapsed" in captured.err

    quiet = tmp_path / "without-progress"
    assert main([*args, "--output", str(quiet), "--no-progress"]) == 1
    captured = capsys.readouterr()
    assert captured.err == "" and json.loads(captured.out)["windows"] == 48
    for name in ("30min_wearable", "30min_wearable_normalized", "hrv_blocks", "hrv_epochs"):
        assert pq.read_table(output / f"{name}.parquet").equals(pq.read_table(quiet / f"{name}.parquet"))
        assert (output / f"{name}.csv").read_bytes() == (quiet / f"{name}.csv").read_bytes()


def test_window_cli_reports_input_errors_without_claiming_completion(factory, tmp_path, capsys):
    subject = factory()
    with pytest.raises(SystemExit) as error:
        main(["garmin-windows", str(subject), "--output", str(tmp_path / "bad")])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Stopped: dates_required" in captured.err
    assert "Finished" not in captured.err


def test_pruned_unsorted_groups_keep_original_csv_positions(factory):
    pipeline = run(factory(health_change=lambda rows: [rows[-1].copy()] + rows))
    export = pipeline.exports[0]
    desc = next(
        t
        for t in export.descriptor["tables"]
        if t["source_table"].endswith("detailed-health-log.csv")
    )
    path = export.descriptor_path.parent / desc["file"]
    values = pq.read_table(path)
    pq.write_table(values, path, row_group_size=1)
    desc["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(export.descriptor_path, export.descriptor)
    pipeline = run(pipeline.reader.root)
    prepared = pipeline.prepare("health")
    assert prepared.data["source_row"].tolist() == [1, 2, 3, 4, 5]
    checked = next(iter(pipeline.exports[0].checked_tables.values()))
    assert checked["pruned_row_groups"] == 2 and checked["decoded_rows"] == 5


def test_bounded_reader_refuses_unbounded_selection(factory):
    pipeline = GarminSubject(
        factory(), "2030-04-09", "2030-04-10", GarminConfig(max_selected_rows=4)
    )
    with pytest.raises(DataContractError) as error:
        pipeline.prepare("health")
    assert error.value.code == "period_too_large"


def test_changed_window_artifact_cannot_generate_new_matrices(factory, tmp_path):
    from garmin_code.garmin_matrices import export_matrices

    output = tmp_path / "windows"
    run_windows([run(factory())], output)
    path = output / "30min_wearable_normalized_day.parquet"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(DataContractError) as error:
        export_matrices(output, tmp_path / "matrices")
    assert error.value.code == "window_checksum_mismatch"


def test_multiple_daily_session_flags_are_stable_on_rerun(factory):
    pipeline = run(
        factory(
            sleep_change=lambda rows: rows
            + [times("2030-04-09T13:00:00", 3600) + [60, "deep"]]
        )
    )
    build_windows(pipeline)
    build_windows(pipeline)
    findings = [
        f
        for f in pipeline.prepare("sleep").findings
        if f["code"] == "multiple_daily_sessions"
    ]
    assert len(findings) == 1 and findings[0]["count"] == 1


def test_zero_coded_record_is_missing_and_unknown_met_keeps_beats(factory):
    def change(rows):
        rows[0][-5:] = [100, 0, 0, 0, 0]  # worn, but MET, steps and calories all 0
        return rows

    pipeline = GarminSubject(factory(health_change=change), "2030-04-09", "2030-04-10", GarminConfig(bbi_screens=True))
    h = pipeline.prepare("health")
    assert not h.eligible.loc[0, ["total_calories_kcal", "active_calories_kcal", "steps", "met_avg"]].any()
    assert h.eligible.loc[1, ["total_calories_kcal", "steps"]].all()
    assert not h.eligible.loc[1, "met_avg"]  # same 15-min MET quarter as the missing record
    frame, _, _ = build_windows(pipeline)
    assert frame.loc[0, "steps_mean"] == pytest.approx((20 + 40) / 2)
    bbi = pipeline.prepare("bbi")
    assert bbi.eligible["bbi_ms"].sum() == 6
    assert any(f["code"] == "rest_unverified_met_unknown" for f in bbi.findings)


def test_day_without_resting_energy_keeps_active_but_not_total(factory):
    def change(rows):
        for row in rows:
            row[-1] = row[-2]
        return rows

    h = run(factory(health_change=change)).prepare("health")
    assert not h.eligible["total_calories_kcal"].any()
    assert h.eligible.loc[[0, 1, 3, 4], "active_calories_kcal"].all()


def test_second_beat_stream_excludes_beats_and_stress_but_not_health(factory):
    def double(rows):
        start = datetime(2030, 4, 9, 0, 1)
        return rows + [
            times((start + timedelta(milliseconds=450 * k)).isoformat(timespec="milliseconds")) + [900]
            for k in range(400)
        ]

    pipeline = GarminSubject(factory(bbi_change=double), "2030-04-09", "2030-04-10", GarminConfig(exclude_second_stream=True))
    assert not pipeline.prepare("bbi").eligible["bbi_ms"].any()
    stress = pipeline.prepare("stress")
    assert not stress.eligible["stress_level_value"].any() and stress.eligible["body_battery"].all()
    assert pipeline.prepare("health").eligible.loc[0, "steps"]


def test_pair_requires_time_step_equal_to_following_interval(factory):
    def missed_beat(rows):
        rows[2][:3] = times("2030-04-09T00:00:02.900")  # one beat between 0.9 s and 2.9 s was not detected
        return rows

    frame, _, _ = build_windows(GarminSubject(factory(bbi_change=missed_beat), "2030-04-09", "2030-04-10",
                                              GarminConfig(window_minutes=360, hrv_method="segments", bbi_screens=True)))
    assert frame.iloc[0].n_adjacent_pairs == 2 and frame.iloc[0].RMSSD == 100


def test_default_uses_every_beat_with_notebook_formulas(factory):
    frame, _, _ = build_windows(GarminSubject(factory(), "2030-04-09", "2030-04-10", GarminConfig(window_minutes=360)))
    row = frame.iloc[0]
    x = np.array([800, 900, 1000, 1100, 1200, 9999, 1300], float)  # no range, wear or rest screen
    d = np.diff(x)
    assert row["n_beats"] == 7 and row["RMSSD"] == pytest.approx(np.sqrt(np.mean(d ** 2)))
    assert pd.isna(row["SD2_SD1_ratio"])  # the notebook needs 10 beats
    assert pd.isna(row["SampEn"]) and pd.isna(row["LF_HF_ratio"])  # and 50 beats


def test_zero_coded_met_is_recomputed_from_an_intact_quarter(factory):
    def change(rows):
        for row, total in zip(rows[:3], (10, 20, 30)):
            row[-5:] = [100, 0, 5, total, 1]
        return rows

    h = GarminSubject(factory(health_change=change), "2030-04-09", "2030-04-10").prepare("health")
    assert h.values.loc[:2, "met_avg"].tolist() == pytest.approx([60 / 57] * 3)
    assert h.eligible.loc[:2, "met_avg"].all() and h.data.loc[:2, "_met_recomputed"].all()
    assert not h.data.loc[3:, "_met_recomputed"].any()
