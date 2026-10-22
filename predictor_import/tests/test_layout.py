"""Readable naming, legacy-cache migration and failure preservation."""
import hashlib
import json

import openpyxl
import pytest

from predictor_import.adapter import atomic_json, make_plan, write_plan
from predictor_import.formatter import format_subject
from predictor_import.inventory import scan
from predictor_import.layout import migrate_layout, migrate_subject
from predictor_import.storage_names import xsens_table_names, xsens_targets
from predictor_import.tests.test_formatter import acquisition
from shared_code.reader import SubjectReader


def as_legacy(subject, *, rename_directory=True):
    manifest = json.loads((subject / "manifest.json").read_text())
    record = manifest["records"][0]
    target = subject / record["normalized_target"]
    descriptor = json.loads((target / "recording.json").read_text())
    for index, table in enumerate(descriptor["tables"]):
        if table["format"] == "parquet":
            filename = f"sheet_{index:02d}.parquet"
            (target / table["file"]).rename(target / filename)
            table["file"] = filename
    atomic_json(target / "recording.json", descriptor)
    if rename_directory:
        relative = f"xsens/{record['session_id']}/{record['recording_id']}"
        target.rename(subject / relative)
        record["normalized_target"] = relative
        record["formatter_report"]["descriptor_path"] = relative + "/recording.json"
    atomic_json(subject / "manifest.json", manifest)
    return subject / record["normalized_target"]


@pytest.fixture
def legacy(tmp_path):
    raw = tmp_path / "raw"
    source = raw / "SANI/SANO_BA001/XSens/BA001_t1_task1_dx.xlsx"
    acquisition(source)
    output = tmp_path / "standardized"
    write_plan(make_plan(scan(raw)), output)
    subject = output / "SANI/SANO_BA001"
    assert format_subject(subject)["converted"] == 1
    old = as_legacy(subject)
    return source, subject, old


def snapshot(folder):
    return {str(p.relative_to(folder)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in folder.rglob("*") if p.is_file()}


def test_readable_targets_distinguish_arms_sessions_and_repetitions(tmp_path):
    raw = tmp_path / "raw"
    paths = [
        "PAZIENTI/FG91/Xsens/FG91_01_TASK4_T1.xlsx",
        "PAZIENTI/FG91/Xsens/FG91_00_TASK4_T1.xlsx",
        "PAZIENTI/FG91/Xsens/FG91_01_TASK4_T1_d2.xlsx",
        "SANI/SANO_BA001/XSens/BA001_t1_task1_dx.xlsx",
        "SANI/SANO_BA001/XSens/BA001_t1_task1_sx.xlsx",
        "SANI/SANO_BA001/XSens/BA001_task2_rip1_dx.xlsx",
        "SANI/SANO_BA001/XSens/BA001_task2_rip2_dx.xlsx",
        "SANI/SANO_BA001/XSens/BA001_t1_task3_sim.xlsx",
        "CONTROLLO PAZIENTI/CTRLFG_91/XSens/ctrlfg91_01_b&b_t1.xlsx",
    ]
    for path in paths:
        acquisition(raw / path)
    records = [r for s in make_plan(scan(raw))["subjects"] for r in s["manifest"]["records"]]
    assert {r["normalized_target"] for r in records} == {
        "xsens/T1/task4_paretic", "xsens/T1/task4_non_paretic", "xsens/T1_d2/task4_paretic",
        "xsens/T1/task1_right", "xsens/T1/task1_left", "xsens/T1/task3_simulated",
        "xsens/unlabelled_sani_acquisition/task2_right_rep1",
        "xsens/unlabelled_sani_acquisition/task2_right_rep2", "xsens/T1/box_and_blocks_paretic",
    }
    assert all(r["recording_id"].startswith("rec_") for r in records)


def test_actual_recording_collisions_use_source_suffixes_independent_of_order():
    base = {"artifact_kind": "xsens_workbook", "disposition": "mapped", "session_id": "T1",
            "activity": "task1", "limb_condition": "paretic"}
    records = [dict(base, source_id="src_" + value + "0" * 16)
               for value in ("12345678", "87654321")]
    names = xsens_targets(records)
    assert set(names.values()) == {"xsens/T1/task1_paretic__12345678", "xsens/T1/task1_paretic__87654321"}
    assert names == xsens_targets(list(reversed(records)))
    records[1]["source_id"] = "src_123456781234" + "0" * 12
    assert len(set(xsens_targets(records).values())) == 2


def test_worksheet_collisions_are_safe_and_independent_of_order():
    titles = ["Markers", "Segment Orientation - Quat", "Joint Angles ZXY",
              "Custom Value", "Custom-Value", "Custom Value 01", "General Information"]
    names = xsens_table_names(titles)
    assert names["Markers"] == "markers.parquet"
    assert names["Segment Orientation - Quat"] == "segment_orientation_quat.parquet"
    assert names["Joint Angles ZXY"] == "joint_angles_zxy.parquet"
    assert names["Custom Value"] != names["Custom-Value"]
    assert len(set(names.values())) == len(titles)
    assert names == xsens_table_names(list(reversed(titles)))


def test_formatter_preserves_custom_worksheets_with_colliding_names(tmp_path):
    raw = tmp_path / "raw"
    source = raw / "SANI/SANO_BA001/XSens/BA001_t1_task1_dx.xlsx"
    acquisition(source)
    book = openpyxl.load_workbook(source)
    for name, value in [("Custom Value", 12.5), ("Custom-Value", 99.5)]:
        sheet = book.create_sheet(name)
        sheet.append(["Frame", "value"])
        sheet.append([1, value])
    book.save(source)
    book.close()
    output = tmp_path / "standardized"
    write_plan(make_plan(scan(raw)), output)
    subject = output / "SANI/SANO_BA001"
    assert format_subject(subject)["failed"] == 0
    reader = SubjectReader(subject)
    recording = reader.open_xsens(reader.select(modality="xsens")[0])
    assert recording.read_table("Custom Value")["value"].to_pylist() == [12.5]
    assert recording.read_table("Custom-Value")["value"].to_pylist() == [99.5]


def test_migration_preserves_values_checksums_timestamps_ids_and_raw_files(legacy):
    source, subject, old = legacy
    raw_before, source_stamp = source.read_bytes(), source.stat().st_mtime_ns
    manifest_before = json.loads((subject / "manifest.json").read_text())
    descriptor_before = json.loads((old / "recording.json").read_text())
    stamps = {t["source_sheet"]: (old / t["file"]).stat().st_mtime_ns for t in descriptor_before["tables"]}
    result = migrate_subject(subject)
    assert result["recordings_migrated"] == 1 and result["tables_rebuilt"] == 0
    assert not old.exists()
    reader = SubjectReader(subject)
    record = reader.select(modality="xsens")[0]
    assert record["normalized_target"] == "xsens/T1/task1_right"
    assert record["source_id"] == manifest_before["records"][0]["source_id"]
    assert record["recording_id"] == manifest_before["records"][0]["recording_id"]
    recording = reader.open_xsens(record)
    assert recording.read_table("Segment Velocity")["Left Hand x"].to_pylist()[3] is None
    for before, after in zip(descriptor_before["tables"], recording.descriptor["tables"]):
        assert before["sha256"] == after["sha256"]
        assert before["row_count"] == after["row_count"]
        assert stamps[after["source_sheet"]] == (recording.descriptor_path.parent / after["file"]).stat().st_mtime_ns
    assert source.read_bytes() == raw_before and source.stat().st_mtime_ns == source_stamp
    current = snapshot(subject)
    assert migrate_subject(subject)["records_changed"] == 0
    assert snapshot(subject) == current


def test_dry_run_does_not_change_existing_metadata_or_tables(legacy):
    _, subject, _ = legacy
    before = snapshot(subject)
    assert migrate_layout(subject.parents[1], dry_run=True)["recordings_migrated"] == 1
    assert snapshot(subject) == before


@pytest.mark.parametrize("entrypoint", ["adapter", "formatter"])
def test_adapter_and_formatter_reuse_legacy_outputs_without_rebuilding(legacy, entrypoint):
    source, subject, old = legacy
    if entrypoint == "adapter":
        result = write_plan(make_plan(scan(source.parents[3])), subject.parents[1])
        assert len(result["retained_verified_formatter_sources"]) == 1
    result = format_subject(subject)
    assert result["cached"] == 1 and result["converted"] == result["failed"] == 0
    assert result["results"][0]["tables_rebuilt"] == 0
    assert not old.exists()


def test_table_only_migration_replaces_directory_without_changing_its_name(legacy):
    _, subject, _ = legacy
    migrate_subject(subject)
    old = as_legacy(subject, rename_directory=False)
    assert migrate_subject(subject)["recordings_migrated"] == 1
    assert old.exists() and (old / "markers.parquet").is_file()
    assert not list(subject.rglob(".layout-*"))


@pytest.mark.parametrize("damage", ["occupied_destination", "corrupt_table", "unexpected_file"])
def test_migration_refuses_conflicts_and_preserves_originals(legacy, damage):
    _, subject, old = legacy
    if damage == "occupied_destination":
        destination = subject / "xsens/T1/task1_right"
        destination.mkdir()
        (destination / "recording.json").write_text("{}")
    elif damage == "corrupt_table":
        (old / "sheet_01.parquet").write_bytes(b"damaged")
    else:
        (old / "user_notes.txt").write_text("preserve me")
    before = snapshot(subject)
    with pytest.raises((ValueError, KeyError)):
        migrate_subject(subject)
    assert snapshot(subject) == before


@pytest.mark.parametrize("table_only", [False, True])
def test_manifest_publication_failure_rolls_back_renamed_directories(legacy, monkeypatch, table_only):
    _, subject, _ = legacy
    if table_only:
        migrate_subject(subject)
        as_legacy(subject, rename_directory=False)
    before = snapshot(subject)
    def fail_manifest(path, value):
        if path == subject / "manifest.json":
            raise OSError("simulated manifest write failure")
        atomic_json(path, value)
    monkeypatch.setattr("predictor_import.adapter.atomic_json", fail_manifest)
    with pytest.raises(OSError, match="manifest write failure"):
        migrate_subject(subject)
    assert snapshot(subject) == before
    assert not list(subject.rglob(".layout-*"))


def test_migration_resumes_a_verified_already_published_destination(legacy):
    _, subject, old = legacy
    # Simulate interruption after destination publication, before manifest update.
    before_manifest = (subject / "manifest.json").read_bytes()
    before_descriptor = (old / "recording.json").read_bytes()
    before = json.loads(before_descriptor)
    migrate_subject(subject)
    destination = subject / "xsens/T1/task1_right"
    old.mkdir()
    new_descriptor = json.loads((destination / "recording.json").read_text())
    for original, renamed in zip(before["tables"], new_descriptor["tables"]):
        (old / original["file"]).hardlink_to(destination / renamed["file"])
    (old / "recording.json").write_bytes(before_descriptor)
    (subject / "manifest.json").write_bytes(before_manifest)
    assert migrate_subject(subject)["recordings_migrated"] == 1
    assert not old.exists()
    assert format_subject(subject)["cached"] == 1


def test_formatter_never_overwrites_another_source_at_a_readable_destination(legacy):
    source, subject, _ = legacy
    migrate_subject(subject)
    target = subject / "xsens/T1/task1_right"
    before = snapshot(target)
    # The renamed raw location is a different source ID, despite identical bytes.
    source.rename(source.with_suffix(".XLSX"))
    write_plan(make_plan(scan(source.parents[3])), subject.parents[1])
    result = format_subject(subject)
    assert result["failed"] == 1
    assert "different source" in result["results"][0]["error"]
    assert snapshot(target) == before
