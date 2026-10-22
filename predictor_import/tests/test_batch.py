from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import openpyxl
import pyarrow.parquet as pq
import pytest
from jsonschema import Draft202012Validator

from predictor_import.adapter import make_plan, write_plan
from predictor_import.batch import run_batch
from predictor_import.inventory import scan
from predictor_import.garmin_formatter import format_subject
from predictor_import.tests.test_formatter import acquisition


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "raw"
    source = root / "PAZIENTI/FG92/Garmin/value.csv"
    source.parent.mkdir(parents=True)
    source.write_text("Value\n1\n", encoding="utf-8")
    return root, source, tmp_path / "standardized", tmp_path / "reports"


def read_export(output):
    subject = output / "PAZIENTI/FG92"
    manifest = json.loads((subject / "manifest.json").read_text())
    record = next(r for r in manifest["records"] if r["artifact_kind"] in {"garmin_archive", "garmin_table"} and r["disposition"] == "mapped")
    directory = subject / record["normalized_target"]
    descriptor = json.loads((directory / "export.json").read_text())
    return record, directory, descriptor


def test_new_then_unchanged_data_is_not_reformatted(dataset):
    root, _, output, reports = dataset
    first = run_batch(root, output, reports / "first")
    assert first["results"][0]["content_state"] == "new"
    record, directory, descriptor = read_export(output)
    stat = (directory / descriptor["tables"][0]["file"]).stat()
    second = run_batch(root, output, reports / "second")
    assert second["results"][0]["status"] == "cached"
    assert second["results"][0]["content_state"] == "unchanged"
    assert second["summary"]["tables_rebuilt"] == 0
    assert (directory / descriptor["tables"][0]["file"]).stat().st_mtime_ns == stat.st_mtime_ns
    assert second["staging"]["retained_verified_formatter_sources"] == [record["source_id"]]


def test_same_name_size_and_timestamp_with_new_bytes_is_detected(dataset):
    root, source, output, reports = dataset
    run_batch(root, output, reports / "first")
    before = source.stat()
    source.write_text("Value\n9\n")
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert source.stat().st_size == before.st_size
    result = run_batch(root, output, reports / "second")["results"][0]
    assert result["content_state"] == "modified"
    assert result["previous_sha256"] != result["sha256"]
    assert result["tables_rebuilt"] == 1
    _, directory, descriptor = read_export(output)
    assert pq.read_table(directory / descriptor["tables"][0]["file"]).column(0).to_pylist() == [9]
    ledger = json.loads((output / "source_registry.json").read_text())
    assert len(ledger["sources"][result["source_id"]]["versions"]) == 2
    schema = json.loads((Path(__file__).resolve().parents[2] / "schemas/source_registry.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER).validate(ledger)


def test_timestamp_change_only_keeps_content_cache(dataset):
    root, source, output, reports = dataset
    run_batch(root, output, reports / "first")
    stamp = source.stat()
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1000000))
    result = run_batch(root, output, reports / "second")["results"][0]
    assert result["status"] == "cached" and result["content_state"] == "unchanged"


def test_same_named_xsens_update_rebuilds_and_keeps_revision_history(dataset):
    root, _, output, reports = dataset
    source = root / "SANI/SANO_BA001/XSens/BA001_t1_task1_dx.xlsx"
    acquisition(source)
    run_batch(root, output, reports / "first", modalities=["xsens"])
    stamp = source.stat()
    book = openpyxl.load_workbook(source)
    book["Segment Velocity"]["B2"] = 9.0
    book.save(source)
    book.close()
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    report = run_batch(root, output, reports / "second", modalities=["xsens"])
    assert report["results"][0]["content_state"] == "modified"
    subject = output / "SANI/SANO_BA001"
    manifest = json.loads((subject / "manifest.json").read_text())
    target = subject / manifest["records"][0]["normalized_target"]
    descriptor = json.loads((target / "recording.json").read_text())
    assert report["results"][0]["tables_rebuilt"] == len(descriptor["tables"])
    table = next(t for t in descriptor["tables"] if t["source_sheet"] == "Segment Velocity")
    assert pq.read_table(target / table["file"]).column("Right Hand x")[0].as_py() == 9.0


def test_zip_disappears_and_reappears_without_rebuilding_same_csv(dataset):
    root, source, output, reports = dataset
    archive = source.parent / "export.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("folder/value.csv", source.read_bytes())
    first = run_batch(root, output, reports / "first")
    assert first["summary"]["tables_rebuilt"] == 1
    old_source_id = first["results"][0]["source_id"]
    archive.unlink()
    second = run_batch(root, output, reports / "second")
    assert second["results"][0]["status"] == "cached"
    assert second["results"][0]["tables_rebuilt"] == 0
    ledger = json.loads((output / "source_registry.json").read_text())
    assert ledger["sources"][old_source_id]["present"] is False
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as z:
        z.writestr("folder/value.csv", source.read_bytes())
    third = run_batch(root, output, reports / "third")
    assert third["results"][0]["status"] == "cached"
    assert third["summary"]["tables_rebuilt"] == 0
    assert third["summary"]["missing_previously_seen_sources"] == 0


def test_modified_archive_reuses_only_unchanged_member(dataset):
    root, source, output, reports = dataset
    source.unlink()
    archive = source.parent / "export.zip"
    with ZipFile(archive, "w", compression=ZIP_STORED) as z:
        z.writestr("first.csv", "Value\n1\n")
        z.writestr("second.csv", "Value\n2\n")
    run_batch(root, output, reports / "first")
    stamp = archive.stat()
    with ZipFile(archive, "w", compression=ZIP_STORED) as z:
        z.writestr("first.csv", "Value\n1\n")
        z.writestr("second.csv", "Value\n9\n")
    os.utime(archive, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert archive.stat().st_size == stamp.st_size
    result = run_batch(root, output, reports / "second")["results"][0]
    assert result["content_state"] == "modified"
    assert result["tables_rebuilt"] == result["tables_reused"] == 1


def test_refresh_preserves_valid_partial_export_metadata(dataset):
    root, source, output, _ = dataset
    archive = source.parent / "export.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("first.csv", "Value\n1\n")
        z.writestr("second.csv", "Value\n2\n")
    write_plan(make_plan(scan(root)), output)
    subject = output / "PAZIENTI/FG92"
    format_subject(subject, table_names=["first.csv"])
    report = write_plan(make_plan(scan(root)), output)
    manifest = json.loads((subject / "manifest.json").read_text())
    record = next(r for r in manifest["records"] if r["artifact_kind"] == "garmin_archive")
    assert record["formatter_report"]["complete_selected_tables"] is False
    assert record["formatter_status"] == "not_converted"
    assert record["source_id"] in report["retained_verified_formatter_sources"]


def test_corrupt_output_is_not_restored_as_converted(dataset):
    root, _, output, reports = dataset
    run_batch(root, output, reports / "first")
    _, directory, descriptor = read_export(output)
    (directory / descriptor["tables"][0]["file"]).write_bytes(b"damaged parquet")
    staging = write_plan(make_plan(scan(root)), output)
    record, _, _ = read_export(output)
    assert record["formatter_status"] == "not_converted"
    assert staging["retained_verified_formatter_sources"] == []
    assert run_batch(root, output, reports / "second")["results"][0]["tables_rebuilt"] == 1


def test_batch_continues_after_malformed_source(dataset):
    root, source, output, reports = dataset
    (source.parent / "bad.csv").write_text("A,B\n1,2\n3\n")
    report = run_batch(root, output, reports / "run")
    assert report["state"] == "completed_with_failures"
    assert report["summary"]["conversion_statuses"] == {"failed": 1, "converted": 1}
    assert (reports / "run/coverage.csv").exists()


def test_storage_guard_and_plan_mode_do_not_convert(dataset, monkeypatch):
    root, _, output, reports = dataset
    plan = run_batch(root, output, reports / "plan", plan_only=True)
    assert plan["state"] == "plan_only" and not output.exists()
    monkeypatch.setattr("predictor_import.batch.shutil.disk_usage", lambda _: SimpleNamespace(free=0))
    report = run_batch(root, output, reports / "run")
    assert report["state"] == "partial"
    assert not report["results"]
    assert not list(output.rglob("*.parquet"))
    schema = json.loads((Path(__file__).resolve().parents[2] / "schemas/source_registry.schema.json").read_text())
    Draft202012Validator(schema).validate(json.loads((output / "source_registry.json").read_text()))


def test_record_limit_reports_remaining_sources(dataset):
    root, source, output, reports = dataset
    (source.parent / "extra.csv").write_text("Value\n2\n")
    report = run_batch(root, output, reports / "run", max_records=1)
    assert report["state"] == "partial" and len(report["results"]) == 1
    assert report["selected_sources_this_run"] == 2
    again = run_batch(root, output, reports / "resumed", max_records=1)
    assert again["results"][0]["source_id"] != report["results"][0]["source_id"]
    assert again["results"][0]["status"] == "converted"
    assert again["coverage"][0]["converted_sources"] == 2


def test_batch_covers_all_cohorts_and_xsens_cache_is_preserved(dataset):
    root, _, output, reports = dataset
    acquisition(root / "SANI/SANO_BA001/XSens/BA001_t1_task1_dx.xlsx")
    acquisition(root / "CONTROLLO PAZIENTI/CTRLFG_91/XSENS/ctrlfg91_01_task1_t1.xlsx")
    report = run_batch(root, output, reports / "first")
    assert {r["participant_key"].split('/')[0] for r in report["results"]} == {"SANI", "PAZIENTI", "CONTROLLO PAZIENTI"}
    again = run_batch(root, output, reports / "second")
    assert again["summary"]["conversion_statuses"] == {"cached": 3}
    assert len(again["staging"]["retained_verified_formatter_sources"]) == 3


def test_unknown_subject_and_raw_output_are_rejected(dataset):
    root, _, output, reports = dataset
    with pytest.raises(ValueError, match="Unknown"):
        run_batch(root, output, reports / "run", subjects=["PAZIENTI/FG99"])
    with pytest.raises(ValueError, match="separate"):
        run_batch(root, root / "output", reports / "run")


def test_interruption_checkpoints_without_claiming_completion(dataset, monkeypatch):
    root, _, output, reports = dataset

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("predictor_import.garmin_formatter.format_subject", interrupt)
    report = run_batch(root, output, reports / "run")
    assert report["state"] == "partial" and report["stop_reason"] == "interrupted"
    assert report["summary"]["processed_sources"] == 0
    assert json.loads((reports / "run/batch_report.json").read_text())["state"] == "partial"
