"""Garmin readable names, partial exports and byte-preserving cache migration."""
import hashlib
import json
from zipfile import ZipFile

import pyarrow.parquet as pq
import pytest

from predictor_import.adapter import atomic_json, make_plan, write_plan
from predictor_import.garmin_formatter import format_subject
from predictor_import.inventory import scan
from predictor_import.layout import migrate_layout, migrate_subject
from predictor_import.storage_names import garmin_export_name, garmin_table_names, garmin_targets
from predictor_import.tests.test_garmin_formatter import export, staged
from shared_code.reader import SubjectReader


def snapshot(folder):
    return {str(p.relative_to(folder)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns, p.stat().st_ino)
            for p in folder.rglob("*") if p.is_file()}


def as_legacy(subject):
    manifest = json.loads((subject / "manifest.json").read_text())
    record = next(r for r in manifest["records"] if r["artifact_kind"] == "garmin_archive")
    directory = subject / record["normalized_target"]
    descriptor = json.loads((directory / "export.json").read_text())
    for table in descriptor["tables"]:
        index = descriptor["selected_source_tables"].index(table["source_table"])
        filename = f"table_{index:02d}.parquet"
        (directory / table["file"]).rename(directory / filename)
        table["file"] = filename
        cache = subject / "garmin/.table_cache" / descriptor["formatter_version"] / table["source_csv_sha256"] / "table.json"
        cached = json.loads(cache.read_text())
        cached["table"]["file"] = filename
        atomic_json(cache, cached)
    atomic_json(directory / "export.json", descriptor)
    relative = f"garmin/{record['source_id']}"
    directory.rename(subject / relative)
    record["normalized_target"] = relative
    record["formatter_report"]["descriptor_path"] = relative + "/export.json"
    atomic_json(subject / "manifest.json", manifest)
    return subject / relative


@pytest.fixture
def legacy(staged):
    root, source, subject = staged
    assert format_subject(subject)["converted"] == 1
    old = as_legacy(subject)
    return root, source, subject, old


@pytest.mark.parametrize("path,kind,expected", [
    ("PAZIENTI/FG91/Garmin/fg91.zip", "garmin_archive", "export"),
    ("SANI/SANO_FG91/Garmin/FG91.zip", "garmin_archive", "export"),
    ("PAZIENTI/FG91/Garmin/Export May.zip", "garmin_archive", "export_may"),
    ("SANI/SANO_BA001/Garmin/BA001_30.04.2030-14.05.2030.csv.zip", "garmin_archive", "export_2030-04-30_to_2030-05-14"),
    ("SANI/SANO_BA001/Garmin/BA001_07.02.2030-26.02.2030.csv/daily-health-log.csv", "garmin_table", "table_daily_health_log_2030-02-07_to_2030-02-26"),
    ("PAZIENTI/FG91/Garmin/bbi-(raw).csv", "garmin_table", "table_bbi_raw"),
])
def test_export_names_use_explicit_labels_without_inventing_dates(path, kind, expected):
    assert garmin_export_name({"source_path": path, "artifact_kind": kind}) == expected


def test_same_period_exports_get_deterministic_source_suffixes():
    records = [{"source_path": f"SANI/SANO_BA001/Garmin/{label}_30.04.2030-14.05.2030.zip",
                "artifact_kind": "garmin_archive", "disposition": "mapped", "source_id": "src_" + label * 24}
               for label in ["a", "b"]]
    assert set(garmin_targets(records).values()) == {
        "garmin/export_2030-04-30_to_2030-05-14__aaaaaaaa",
        "garmin/export_2030-04-30_to_2030-05-14__bbbbbbbb",
    }
    assert garmin_targets(records) == garmin_targets(list(reversed(records)))


def test_csv_basename_collisions_are_stable_for_whole_selected_set():
    names = ["a/sleep.csv", "b/sleep.csv", "bbi-(raw).csv", "daily-health-log.csv", "custom-value.csv", "custom_value.csv"]
    actual = garmin_table_names(names)
    assert actual["bbi-(raw).csv"] == "bbi_raw.parquet"
    assert actual["daily-health-log.csv"] == "daily_health_log.parquet"
    assert actual["a/sleep.csv"] == "sleep__01.parquet"
    assert actual["b/sleep.csv"] == "sleep__02.parquet"
    assert len(set(actual.values())) == len(names)
    assert actual == garmin_table_names(list(reversed(names)))


def test_migration_keeps_table_bytes_cache_links_source_ids_and_raw_files(legacy):
    _, source, subject, old = legacy
    record_before, _, descriptor_before = export(subject)
    raw_before = snapshot(source.parent)
    cache_before = snapshot(subject / "garmin/.table_cache")
    old_tables = {t["source_table"]: (t["sha256"], (old / t["file"]).stat().st_mtime_ns, (old / t["file"]).stat().st_ino)
                  for t in descriptor_before["tables"]}
    report = migrate_subject(subject, modality="garmin")
    assert report["exports_migrated"] == 1 and report["tables_rebuilt"] == 0
    assert not old.exists()
    record, directory, descriptor = export(subject)
    assert record["normalized_target"] == "garmin/export"
    assert record["source_id"] == record_before["source_id"]
    assert record["recording_id"] == record_before["recording_id"]
    assert descriptor["selected_source_tables"] == descriptor_before["selected_source_tables"]
    for table in descriptor["tables"]:
        file = directory / table["file"]
        assert old_tables[table["source_table"]] == (hashlib.sha256(file.read_bytes()).hexdigest(), file.stat().st_mtime_ns, file.stat().st_ino)
    sleep = next(t for t in descriptor["tables"] if t["source_table"].endswith("sleep.csv"))
    assert sleep["file"] == "sleep.parquet"
    daily = next(t for t in descriptor["tables"] if t["source_table"].endswith("daily-health-log.csv"))
    values = pq.read_table(directory / daily["file"])
    assert values["total_calories_kcal"].to_pylist() == [1206215.675888896, None]
    assert snapshot(source.parent) == raw_before
    assert snapshot(subject / "garmin/.table_cache") == cache_before
    assert format_subject(subject)["results"][0]["tables_rebuilt"] == 0
    current = snapshot(subject)
    assert migrate_layout(subject, modality="garmin")["records_changed"] == 0
    assert snapshot(subject) == current


def test_dry_run_and_failed_publication_preserve_originals_and_cache(legacy, monkeypatch):
    _, _, subject, _ = legacy
    before = snapshot(subject)
    assert migrate_subject(subject, modality="garmin", dry_run=True)["exports_migrated"] == 1
    assert snapshot(subject) == before
    def fail(path, value):
        if path == subject / "manifest.json":
            raise OSError("simulated manifest failure")
        atomic_json(path, value)
    monkeypatch.setattr("predictor_import.adapter.atomic_json", fail)
    with pytest.raises(OSError, match="manifest failure"):
        migrate_subject(subject, modality="garmin")
    assert snapshot(subject) == before
    assert not list(subject.rglob(".layout-*"))


@pytest.mark.parametrize("entrypoint", ["adapter", "formatter"])
def test_normal_entrypoints_reuse_legacy_conversions(legacy, entrypoint):
    root, _, subject, old = legacy
    if entrypoint == "adapter":
        assert write_plan(make_plan(scan(root)), subject.parents[1])["retained_verified_formatter_sources"]
    result = format_subject(subject)
    assert result["cached"] == 1 and result["converted"] == result["failed"] == 0
    assert result["results"][0]["tables_rebuilt"] == 0
    assert not old.exists()


def test_partial_export_collision_names_survive_migration_and_resumption(staged):
    root, source, subject = staged
    (source.parent / "daily-health-log.csv").unlink()
    with ZipFile(source, "w") as archive:
        archive.writestr("a/sleep-log.csv", "Value\n1\n")
        archive.writestr("b/sleep_log.csv", "Value\n9\n")
    write_plan(make_plan(scan(root)), subject.parents[1])
    assert format_subject(subject, table_names=["b/sleep_log.csv"])["failed"] == 0
    old = as_legacy(subject)
    assert migrate_subject(subject, modality="garmin")["exports_migrated"] == 1
    record, directory, descriptor = export(subject)
    assert not descriptor["complete_selected_tables"]
    assert record["formatter_status"] == "not_converted"
    assert descriptor["tables"][0]["file"] == "sleep_log__02.parquet"
    stamp = (directory / "sleep_log__02.parquet").stat().st_mtime_ns
    assert write_plan(make_plan(scan(root)), subject.parents[1])["retained_verified_formatter_sources"]
    result = format_subject(subject)
    assert result["results"][0]["tables_rebuilt"] == 1
    assert pq.read_table(directory / "sleep_log__01.parquet").column(0).to_pylist() == [1]
    assert pq.read_table(directory / "sleep_log__02.parquet").column(0).to_pylist() == [9]
    assert (directory / "sleep_log__02.parquet").stat().st_mtime_ns == stamp
    assert export(subject)[2]["complete_selected_tables"]
    assert not old.exists()


@pytest.mark.parametrize("problem", ["corrupt_table", "occupied_destination", "unexpected_file"])
def test_migration_refuses_bad_outputs_without_changing_files(legacy, problem):
    _, _, subject, old = legacy
    if problem == "corrupt_table":
        (old / "table_00.parquet").write_bytes(b"damaged")
    elif problem == "occupied_destination":
        destination = subject / "garmin/export"
        destination.mkdir()
        (destination / "export.json").write_text("{}")
    else:
        (old / "notes.txt").write_text("preserve this")
    before = snapshot(subject)
    with pytest.raises((ValueError, KeyError)):
        migrate_subject(subject, modality="garmin")
    assert snapshot(subject) == before


def test_repacked_renamed_source_reuses_content_cache_after_migration(legacy):
    root, source, subject, _ = legacy
    migrate_subject(subject, modality="garmin")
    cache_before = snapshot(subject / "garmin/.table_cache")
    source.rename(source.with_name("repacked.zip"))
    write_plan(make_plan(scan(root)), subject.parents[1])
    result = format_subject(subject)
    assert result["cached"] == 1 and result["results"][0]["tables_rebuilt"] == 0
    assert result["results"][0]["tables_reused"] == 3
    assert snapshot(subject / "garmin/.table_cache") == cache_before
    assert export(subject)[0]["normalized_target"] == "garmin/export_repacked"


def test_formatter_refuses_to_overwrite_another_source_at_same_readable_path(legacy):
    root, source, subject, _ = legacy
    migrate_subject(subject, modality="garmin")
    directory = subject / "garmin/export"
    before = snapshot(directory)
    source.rename(source.with_suffix(".ZIP"))
    write_plan(make_plan(scan(root)), subject.parents[1])
    result = format_subject(subject)
    assert result["failed"] == 1
    assert "different source" in result["results"][0]["error"]
    assert snapshot(directory) == before
