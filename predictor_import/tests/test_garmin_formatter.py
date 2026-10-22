from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from jsonschema import Draft202012Validator
import pyarrow.parquet as pq
import pytest

from predictor_import.adapter import make_plan, write_plan
from predictor_import.garmin_formatter import convert_table, format_subject
from predictor_import.inventory import scan
from predictor_import.__main__ import main


def csv_bytes(headers, rows, delimiter=","):
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter=delimiter)
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


@pytest.fixture
def staged(tmp_path):
    root = tmp_path / "raw"
    source = root / "PAZIENTI/FG91/Garmin/export.zip"
    source.parent.mkdir(parents=True)
    daily = csv_bytes(["User Id", "Start Time (Local)", "Start Time (UTC)", "Start Time (s)", "Time Zone (Local)", "Total Calories (kcal)", "Status"],
                      [["00123", "2030-04-09T00:00:00", "2030-04-08T22:00:00", "1901916000", "+0200", "1206215.675888896", "1"],
                       ["00123", "2030-04-10T00:00:00", "2030-04-09T22:00:00", "1902002400", "+0200", "", "NA"]])
    sleep = csv_bytes(["Time Zone (s)", "Sleep Level", "Time Zone (s)"], [["3600", "deep", "7200"]])
    with ZipFile(source, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("fg91/daily-health-log.csv", daily)
        archive.writestr("fg91/sleep.csv", sleep)
        archive.writestr("fg91/empty.csv", csv_bytes(["Value"], []))
    (source.parent / "daily-health-log.csv").write_bytes(daily)
    output = tmp_path / "standardized"
    write_plan(make_plan(scan(root)), output)
    return root, source, output / "PAZIENTI/FG91"


def export(subject):
    manifest = json.loads((subject / "manifest.json").read_text())
    record = next(r for r in manifest["records"] if r["artifact_kind"] == "garmin_archive")
    directory = subject / record["normalized_target"]
    return record, directory, json.loads((directory / "export.json").read_text())


def test_archive_roundtrip_keeps_extreme_values_and_all_time_representations(staged):
    _, source, subject = staged
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    report = format_subject(subject)
    assert report["converted"] == 1 and report["failed"] == 0
    record, directory, descriptor = export(subject)
    table = next(t for t in descriptor["tables"] if t["source_table"].endswith("daily-health-log.csv"))
    values = pq.read_table(directory / table["file"]).to_pydict()
    assert values["user_id"] == ["00123", "00123"]
    assert values["total_calories_kcal"] == [1206215.675888896, None]
    assert values["start_time_local"][0] == "2030-04-09T00:00:00"
    assert values["start_time_utc"][0] == "2030-04-08T22:00:00"
    assert values["time_zone_local"] == ["+0200", "+0200"]
    assert values["start_time_s"] == [1901916000, 1902002400]
    assert values["status"] == ["1", "NA"]
    assert table["columns"][5]["unit"] == "kcal"
    assert table["columns"][5]["null_count"] == 1
    assert descriptor["complete_selected_tables"]
    assert all(t["roundtrip_verified"] for t in descriptor["tables"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before == record["sha256"]
    assert record["formatter_status"] == "converted" and not record["ready_for_features"]
    base = Path(__file__).resolve().parents[2] / "schemas"
    Draft202012Validator(json.loads((base / "manifest.schema.json").read_text())).validate(json.loads((subject / "manifest.json").read_text()))
    Draft202012Validator(json.loads((base / "garmin_export.schema.json").read_text())).validate(descriptor)


def test_repeated_sleep_headers_have_distinct_positional_columns(staged):
    _, _, subject = staged
    format_subject(subject)
    _, directory, descriptor = export(subject)
    table = next(t for t in descriptor["tables"] if t["source_table"].endswith("sleep.csv"))
    assert table["source_headers"] == ["Time Zone (s)", "Sleep Level", "Time Zone (s)"]
    values = pq.read_table(directory / table["file"]).to_pydict()
    assert values["time_zone_s"] == [3600]
    assert values["time_zone_s__3"] == [7200]
    assert "repeated_source_headers_disambiguated" in table["warnings"]


def test_header_only_table_is_valid(staged):
    _, _, subject = staged
    format_subject(subject)
    _, directory, descriptor = export(subject)
    table = next(t for t in descriptor["tables"] if t["source_table"].endswith("empty.csv"))
    assert table["row_count"] == 0
    assert pq.read_table(directory / table["file"]).num_rows == 0


def test_partial_archive_resumes_and_never_claims_completion_early(staged):
    _, _, subject = staged
    format_subject(subject, table_names=["daily-health-log.csv"])
    record, directory, descriptor = export(subject)
    assert len(descriptor["tables"]) == 1 and not descriptor["complete_selected_tables"]
    assert record["formatter_status"] == "not_converted"
    table = descriptor["tables"][0]
    before = (directory / table["file"]).read_bytes()
    assert format_subject(subject, table_names=["daily-health-log.csv"])["cached"] == 1
    format_subject(subject, table_names=["sleep.csv"])
    assert len(export(subject)[2]["tables"]) == 2
    format_subject(subject)
    record, directory, descriptor = export(subject)
    assert descriptor["complete_selected_tables"] and record["formatter_status"] == "converted"
    assert (directory / table["file"]).read_bytes() == before
    assert format_subject(subject)["cached"] == 1


def test_corrupt_cache_is_rebuilt(staged):
    _, _, subject = staged
    format_subject(subject)
    _, directory, descriptor = export(subject)
    table = descriptor["tables"][0]
    (directory / table["file"]).write_bytes(b"bad parquet")
    assert format_subject(subject)["converted"] == 1
    assert format_subject(subject)["cached"] == 1


def test_changed_source_is_rejected_and_completed_outputs_survive(staged):
    _, source, subject = staged
    format_subject(subject)
    _, directory, _ = export(subject)
    before = (directory / "export.json").read_bytes()
    source.write_bytes(source.read_bytes() + b"changed")
    assert format_subject(subject)["failed"] == 1
    assert (directory / "export.json").read_bytes() == before


@pytest.mark.parametrize("value", ["0.123456789012345678901", "9223372036854775808", "NaN", "-Infinity", "0007", "1e-400"])
def test_unrepresentable_and_special_values_are_preserved_as_text(tmp_path, value):
    payload = csv_bytes(["Measurement"], [[value]])
    path = tmp_path / "table.parquet"
    descriptor = convert_table(lambda: io.BytesIO(payload), ["Measurement"], path, "table.csv")
    assert pq.read_table(path).column(0).to_pylist() == [value]
    assert descriptor["columns"][0]["dtype"] == "string"


def test_late_type_change_and_large_mixed_integer_do_not_lose_values(tmp_path):
    payload = csv_bytes(["Measurement", "Comment"], [["9007199254740993", "a,b\nquoted"], ["0.5", "è"]], delimiter=";")
    path = tmp_path / "table.parquet"
    descriptor = convert_table(lambda: io.BytesIO(payload), ["Measurement", "Comment"], path, "table.csv")
    assert descriptor["csv_delimiter"] == ";"
    assert pq.read_table(path).to_pydict() == {"measurement": ["9007199254740993", "0.5"], "comment": ["a,b\nquoted", "è"]}


def test_loose_table_conversion_and_cli(staged):
    root, _, subject = staged
    csv_path = root / "PAZIENTI/FG91/Garmin/extra.csv"
    csv_path.write_bytes(csv_bytes(["Steps"], [["10"], ["20"]]))
    write_plan(make_plan(scan(root)), subject.parents[1])
    manifest = json.loads((subject / "manifest.json").read_text())
    record = next(r for r in manifest["records"] if r["source_path"].endswith("extra.csv"))
    assert format_subject(subject, source_ids=[record["source_id"]])["failed"] == 0
    directory = subject / record["normalized_target"]
    assert pq.read_table(directory / "extra.parquet").column(0).to_pylist() == [10, 20]


def test_failed_table_never_publishes_partial_output(staged):
    root, source, subject = staged
    with ZipFile(source, "w") as archive:
        archive.writestr("bad.csv", b"A,B\n1,2\n3\n")
    write_plan(make_plan(scan(root)), subject.parents[1])
    manifest = json.loads((subject / "manifest.json").read_text())
    source_id = next(r["source_id"] for r in manifest["records"] if r["artifact_kind"] == "garmin_archive")
    assert format_subject(subject, source_ids=[source_id])["failed"] == 1
    assert not list(subject.rglob("*.parquet"))
    assert not list(subject.rglob(".format-*"))


def test_unknown_table_preserves_previously_completed_output(staged):
    _, _, subject = staged
    format_subject(subject)
    _, directory, _ = export(subject)
    before = (directory / "export.json").read_bytes()
    assert format_subject(subject, table_names=["absent.csv"])["failed"] == 1
    assert (directory / "export.json").read_bytes() == before


def test_unknown_source_id_fails_before_writing(staged):
    _, _, subject = staged
    with pytest.raises(ValueError, match="source ID"):
        format_subject(subject, source_ids=["unknown"])


def test_manifest_cannot_redirect_output_into_raw_sources(staged):
    _, source, subject = staged
    path = subject / "manifest.json"
    manifest = json.loads(path.read_text())
    record = next(r for r in manifest["records"] if r["artifact_kind"] == "garmin_archive")
    record["normalized_target"] = str(source.parent / "new_output")
    path.write_text(json.dumps(manifest))
    assert format_subject(subject)["failed"] == 1
    assert not (source.parent / "new_output").exists()


def test_manifest_cannot_replace_entire_subject_folder(staged):
    _, _, subject = staged
    path = subject / "manifest.json"
    manifest = json.loads(path.read_text())
    record = next(r for r in manifest["records"] if r["artifact_kind"] == "garmin_archive")
    record["normalized_target"] = "."
    path.write_text(json.dumps(manifest))
    assert format_subject(subject)["failed"] == 1
    assert (subject / "subject.json").exists()


def test_inventory_and_formatter_support_semicolon_multiline_csv(staged):
    root, _, subject = staged
    path = root / "PAZIENTI/FG91/Garmin/extra.csv"
    path.write_bytes(csv_bytes(["Steps", "Comment"], [["10", "a,b\nquoted"], ["20", "è"]], delimiter=";"))
    inventory = scan(root)
    file = next(f for f in inventory["files"] if f["source_path"].endswith("extra.csv"))
    assert file["columns"] == ["Steps", "Comment"]
    write_plan(make_plan(inventory), subject.parents[1])
    assert format_subject(subject, source_ids=[file["source_id"]])["converted"] == 1


def test_corrupt_zip_payload_is_rejected_even_if_header_is_readable(staged):
    root, source, subject = staged
    with ZipFile(source, "w") as archive:
        archive.writestr("broken.csv", b"Value\n" + b"1\n" * 50000)
    with ZipFile(source) as archive:
        info = archive.getinfo("broken.csv")
        body_offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    damaged = bytearray(source.read_bytes())
    damaged[body_offset + 80000] = ord("2")
    source.write_bytes(damaged)
    write_plan(make_plan(scan(root)), subject.parents[1])
    manifest = json.loads((subject / "manifest.json").read_text())
    source_id = next(r["source_id"] for r in manifest["records"] if r["artifact_kind"] == "garmin_archive")
    assert format_subject(subject, source_ids=[source_id])["failed"] == 1
    assert not list(subject.rglob("*.parquet"))
