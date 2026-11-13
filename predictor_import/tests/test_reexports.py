"""Declared re-exports (reexports.json) replace the first export of named visits."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from jsonschema import Draft202012Validator
import openpyxl
import pyarrow.parquet as pq
import pytest

from predictor_import import reexports
from predictor_import.adapter import make_plan, write_plan
from predictor_import.formatter import format_subject
from predictor_import.inspection import XSENS_SHEETS
from predictor_import.inventory import scan
from predictor_import.xlsx_repair import load_workbook, processing_quality

SCHEMAS = Path(__file__).resolve().parents[2] / "schemas"
RECORDED = datetime(2030, 1, 2, 12, 30)


def export(path: Path, *, quality="Normal", markers=(1, 2, 3, 4, 5), recorded=RECORDED, name="recording.mvn") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    book = openpyxl.Workbook()
    book.remove(book.active)
    for title in XSENS_SHEETS:
        sheet = book.create_sheet(title)
        if title == "General Information":
            for row in (["Frame Rate", 60], ["Recorded Date UTC", recorded], ["Processing Quality", quality],
                        ["Original File Name", name]):
                sheet.append(row)
        elif title == "Markers":
            sheet.append(["Frame", "Marker Text"])
            for frame in markers:
                sheet.append([frame, f"Marker_{frame}"])
        else:
            sheet.append(["Frame", "Right Hand x", "Left Hand x"])
            for frame in range(7):
                sheet.append([frame, 0.1 * frame + (0.01 if quality == "HD" else 0), 0.2 * frame])
    book.save(path)
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "Dati PREDICTOR"
    (root / "PAZIENTI").mkdir(parents=True)
    return root


def declare(monkeypatch, **extra):
    table = {"PAZIENTI/FG91": {"folder": "XSens_new", "timepoints": ["T1"], "decided_on": "2030-01-05", **extra}}
    monkeypatch.setattr(reexports, "REEXPORTS", table)


def files_by_path(inventory):
    return {f["source_path"].split("FG91/", 1)[1]: f for f in inventory["files"]}


def test_local_reexport_table_is_well_formed():
    # reexports.json is local and optional (git-ignored); check its shape only.
    table = json.loads(reexports.TABLE.read_text()) if reexports.TABLE.exists() else {}
    for key, entry in table.items():
        assert key.startswith("PAZIENTI/") and entry["folder"] and entry["timepoints"] and entry.get("reason")
        for path, marker in entry.get("markers_from_original", {}).items():
            assert path.startswith(f"{key}/{entry['folder']}/") and len(marker["frames"]) == 5
            assert len(marker["reexport_sha256"]) == len(marker["original_sha256"]) == 64


def test_without_a_table_the_reexport_folder_stays_ignored(root, monkeypatch):
    monkeypatch.setattr(reexports, "REEXPORTS", {})
    export(root / "PAZIENTI/FG91/Xsens/T1/FG91_00_TASK1_T1.xlsx", quality="HD")
    export(root / "PAZIENTI/FG91/XSens_new/FG91/T1/fg91_00_task1_t1.xlsx")
    files = files_by_path(scan(root))
    assert files["Xsens/T1/FG91_00_TASK1_T1.xlsx"]["disposition"] == "mapped"
    assert "export_role" not in files["Xsens/T1/FG91_00_TASK1_T1.xlsx"]
    reexport = files["XSens_new/FG91/T1/fg91_00_task1_t1.xlsx"]
    assert reexport["disposition"] == "excluded" and "modality_folder_conflict" in reexport["issues"]


def test_declared_reexport_replaces_original_and_hd_without_replacement_is_left_out(root, monkeypatch):
    declare(monkeypatch)
    export(root / "PAZIENTI/FG91/Xsens/T1/FG91_00_TASK1_T1.xlsx", quality="HD")
    export(root / "PAZIENTI/FG91/XSens_new/FG91/T1/fg91_00_task1_t1.xlsx")
    export(root / "PAZIENTI/FG91/Xsens/T1/FG91_00_B&B_T1.xlsx", quality="HD", markers=())      # HD, nothing replaces it
    export(root / "PAZIENTI/FG91/Xsens/T1/FG91_01_TASK1_T1.xlsx", quality="Live")             # Live, nothing replaces it
    export(root / "PAZIENTI/FG91/XSens_new/FG91/T3/fg91_00_task1_t3.xlsx")                     # undeclared visit
    export(root / "PAZIENTI/FG91/Xsens/T3/FG91_00_TASK1_T3.xlsx", quality="HD")                # undeclared visit
    inventory = scan(root)
    files = files_by_path(inventory)
    original, reexport = files["Xsens/T1/FG91_00_TASK1_T1.xlsx"], files["XSens_new/FG91/T1/fg91_00_task1_t1.xlsx"]
    assert (reexport["disposition"], reexport["export_role"], reexport["selection_reason"]) == (
        "mapped", "declared_reexport", "declared_reexport_replaces_original")
    assert reexport["replaces_source_ids"] == [original["source_id"]]
    assert (original["disposition"], original["selection_reason"], original["duplicate_of_source_id"]) == (
        "excluded", "superseded_by_declared_reexport", reexport["source_id"])
    hd = files["Xsens/T1/FG91_00_B&B_T1.xlsx"]
    assert (hd["disposition"], hd["selection_reason"], hd["processing_quality"]) == ("excluded", "hd_without_normal_reexport", "HD")
    live = files["Xsens/T1/FG91_01_TASK1_T1.xlsx"]
    assert (live["disposition"], live["processing_quality"]) == ("mapped", "Live")
    assert "modality_folder_conflict" in files["XSens_new/FG91/T3/fg91_00_task1_t3.xlsx"]["issues"]
    other = files["Xsens/T3/FG91_00_TASK1_T3.xlsx"]
    assert other["disposition"] == "mapped" and "export_role" not in other and "processing_quality" not in other
    Draft202012Validator(json.loads((SCHEMAS / "inventory.schema.json").read_text())).validate(inventory)
    plan = make_plan(inventory)
    manifest_schema = Draft202012Validator(json.loads((SCHEMAS / "manifest.schema.json").read_text()))
    for planned in plan["subjects"]:
        manifest_schema.validate(planned["manifest"])
    records = {r["source_path"]: r for r in plan["subjects"][0]["manifest"]["records"]}
    assert records[reexport["source_path"]]["normalized_target"] == "xsens/T1/task1_non_paretic"
    assert records[original["source_path"]]["normalized_target"] is None


def test_reviewed_markers_replace_the_reexported_ones_in_the_same_folder(root, monkeypatch, tmp_path):
    output = tmp_path / "standardized"
    original = export(root / "PAZIENTI/FG91/Xsens/T1/FG91_00_TASK1_T1.xlsx", quality="HD")
    reexport = export(root / "PAZIENTI/FG91/XSens_new/FG91/T1/fg91_00_task1_t1.xlsx", markers=(1, 3, 4, 5))
    monkeypatch.setattr(reexports, "REEXPORTS", {})
    write_plan(make_plan(scan(root)), output)
    format_subject(output / "PAZIENTI/FG91")
    folder = output / "PAZIENTI/FG91/xsens/T1/task1_non_paretic"
    assert json.loads((folder / "recording.json").read_text())["source_path"].endswith("FG91_00_TASK1_T1.xlsx")

    entry = {"reexport_sha256": sha(reexport), "original_source_path": "PAZIENTI/FG91/Xsens/T1/FG91_00_TASK1_T1.xlsx",
             "original_sha256": sha(original), "frames": [1, 2, 3, 4, 5], "verified_on": "2030-01-04", "check": "test"}
    declare(monkeypatch, markers_from_original={"PAZIENTI/FG91/XSens_new/FG91/T1/fg91_00_task1_t1.xlsx": entry})
    write_plan(make_plan(scan(root)), output)
    report = format_subject(output / "PAZIENTI/FG91")
    assert report["converted"] == 1 and report["failed"] == 0
    descriptor = json.loads((folder / "recording.json").read_text())
    assert descriptor["source_path"].endswith("fg91_00_task1_t1.xlsx")
    assert pq.read_table(folder / "markers.parquet").column("Frame").to_pylist() == [1, 2, 3, 4, 5]
    assert pq.read_table(folder / "markers_as_exported.parquet").column("Frame").to_pylist() == [1, 3, 4, 5]
    assert descriptor["observed_marker_count"] == 5 and "marker_count_mismatch" not in descriptor["warnings"]
    assert descriptor["marker_selection"]["valid_for_expected_repetitions"] is True
    assert descriptor["marker_override"]["markers_as_exported"] == [1, 3, 4, 5]
    assert descriptor["marker_override"]["original_sha256"] == sha(original)
    Draft202012Validator(json.loads((SCHEMAS / "recording.schema.json").read_text())).validate(descriptor)
    manifest = json.loads((output / "PAZIENTI/FG91/manifest.json").read_text())
    Draft202012Validator(json.loads((SCHEMAS / "manifest.schema.json").read_text())).validate(manifest)
    assert format_subject(output / "PAZIENTI/FG91")["cached"] == 1

    # A substitution that no longer matches the original export is refused, not applied.
    declare(monkeypatch, markers_from_original={"PAZIENTI/FG91/XSens_new/FG91/T1/fg91_00_task1_t1.xlsx": {**entry, "frames": [1, 2, 3, 4, 6]}})
    write_plan(make_plan(scan(root)), output)
    failed = format_subject(output / "PAZIENTI/FG91")
    assert failed["failed"] == 1 and "differ from the reviewed substitution" in failed["results"][0]["error"]


def test_other_recording_time_is_not_the_same_recording(root, monkeypatch, tmp_path):
    output = tmp_path / "standardized"
    original = export(root / "PAZIENTI/FG91/Xsens/T1/FG91_00_TASK1_T1.xlsx", quality="HD", recorded=datetime(2030, 1, 2, 12, 31))
    reexport = export(root / "PAZIENTI/FG91/XSens_new/FG91/T1/fg91_00_task1_t1.xlsx", markers=(1, 3, 4, 5))
    entry = {"reexport_sha256": sha(reexport), "original_source_path": "PAZIENTI/FG91/Xsens/T1/FG91_00_TASK1_T1.xlsx",
             "original_sha256": sha(original), "frames": [1, 2, 3, 4, 5]}
    declare(monkeypatch, markers_from_original={"PAZIENTI/FG91/XSens_new/FG91/T1/fg91_00_task1_t1.xlsx": entry})
    write_plan(make_plan(scan(root)), output)
    report = format_subject(output / "PAZIENTI/FG91")
    assert report["failed"] == 1 and "not the same recording" in report["results"][0]["error"]


SST = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def unescape_ampersands(path: Path, text: str, *, defect=lambda xml: xml.replace("&amp;", "&")) -> None:
    """Reproduce the MVN defect: move the cell holding `text` into shared strings, unescaped.

    openpyxl writes inline strings, while MVN writes xl/sharedStrings.xml; the bare '&'
    lives there in the real exports.
    """
    escaped = text.replace("&", "&amp;")
    with ZipFile(path) as archive:
        parts = {info.filename: (info, archive.read(info.filename)) for info in archive.infolist()}
    info, sheet = parts["xl/worksheets/sheet1.xml"]
    inline = f't="inlineStr"><is><t>{escaped}</t></is></c>'.encode()
    assert inline in sheet
    parts["xl/worksheets/sheet1.xml"] = (info, sheet.replace(inline, b't="s"><v>0</v></c>'))
    shared = f'<?xml version="1.0" encoding="UTF-8"?><sst xmlns="{SST}" count="1" uniqueCount="1"><si><t>{escaped}</t></si></sst>'
    info, rels = parts["xl/_rels/workbook.xml.rels"]
    parts["xl/_rels/workbook.xml.rels"] = (info, rels.replace(b"</Relationships>", (
        '<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" '
        'Target="sharedStrings.xml" Id="rIdShared" /></Relationships>').encode()))
    info, types = parts["[Content_Types].xml"]
    parts["[Content_Types].xml"] = (info, types.replace(b"</Types>", (
        '<Override PartName="/xl/sharedStrings.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml" /></Types>').encode()))
    with ZipFile(path, "w") as archive:
        for info, data in parts.values():
            archive.writestr(info, data)
        archive.writestr("xl/sharedStrings.xml", defect(shared).encode())


def test_bare_ampersand_in_shared_strings_is_repaired_in_memory_only(root, monkeypatch, tmp_path):
    path = export(root / "PAZIENTI/FG91/XSens_new/FG91/T1/fg91_00_B&B_t1.xlsx", name="C:\\x\\fg91_00_B&B_t1.mvn", markers=())
    unescape_ampersands(path, "C:\\x\\fg91_00_B&B_t1.mvn")
    before = sha(path)
    with pytest.raises(SyntaxError):
        openpyxl.load_workbook(path, read_only=True)
    book, repairs = load_workbook(path)
    rows = {row[0]: row[1] for row in book["General Information"].iter_rows(values_only=True)}
    book.close()
    assert rows["Original File Name"] == "C:\\x\\fg91_00_B&B_t1.mvn" and repairs[0]["replacements"] == 1
    assert processing_quality(path) == "Normal"
    declare(monkeypatch)
    output = tmp_path / "standardized"
    write_plan(make_plan(scan(root)), output)
    report = format_subject(output / "PAZIENTI/FG91")
    assert report["converted"] == 1
    descriptor = json.loads((output / "PAZIENTI/FG91/xsens/T1/box_and_blocks_non_paretic/recording.json").read_text())
    assert "source_text_repaired" in descriptor["warnings"] and descriptor["source_repairs"][0]["replacements"] == 1
    Draft202012Validator(json.loads((SCHEMAS / "recording.schema.json").read_text())).validate(descriptor)
    assert sha(path) == before


def test_other_xml_defects_are_not_repaired(tmp_path):
    path = export(tmp_path / "broken.xlsx", name="a&b.mvn")
    unescape_ampersands(path, "a&b.mvn", defect=lambda xml: xml.replace("</sst>", ""))
    with pytest.raises(SyntaxError):
        load_workbook(path)
    assert processing_quality(path) is None
