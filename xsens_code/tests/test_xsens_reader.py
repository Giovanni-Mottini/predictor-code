from __future__ import annotations

import hashlib
import json
from pathlib import Path

import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from predictor_import.adapter import atomic_json, make_plan, write_plan
from predictor_import.formatter import format_subject
from predictor_import.inspection import XSENS_SHEETS
from predictor_import.inventory import scan
from xsens_code.repetitions import audit_subject
from xsens_code.__main__ import main
from shared_code.reader import DataContractError, SubjectReader
from xsens_code.xsens import build_windows, frame_axis, inspect_recording


@pytest.fixture
def factory(tmp_path):
    def create(*, markers=None, frame_start=100, rate=60, cohort="SANI", subject_id="SANO_FG91"):
        raw = tmp_path / "raw"
        folder = raw / cohort / subject_id / "XSens"
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / ("FG91_t1_task1_dx.xlsx" if cohort == "SANI" else "FG91_00_TASK1_T1.xlsx")
        book = openpyxl.Workbook()
        book.remove(book.active)
        for name in XSENS_SHEETS:
            sheet = book.create_sheet(name)
            if name == "General Information":
                sheet.append(["Frame Rate", rate])
            elif name == "Markers":
                sheet.append(["Frame", "Marker Text"])
                for i, frame in enumerate(markers if markers is not None else range(frame_start + 1, frame_start + 6)):
                    sheet.append([frame, f"event{i}"])
                # Blank styled rows must not become marker events.
                sheet["A12"].number_format = "0"
            else:
                sheet.append(["Frame", "Right Hand x", "Left Hand x"])
                for index in range(7):
                    sheet.append([frame_start + index, index / 10, -index / 10])
        book.save(source)
        output = tmp_path / "standardized"
        write_plan(make_plan(scan(raw)), output)
        subject = output / cohort / subject_id
        assert format_subject(subject)["converted"] == 1
        return subject
    return create


def opened(subject):
    reader = SubjectReader(subject)
    return reader.open_xsens(reader.select(modality="xsens")[0])


def rewrite_table(subject, sheet, transform):
    recording = opened(subject)
    desc = recording.table_descriptor(sheet)
    path = recording.descriptor_path.parent / desc["file"]
    table = transform(pq.read_table(path))
    pq.write_table(table, path)
    desc["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    desc["row_count"] = table.num_rows
    for c in desc["columns"]:
        c["null_count"] = table[c["name"]].null_count
    atomic_json(recording.descriptor_path, recording.descriptor)


def replace_column(table, name, values):
    return table.set_column(table.column_names.index(name), table.schema.field(name), pa.array(values, type=table[name].type))


def test_label_mapping_half_open_and_duration(factory):
    result = inspect_recording(opened(factory()))
    first, last = result["windows"][0], result["windows"][-1]
    assert first["start_frame"] == first["first_sample_frame"] == 101
    assert first["end_frame_exclusive"] == 102
    assert (first["row_start"], first["row_end_exclusive"], first["n_samples"]) == (1, 2, 1)
    assert first["duration_s"] == 1 / 60 and first["sample_span_s"] == 0
    assert last["last_sample_frame"] == 104
    assert not result["frame_axis"]["frame_equals_row_index"]
    assert result["full_signal_values_finite"] and not result["ready_for_features"]
    assert len(result["provenance"]["checked_tables"]) == 6


def test_final_boundary_one_past_last_and_slicing():
    table = pa.table({"Frame": range(100, 108), "value": range(8)})
    windows = build_windows(frame_axis(table, "test"), [100, 102, 104, 106, 108], 40)
    assert [w.slice(table)["Frame"].to_pylist() for w in windows] == [[100, 101], [102, 103], [104, 105], [106, 107]]
    assert sum(w.n_samples for w in windows) == 8
    assert windows[-1].duration_s == 2 / 40 and windows[-1].sample_span_s == 1 / 40


@pytest.mark.parametrize("rate", [None, 0, -1, True, float("nan"), float("inf")])
def test_invalid_sampling_rate_never_defaults(rate):
    with pytest.raises(DataContractError) as err:
        build_windows(list(range(8)), [0, 2, 4, 6, 8], rate)
    assert err.value.code == "sampling_rate_invalid"


@pytest.mark.parametrize("boundaries,code", [
    ([100, 101, 102, 103], "boundary_count_or_type"),
    ([100, 101, None, 103, 104], "boundary_count_or_type"),
    ([100, 101, 102.5, 103, 104], "boundary_count_or_type"),
    ([100, 101, 101, 103, 104], "boundary_order_invalid"),
    ([100, 102, 101, 103, 104], "boundary_order_invalid"),
    ([99, 101, 102, 103, 104], "boundary_out_of_bounds"),
    ([100, 101, 102, 103, 109], "boundary_out_of_bounds"),
])
def test_invalid_boundaries_are_not_repaired(boundaries, code):
    original = list(boundaries)
    with pytest.raises(DataContractError) as err:
        build_windows(list(range(100, 108)), boundaries, 60)
    assert err.value.code == code and boundaries == original


@pytest.mark.parametrize("frames,code", [
    ([100, 101, 101, 103, 104, 105, 106], "frame_order_invalid"),
    ([100, 102, 101, 103, 104, 105, 106], "frame_order_invalid"),
    ([100, 101, 102, 103, 104, 105, 107], "frame_gap"),
    ([100, 101, None, 103, 104, 105, 106], "frame_missing"),
])
def test_bad_frame_axes_block_record_with_code(factory, frames, code):
    subject = factory()
    rewrite_table(subject, "Segment Velocity", lambda t: replace_column(t, "Frame", frames))
    result = audit_subject(SubjectReader(subject))
    assert result["failed_recordings"] == 1 and result["repetitions"] == 0
    assert result["results"][0]["flags"][0]["code"] == code


def test_cross_sheet_alignment_is_required(factory):
    subject = factory()
    rewrite_table(subject, "Segment Position", lambda t: replace_column(t, "Frame", range(101, 108)))
    with pytest.raises(DataContractError, match="same frame labels"):
        inspect_recording(opened(subject))


def test_six_markers_are_invalid_and_original_events_are_retained(factory):
    subject = factory(markers=[100, 101, 102, 103, 104, 105])
    recording = opened(subject)
    before = recording.read_table("Markers").to_pydict()
    with pytest.raises(DataContractError) as exc:
        inspect_recording(recording)
    assert exc.value.code == "marker_count_invalid"
    assert recording.descriptor["marker_selection"]["ignored_marker_frames"] == []
    # A stale descriptor from the retired temporary policy cannot re-enable it.
    recording.descriptor["marker_selection"].update(boundary_frames=[101,102,103,104,105], valid_for_expected_repetitions=True)
    with pytest.raises(DataContractError) as exc:
        inspect_recording(recording)
    assert exc.value.code == "marker_count_invalid"
    assert recording.read_table("Markers").to_pydict() == before


def test_marker_selection_is_required_and_checked(factory):
    subject = factory()
    recording = opened(subject)
    desc = recording.descriptor
    original = desc.pop("marker_selection")
    atomic_json(recording.descriptor_path, desc)
    with pytest.raises(DataContractError) as err:
        inspect_recording(opened(subject))
    assert err.value.code == "marker_selection_missing"
    original["boundary_frames"][0] = 100
    desc["marker_selection"] = original
    atomic_json(recording.descriptor_path, desc)
    with pytest.raises(DataContractError) as err:
        inspect_recording(opened(subject))
    assert err.value.code == "marker_selection_mismatch"


def test_other_marker_counts_and_out_of_range_are_not_repaired(factory):
    subject = factory(markers=[101, 102, 103, 104, 105, 106, 107])
    result = audit_subject(SubjectReader(subject))
    assert result["results"][0]["flags"][0]["code"] == "marker_count_invalid"
    subject = factory(markers=[101, 102, 103, 104, 108])
    result = audit_subject(SubjectReader(subject))
    assert result["results"][0]["flags"][0]["code"] == "boundary_out_of_bounds"


def test_missing_and_nonfinite_values_preserved_with_window_counts(factory):
    subject = factory()
    rewrite_table(subject, "Joint Angles ZXY", lambda t: replace_column(t, "Right Hand x", [None, 1, float("inf"), float("nan"), 4, 5, 6]))
    recording = opened(subject)
    result = inspect_recording(recording)
    missing, nonfinite = result["numerical_findings"]
    assert missing["first_source_excel_row"] == 2 and missing["first_frame"] == 100
    assert missing["repetition_counts"] == {"1": 0, "2": 0, "3": 0, "4": 0}
    assert nonfinite["count"] == 2 and nonfinite["first_frame"] == 102
    assert nonfinite["repetition_counts"] == {"1": 0, "2": 1, "3": 1, "4": 0}
    assert not result["full_signal_values_finite"]
    assert result["repetition_values_finite"] == {"1": True, "2": False, "3": False, "4": True}
    assert recording.read_table("Joint Angles ZXY")["Right Hand x"][0].as_py() is None
    assert recording.read_table("Joint Angles ZXY")["Right Hand x"][2].as_py() == float("inf")


def test_rate_comes_from_source_without_60hz_fallback(factory):
    subject = factory(rate=120)
    result = inspect_recording(opened(subject))
    assert result["sampling_rate_hz"] == 120 and result["windows"][0]["duration_s"] == 1 / 120
    recording = opened(subject)
    recording.descriptor["sampling_rate_hz"] = None
    atomic_json(recording.descriptor_path, recording.descriptor)
    with pytest.raises(DataContractError) as err:
        inspect_recording(opened(subject))
    assert err.value.code == "sampling_rate_invalid"


def test_descriptor_rate_mismatch(factory):
    subject = factory()
    recording = opened(subject)
    recording.descriptor["sampling_rate_hz"] = 30
    atomic_json(recording.descriptor_path, recording.descriptor)
    with pytest.raises(DataContractError) as err:
        inspect_recording(opened(subject))
    assert err.value.code == "sampling_rate_mismatch"


def test_same_suffix_cohorts_stay_separate_and_selection_uses_metadata(factory):
    healthy = factory()
    patient = factory(cohort="PAZIENTI", subject_id="FG91")
    h, p = SubjectReader(healthy), SubjectReader(patient)
    assert h.participant_key == "SANI/SANO_FG91" and p.participant_key == "PAZIENTI/FG91"
    assert len(h.select(modality="xsens", sessions=["T1"], tasks=[1], anatomical_sides=["right"])) == 1
    assert h.select(modality="xsens", sessions=["T2"]) == []
    assert h.select(modality="xsens", tasks=[]) == []
    assert p.select(modality="xsens", limb_conditions=["paretic"]) == []
    recording = opened(healthy)
    manifest = h.manifest
    manifest["records"][0]["source_path"] = "unparseable-original-name.xlsx"
    recording.descriptor["source_path"] = "unparseable-original-name.xlsx"
    atomic_json(healthy / "manifest.json", manifest)
    atomic_json(recording.descriptor_path, recording.descriptor)
    assert audit_subject(SubjectReader(healthy))["segmented_recordings"] == 1


def test_corruption_and_revision_mismatch_are_reported(factory):
    subject = factory()
    recording = opened(subject)
    table = recording.table_descriptor("Segment Velocity")
    (recording.descriptor_path.parent / table["file"]).write_bytes(b"corrupted")
    result = audit_subject(SubjectReader(subject))
    assert result["results"][0]["flags"][0]["code"] == "table_checksum_mismatch"
    recording.descriptor["source_sha256"] = "a" * 64
    atomic_json(recording.descriptor_path, recording.descriptor)
    result = audit_subject(SubjectReader(subject))
    assert result["results"][0]["flags"][0]["code"] == "source_revision_mismatch"


def test_pending_conversion_does_not_disappear(factory):
    subject = factory()
    reader = SubjectReader(subject)
    reader.manifest["records"][0]["formatter_status"] = "not_converted"
    atomic_json(subject / "manifest.json", reader.manifest)
    result = audit_subject(SubjectReader(subject))
    assert result["selected_recordings"] == result["failed_recordings"] == 1
    assert result["results"][0]["flags"][0]["code"] == "record_not_converted"


def test_cli_reports_and_exit_codes_and_protects_inputs(factory, tmp_path):
    subject = factory()
    output = tmp_path / "reports"
    assert main(["xsens-repetitions", str(subject), "--output", str(output)]) == 0
    assert len((output / "repetitions.csv").read_text().splitlines()) == 5
    report = json.loads((output / "metadata.json").read_text())
    assert not report["ready_for_features"] and report["subjects"][0]["repetitions"] == 4
    assert main(["xsens-repetitions", str(subject), "--session", "T2", "--output", str(output)]) == 2
    assert len((output / "repetitions.csv").read_text().splitlines()) == 1
    with pytest.raises(SystemExit) as err:
        main(["xsens-repetitions", str(subject), "--output", str(subject / "report")])
    assert err.value.code == 2 and not (subject / "report").exists()
    raw = SubjectReader(subject).manifest["source_root"]
    with pytest.raises(SystemExit):
        main(["xsens-repetitions", str(subject), "--output", str(Path(raw) / "report")])
    recording = opened(subject)
    (recording.descriptor_path.parent / recording.table_descriptor("Markers")["file"]).write_bytes(b"bad")
    assert main(["xsens-repetitions", str(subject), "--output", str(output)]) == 1


def test_identity_path_and_duplicate_manifest_rejected(factory, tmp_path):
    subject = factory()
    reader = SubjectReader(subject)
    reader.manifest["participant_key"] = "PAZIENTI/FG91"
    atomic_json(subject / "manifest.json", reader.manifest)
    with pytest.raises(DataContractError, match="identities disagree"):
        SubjectReader(subject)
    reader.manifest["participant_key"] = reader.participant_key
    reader.manifest["records"][0]["formatter_report"]["descriptor_path"] = "../outside/recording.json"
    atomic_json(subject / "manifest.json", reader.manifest)
    result = audit_subject(SubjectReader(subject))
    assert result["results"][0]["flags"][0]["code"] == "unsafe_path"
    reader.manifest["records"].append(reader.manifest["records"][0])
    atomic_json(subject / "manifest.json", reader.manifest)
    with pytest.raises(DataContractError, match="unique"):
        SubjectReader(subject)


def test_report_symlink_cannot_overwrite_raw_source(factory, tmp_path):
    subject = factory()
    source = Path(SubjectReader(subject).manifest["source_root"]) / opened(subject).record["source_path"]
    original = source.read_bytes()
    output = tmp_path / "reports"
    output.mkdir()
    (output / "metadata.json").symlink_to(source)
    assert main(["xsens-repetitions", str(subject), "--output", str(output)]) == 0
    assert source.read_bytes() == original and not (output / "metadata.json").is_symlink()


def test_table_symlink_cannot_escape_recording(factory, tmp_path):
    subject = factory()
    recording = opened(subject)
    path = recording.descriptor_path.parent / recording.table_descriptor("Segment Velocity")["file"]
    copy = tmp_path / "outside.parquet"
    copy.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(copy)
    with pytest.raises(DataContractError) as err:
        recording.read_table("Segment Velocity")
    assert err.value.code == "unsafe_path"
