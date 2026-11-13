"""Patient deliverables from verified Xsens feature tables (layout of "Features Example").

OUTPUT/
  Features_Xsens_Task_1-4/Pazienti/<ID>/Xsens_tasks/Xsens_tasks.xlsx
      tasks 1-4 per repetition; complete visits only (32 rows, every feature filled)
  Features_Xsens_Task_1-4/Pazienti_Paretic_AVG/<ID>/Xsens_tasks/Xsens_tasks.xlsx
      per complete visit and task: equal-weight mean of the 4 paretic repetitions
  features b&b/<id>_b&b.csv
      paretic Box and Blocks, one row per visit (whole recording)
  features XSens Motore/<id>.csv
      paretic Xsens on MOTORE, one row per visit and task (1 = mot1, 2 = mot2)

OUTPUT_SPEED_LENGTH/ (next to OUTPUT, same layout and columns) is the same export with
the four vel_mean_* divided by the recording's lever length (lever lengths per second);
the other 11 features are identical.

Values are copied or averaged, never recomputed or rounded; every file is read
back and checked, and an existing file is never replaced.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path
import statistics
import tempfile

import openpyxl
from openpyxl.styles import Alignment, Font

from predictor_import.markers import MARKER_POLICY_ID
from shared_code.reader import DataContractError
from .xsens import WHOLE_RECORDING_POLICY_ID
from .xsens_features import FEATURE_NAMES, LEVER_FIELDS, METHOD_ID, SEGMENTS, SPEED_LENGTH_POLICY_ID

TASKS_ROOT, COMPLETE_FOLDER, AVERAGE_FOLDER = "Features_Xsens_Task_1-4", "Pazienti", "Pazienti_Paretic_AVG"
BB_FOLDER, MOTORE_FOLDER = "features b&b", "features XSens Motore"
HEADERS = ("subject", "task", "repetition (1-4)", "arm (dx-sx)",
           "healthy/paretic", "number of samples", *FEATURE_NAMES)
AVG_HEADERS = ("subject", "task", "arm (dx-sx)", "healthy/paretic", *FEATURE_NAMES)
CSV_FIELDS = ("subject", "task", "timepoint", "arto", "arto_eff", "label", "n_samples", "duration_s",
              *FEATURE_NAMES, "n_degenerate", "degenerate_features")
CSV_TASK = {"box_and_blocks": "B&B", "mot1": 1, "mot2": 2}
CSV_KIND = {"box_and_blocks": "bb", "mot1": "motore", "mot2": "motore"}
EXPECTED_ROWS = {(task, rep, arm) for task in range(1, 5) for rep in range(1, 5) for arm in ("dx", "sx")}
# A blank DTW only because the other arm's recording is missing or unusable is not a degenerate feature.
PARTNER_REASONS = {"dtw_reference_missing", "dtw_reference_skipped", "dtw_reference_failed"}
SIDE = {"right": "dx", "left": "sx"}
TOLERANCE = 1e-12
SPEED_LENGTH_SUFFIX = "_SPEED_LENGTH"
SPEED_LEVERS = {f"vel_mean_{role}": f"lever_{role}_m" for role in SEGMENTS}


def speed_length_output(output: Path) -> Path:
    """The length-normalized folder sits next to OUTPUT: <name>_SPEED_LENGTH."""
    output = output.resolve()
    return output.with_name(output.name + SPEED_LENGTH_SUFFIX)


def destinations(output: Path, participant_key: str) -> dict[str, Path]:
    parts = participant_key.split("/")
    if len(parts) != 2 or parts[0] != "PAZIENTI" or parts[1] in {"", ".", ".."} or "\\" in parts[1]:
        raise DataContractError("export_identity_invalid", "Patient export requires PAZIENTI/<subject>")
    subject, root = parts[1], output.resolve()
    paths = {"tasks": root / TASKS_ROOT / COMPLETE_FOLDER / subject / "Xsens_tasks" / "Xsens_tasks.xlsx",
             "tasks_avg": root / TASKS_ROOT / AVERAGE_FOLDER / subject / "Xsens_tasks" / "Xsens_tasks.xlsx",
             "bb": root / BB_FOLDER / f"{subject.lower()}_b&b.csv",
             "motore": root / MOTORE_FOLDER / f"{subject.lower()}.csv"}
    if any(not path.resolve().is_relative_to(root) for path in paths.values()):
        raise DataContractError("unsafe_output", "Destination escapes output root")
    return paths


def _verified(directory: Path, metadata: dict, name: str) -> str:
    payload = (directory / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata["artifacts"][name]["sha256"]:
        raise DataContractError("export_checksum_mismatch", f"{name} checksum differs from extraction metadata")
    return payload.decode("utf-8")


def _number(text: str) -> float | None:
    value = float(text) if text != "" else None
    if value is not None and not math.isfinite(value):
        raise DataContractError("export_nonfinite_value", "Feature table contains a populated nonfinite value")
    return value


def _per_lever(row: dict, values: list, seen: dict) -> list:
    """vel_mean_* divided by the recording's lever; each value checked: times the lever gives back m/s."""
    levers = tuple(_number(row[SPEED_LEVERS[name]]) for name in SPEED_LEVERS)
    if seen.setdefault(row["source_id"], levers) != levers:
        raise DataContractError("speed_lever_inconsistent", "One recording has two different lever lengths")
    normalized = list(values)
    for name, lever in zip(SPEED_LEVERS, levers):
        index = FEATURE_NAMES.index(name)
        speed = values[index]
        if speed is None or lever is None or not lever > 0:
            normalized[index] = None  # no lever: only this normalized speed is blank
            continue
        normalized[index] = speed / lever
        if not math.isclose(normalized[index] * lever, speed, rel_tol=TOLERANCE, abs_tol=0.0):
            raise DataContractError("speed_length_check_failed", "Normalized speed times lever is not the m/s speed")
    return normalized


def _read_rows(directory: Path, speed_length: bool = False):
    """Task rows per participant and visit (all visits), and the paretic whole-recording CSV rows.

    With ``speed_length`` the four speeds are divided by the recording's lever length."""
    metadata = json.loads((directory / "metadata.json").read_text())
    method = metadata["method"]
    if method["method_id"] != METHOD_ID or method["rom"]["marker_policy_id"] != MARKER_POLICY_ID:
        raise DataContractError("export_method_mismatch", "Export requires the current quaternion elbow method")
    overrides = {s["participant_key"]: s.get("side_override") for s in metadata["subjects"]}
    partner_missing = {f["source_id"] for f in csv.DictReader(io.StringIO(_verified(directory, metadata, "qc_flags.csv"), newline=""))
                       if f["feature"] == "" and f["code"] in PARTNER_REASONS}
    groups, whole, seen, levers = {}, {}, set(), {}
    with io.StringIO(_verified(directory, metadata, "features.csv"), newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"participant_key", "timepoint", "task", "repetition", "anatomical_side", "limb_condition",
                    "n_samples", "method_id", "method_sha256", "marker_policy_id", *FEATURE_NAMES}
        if speed_length:
            if method.get("speed_length", {}).get("policy_id") != SPEED_LENGTH_POLICY_ID:
                raise DataContractError("export_method_mismatch", "Feature table has no lever lengths for the speed export")
            required |= {"source_id", *LEVER_FIELDS}
        if not required <= set(reader.fieldnames or []):
            raise DataContractError("export_columns_missing", "Feature CSV does not have the required columns")
        for row in reader:
            is_whole = row.get("window_kind") == "whole_recording"  # Older tables have task rows only.
            if (row["method_id"] != METHOD_ID or row["method_sha256"] != method["method_sha256"]
                    or row["marker_policy_id"] != (WHOLE_RECORDING_POLICY_ID if is_whole else MARKER_POLICY_ID)):
                raise DataContractError("export_method_mismatch", "Feature rows contain mixed methods")
            participant = row["participant_key"]
            destinations(Path("."), participant)  # validates the cohort-qualified identity
            subject = participant.split("/")[1]
            visit = row["timepoint"] or "unlabelled"
            arm = SIDE.get(row["anatomical_side"])
            condition = {"non_paretic": "healthy", "paretic": "paretic"}.get(row["limb_condition"])
            samples = int(row["n_samples"])
            if samples <= 0 or arm is None or condition is None:
                raise DataContractError("export_row_invalid", "Invalid side, condition or sample count")
            values = [_number(row[name]) for name in FEATURE_NAMES]
            if speed_length:
                values = _per_lever(row, values, levers)
            if is_whole:
                number = re.fullmatch(r"T(\d+)", visit)
                if row.get("activity") not in CSV_TASK or row["task"] != "" or row["repetition"] != "" or not number:
                    raise DataContractError("export_row_invalid", "Invalid B&B/mot recording, visit or repetition")
                key = (participant, visit, row["activity"], arm, condition)
                if key in seen:
                    raise DataContractError("export_duplicate_row", "Ambiguous repeated acquisition in a visit")
                seen.add(key)
                if condition != "paretic":
                    continue  # The CSVs hold the paretic arm only; both arms stay in features.csv.
                override = overrides.get(participant)
                degenerate = [name for name, value in zip(FEATURE_NAMES, values) if value is None
                              and not (name == "dtw_hand" and row["source_id"] in partner_missing)]
                whole.setdefault(participant, {}).setdefault(CSV_KIND[row["activity"]], []).append({
                    "subject": subject, "task": CSV_TASK[row["activity"]], "timepoint": int(number.group(1)),
                    "arto": SIDE[override["workbook_paretic_side"]] if override else arm, "arto_eff": arm,
                    "label": "malato", "n_samples": samples, "duration_s": _number(row["duration_s"]),
                    **dict(zip(FEATURE_NAMES, values)),
                    "n_degenerate": len(degenerate), "degenerate_features": ";".join(degenerate) or "none"})
                continue
            if len(visit) > 31 or any(c in visit for c in "\\/*?:[]") or visit.startswith("'") or visit.endswith("'"):
                raise DataContractError("export_sheet_invalid", "Unsupported visit worksheet name")
            task, repetition = int(row["task"]), int(row["repetition"])
            if task not in {1, 2, 3, 4} or repetition not in {1, 2, 3, 4}:
                raise DataContractError("export_row_invalid", "Invalid task or repetition")
            key = (participant, visit, task, repetition, arm, condition)
            if key in seen:
                raise DataContractError("export_duplicate_row", "Ambiguous repeated acquisition in a visit worksheet")
            seen.add(key)
            groups.setdefault(participant, {}).setdefault(visit, []).append(
                [subject, task, repetition, arm, condition, samples, *values])
    if not seen:
        raise DataContractError("export_empty", "No patient feature rows to export")
    for visits in groups.values():
        # Excel worksheet names are case-insensitive.
        if len({name.casefold() for name in visits}) != len(visits):
            raise DataContractError("export_sheet_collision", "Visit names collide in Excel")
        for rows in visits.values():
            rows.sort(key=lambda r: (r[1], r[3], r[2]))
    for kinds in whole.values():
        for rows in kinds.values():
            rows.sort(key=lambda r: (r["timepoint"], str(r["task"])))
    return groups, whole


def visit_problems(rows: list[list]) -> list[str]:
    """Empty when all 32 task rows exist with every feature filled; otherwise why not."""
    problems = []
    missing = len(EXPECTED_ROWS - {(r[1], r[2], r[3]) for r in rows})
    if missing:
        problems.append(f"{missing} of 32 rows missing")
    blank = sum(v is None for r in rows for v in r[6:])
    if blank:
        problems.append(f"{blank} blank feature cells")
    return problems


def paretic_means(subject: str, visit: str, rows: list[list]) -> list[list]:
    paretic = [r for r in rows if r[4] == "paretic"]
    if len(paretic) != 16 or len({r[3] for r in paretic}) != 1:
        raise DataContractError("export_average_invalid", f"Expected 16 paretic rows on one side: {subject}/{visit}")
    averaged = []
    for task in range(1, 5):
        reps = sorted((r for r in paretic if r[1] == task), key=lambda r: r[2])
        if [r[2] for r in reps] != [1, 2, 3, 4]:
            raise DataContractError("export_average_invalid", f"Incomplete paretic group: {subject}/{visit}/task{task}")
        # Equal weight per repetition; correctly rounded sum, no output rounding.
        averaged.append([subject, task, reps[0][3], "paretic",
                         *(math.fsum(r[6 + i] for r in reps) / 4 for i in range(len(FEATURE_NAMES)))])
    return averaged


def _sheet(book, name, headers, rows, identity, integers):
    sheet = book.create_sheet(name)
    sheet.append(headers)
    for row in rows:
        sheet.append(row)
    sheet.freeze_panes = "A2"
    sheet.sheet_view.showGridLines = True
    sheet.row_dimensions[1].height = 32
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for column in range(1, len(headers) + 1):
        sheet.column_dimensions[openpyxl.utils.get_column_letter(column)].width = 18 if 2 <= column <= identity else 24
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            if isinstance(cell.value, str):
                cell.data_type = "s"
            elif cell.column in integers:
                cell.number_format = "0"
            elif cell.column > identity:
                cell.number_format = "0.000000"


def _publish(destination: Path, write) -> None:
    """Write next to the destination, check, then link: atomic and never replaces a file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".xsens-", suffix=destination.suffix, dir=destination.parent)
    os.close(fd)
    try:
        write(Path(temporary))
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_sheets(path: Path) -> dict[str, list[tuple]]:
    book = openpyxl.load_workbook(path, read_only=True, data_only=False)
    try:
        return {name: list(book[name].values) for name in book.sheetnames}
    finally:
        book.close()


def _verify_tasks(path: Path, visits: dict):
    written = _read_sheets(path)
    if list(written) != list(visits):
        raise DataContractError("export_roundtrip_failed", "Workbook worksheet order changed")
    for name, expected_rows in visits.items():
        actual = written[name]
        if tuple(actual[0]) != HEADERS or len(actual) != len(expected_rows) + 1:
            raise DataContractError("export_roundtrip_failed", "Workbook header or row count differs")
        for actual_row, expected in zip(actual[1:], expected_rows):
            if list(actual_row[:6]) != expected[:6]:
                raise DataContractError("export_roundtrip_failed", "Workbook row identity differs")
            for got, value in zip(actual_row[6:], expected[6:]):
                equal = got is None if value is None else isinstance(got, (int, float)) and math.isclose(got, value, rel_tol=1e-13, abs_tol=1e-10)
                if not equal:
                    raise DataContractError("export_roundtrip_failed", "Workbook feature value differs")


def _verify_average(path: Path, visits: dict):
    """Independent re-read and recomputation with statistics.mean."""
    written = _read_sheets(path)
    if list(written) != list(visits):
        raise DataContractError("export_roundtrip_failed", "AVG worksheet order changed")
    for name, rows in visits.items():
        actual = written[name]
        if tuple(actual[0]) != AVG_HEADERS or len(actual) != 5:
            raise DataContractError("export_roundtrip_failed", f"AVG sheet shape differs: {name}")
        for task, row in zip(range(1, 5), actual[1:]):
            reps = [r for r in rows if r[4] == "paretic" and r[1] == task]
            if list(row[:4]) != [reps[0][0], task, reps[0][3], "paretic"]:
                raise DataContractError("export_roundtrip_failed", f"AVG identity differs: {name}/task{task}")
            for index, value in enumerate(row[4:]):
                expected = statistics.mean(r[6 + index] for r in reps)
                if not isinstance(value, float) or not math.isclose(value, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
                    raise DataContractError("export_roundtrip_failed", f"AVG value differs: {name}/task{task}")


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)  # floats in full precision (shortest exact representation)
    with path.open(newline="", encoding="utf-8") as stream:
        back = list(csv.DictReader(stream))
    if len(back) != len(rows):
        raise DataContractError("export_roundtrip_failed", "CSV row count differs")
    for got, expected in zip(back, rows):
        for field in CSV_FIELDS:
            value = expected[field]
            same = (got[field] == "" if value is None else float(got[field]) == value if isinstance(value, float)
                    else got[field] == str(value))
            if not same:
                raise DataContractError("export_roundtrip_failed", f"CSV value differs: {field}")


def _load(feature_directories: list[Path], speed_length: bool):
    tasks, whole = {}, {}
    for directory in feature_directories:
        task_groups, whole_groups = _read_rows(directory, speed_length)
        if (set(task_groups) | set(whole_groups)) & (set(tasks) | set(whole)):
            raise DataContractError("export_duplicate_subject", "Supply one feature directory per patient")
        tasks.update(task_groups)
        whole.update(whole_groups)
    natural = lambda item: [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", item[0])]
    complete, incomplete = {}, []
    for participant, visits in tasks.items():
        for visit, rows in sorted(visits.items(), key=natural):
            problems = visit_problems(rows)
            if problems:
                incomplete.append({"participant_key": participant, "visit": visit, "reasons": problems})
            else:
                complete.setdefault(participant, {})[visit] = rows
    plan = [(participant, "tasks") for participant in complete] + [(participant, "tasks_avg") for participant in complete]
    plan += [(participant, kind) for participant, kinds in whole.items() for kind in kinds]
    return complete, whole, incomplete, plan


def _write_tree(output: Path, complete: dict, whole: dict, plan: list, paths: dict) -> list[dict]:
    files = []
    for participant, kind in plan:
        destination = paths[participant, kind]
        if kind in {"tasks", "tasks_avg"}:
            visits = complete[participant]
            if kind == "tasks_avg":
                # Average the values exactly as stored in the published Pazienti workbook (written first).
                written = _read_sheets(paths[participant, "tasks"])
                visits = {name: [list(row) for row in written[name][1:]] for name in visits}
            book = openpyxl.Workbook()
            book.remove(book.active)
            for name, rows in visits.items():
                if kind == "tasks":
                    _sheet(book, name, HEADERS, rows, 6, {2, 3, 6})
                else:
                    _sheet(book, name, AVG_HEADERS, paretic_means(rows[0][0], name, rows), 4, {2})
            check = _verify_tasks if kind == "tasks" else _verify_average
            def write(path, book=book, visits=visits, check=check):
                book.save(path)
                check(path, visits)
            try:
                _publish(destination, write)
            finally:
                book.close()
            count = sum(len(rows) for rows in visits.values()) if kind == "tasks" else 4 * len(visits)
            detail = {"visits": list(visits), "rows": count}
        else:
            rows = whole[participant][kind]
            _publish(destination, lambda path, rows=rows: _write_csv(path, rows))
            detail = {"visits": sorted({f"T{r['timepoint']}" for r in rows}, key=lambda v: int(v[1:])), "rows": len(rows)}
        files.append({"participant_key": participant, "kind": kind, "path": str(destination.relative_to(output.resolve())),
                      **detail, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    return files


def export_outputs(feature_directories: list[Path], output: Path, speed_length: Path | None = None) -> dict:
    """Write the four deliverable kinds into ``output`` and, if given, the length-normalized copy into
    ``speed_length``. Both trees are read, checked and preflighted before the first file is written."""
    trees = [(output, False)] + ([(speed_length, True)] if speed_length is not None else [])
    if speed_length is not None:
        a, b = output.resolve(), speed_length.resolve()
        if a == b or a.is_relative_to(b) or b.is_relative_to(a):
            raise DataContractError("unsafe_output", "The two output folders must be separate trees")
    prepared = []
    for root, normalized in trees:
        complete, whole, incomplete, plan = _load(feature_directories, normalized)
        paths = {(participant, kind): destinations(root, participant)[kind] for participant, kind in plan}
        prepared.append((root, complete, whole, incomplete, plan, paths))
    for *_, paths in prepared:
        for path in paths.values():
            if path.exists():
                raise DataContractError("output_exists", f"Output already exists: {path}")
    results = [(root, _write_tree(root, complete, whole, plan, paths), incomplete)
               for root, complete, whole, incomplete, plan, paths in prepared]
    exported = {"files": results[0][1], "incomplete_task_visits": results[0][2],
                "rules": {"tasks": "visit kept only with all 32 rows (4 tasks x 2 arms x 4 repetitions) and every feature filled",
                          "tasks_avg": "equal-weight mean of the four paretic repetitions per task",
                          "csv": "paretic arm only; task B&B, or 1 = mot1 and 2 = mot2; full-precision values; "
                                 "arto = workbook side, arto_eff = side used (they differ only with a side override)"}}
    if speed_length is not None:
        exported["speed_length"] = {
            "output": str(results[1][0].resolve()), "policy_id": SPEED_LENGTH_POLICY_ID,
            "rule": "same files, rows and columns; vel_mean_* divided by the recording's lever length (lever lengths "
                    "per second); every value checked: times the lever gives back the m/s speed; other features identical",
            "files": results[1][1], "incomplete_task_visits": results[1][2]}
    return exported
