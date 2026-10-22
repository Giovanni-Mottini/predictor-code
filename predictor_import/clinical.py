"""Lossless clinical-cell extraction from the inspected TIMELINE_01 layout.

No score interpretation, date repair, scale conversion or inferred assessment day.
Only the assessment section is read; identifying header cells are not exported.
"""
from __future__ import annotations

from datetime import date, datetime
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from shared_code.reader import DataContractError, contained_path
from .inspection import read_patient_metadata

METHOD_ID = "clinical-source-cells-v1"
ACTIVITIES = {"task n4 xsens", "motore xsens", "b&b xsens", "motore training", "garmin"}
LABEL = re.compile(r"T\d+(?:_D\d+)?", re.I)
NUMBER = re.compile(r"[+-]?\d+(?:[.,]\d+)?")
SIDE_NUMBER = re.compile(r"([+-]?\d+(?:[.,]\d+)?)\s*\(([LR])\)", re.I)


def source_value(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def value_parts(value, *, bilateral=False):
    """Keep ordinal/text evidence; only explicit complete scalars become numeric."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return [(None, None, "not_recorded")]
    if isinstance(value, bool):
        return [(None, None, "uninterpreted_text")]
    if isinstance(value, (int, float)) and math.isfinite(value):
        return [(None, float(value), "numeric_source_value")]
    text = str(value).strip()
    if NUMBER.fullmatch(text):
        return [(None, float(text.replace(",", ".")), "numeric_source_value")]
    if bilateral:
        matches = SIDE_NUMBER.findall(text)
        sides = [s.upper() for _, s in matches]
        if matches and len(sides) == len(set(sides)):
            residue = SIDE_NUMBER.sub("", text).replace("/", "").strip()
            status = "explicit_side_value" if len(sides) == 2 and not residue else "partial_side_value"
            return [("left" if s.upper() == "L" else "right", float(v.replace(",", ".")), status)
                    for v, s in matches]
    if text.casefold() in {"o", "x"}:
        return [(None, None, "source_marker")]
    if re.fullmatch(r"\d+\+", text):
        return [(None, None, "ordinal_category")]
    return [(None, None, "uninterpreted_text")]


def extract_workbook(reader, record):
    """Return observation, calendar, issue and input-provenance records."""
    path = contained_path(Path(reader.manifest["source_root"]), record["source_path"])
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if record.get("sha256") and record["sha256"] != digest:
        raise DataContractError("clinical_source_changed", f"Source changed: {record['source_path']}")
    expected = reader.subject["subject_id"].removesuffix("_DROPOUT")
    identity = read_patient_metadata(path, expected)
    if identity.get("status") != "available":
        raise DataContractError("clinical_identity_unverified", f"{reader.participant_key}: {identity.get('status')}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise DataContractError("clinical_source_changed", "Clinical workbook changed during identity check")
    values = load_workbook(BytesIO(payload), read_only=True, data_only=True)
    formulas = load_workbook(BytesIO(payload), read_only=True, data_only=False)
    observations, calendar, issues = [], [], []
    provenance = {"path": str(path), "sha256": digest, "source_id": record["source_id"]}
    common = {"participant_key": reader.participant_key, "source_id": record["source_id"],
              "source_path": record["source_path"], "source_sha256": digest, "sheet": "TIMELINE_01"}
    try:
        if "TIMELINE_01" not in values.sheetnames:
            raise DataContractError("unsupported_clinical_layout", "TIMELINE_01 absent")
        rows = list(values["TIMELINE_01"].iter_rows())
        formula_rows = list(formulas["TIMELINE_01"].iter_rows())
        headers = [(i, [(j, str(c.value).strip().upper()) for j, c in enumerate(row)
                         if isinstance(c.value, str) and LABEL.fullmatch(c.value.strip())])
                   for i, row in enumerate(rows[:8])]
        headers = [(i, h) for i, h in headers if h]
        if len(headers) != 1:
            raise DataContractError("unsupported_clinical_layout", "Expected one timeline header row")
        header, columns = headers[0]
        if len({label for _, label in columns}) != len(columns):
            raise DataContractError("duplicate_clinical_label", "Repeated timeline column")
        dates = {}
        previous = None
        for col, label in columns:
            cell = rows[header + 1][col]
            value = cell.value
            exact = value.date().isoformat() if isinstance(value, datetime) else value.isoformat() if isinstance(value, date) else None
            dates[label] = exact
            event = {**common, "event_kind": "clinical_day", "source_label": label,
                     "timepoint": label.split("_D")[0], "protocol_day": int(label.split("_D")[1]) if "_D" in label else None,
                     "clinical_date": exact, "source_cell": f"{get_column_letter(col+1)}{header+2}",
                     "raw_value_json": json.dumps(source_value(value), ensure_ascii=False),
                     "date_status": "source_date" if exact else "unresolved_source_date"}
            calendar.append(event)
            if exact is None:
                issues.append({**event, "code": "clinical_date_missing_or_text", "details": "Original value retained; no date repair"})
            elif previous and exact < previous:
                issues.append({**event, "code": "clinical_dates_not_chronological", "details": f"Previous dated source column: {previous}; neither value changed"})
            if exact:
                previous = exact
            canonical = reader.subject["clinical_metadata"].get("timeline_dates", {}).get(label)
            if canonical != exact:
                issues.append({**event, "code": "clinical_metadata_date_disagrees", "details": f"subject.json date: {canonical}"})
        instrument = None
        in_activities = False
        for idx in range(header + 2, len(rows)):
            row = rows[idx]
            label = str(row[0].value).strip() if row and row[0].value is not None else ""
            if not label:
                continue
            activity = label.casefold() in ACTIVITIES
            if activity:
                in_activities = True
            if in_activities and not activity:
                # Notes are retained as evidence, not interpreted as measurements or dates.
                calendar.append({**common, "event_kind": "source_note", "source_cell": row[0].coordinate,
                                 "raw_value_json": json.dumps(label, ensure_ascii=False), "date_status": "unparsed_note"})
                continue
            if not label.startswith("-"):
                instrument = label
            measure = f"{instrument} / {label.lstrip('- ').strip()}" if label.startswith("-") else label
            for col, source_label in columns:
                cell = row[col] if col < len(row) else None
                value = cell.value if cell is not None else None
                fcell = formula_rows[idx][col] if idx < len(formula_rows) and col < len(formula_rows[idx]) else None
                formula = fcell.value if fcell is not None and fcell.data_type == "f" else None
                cell_name = f"{get_column_letter(col+1)}{idx+1}"
                base = {**common, "source_cell": cell_name, "source_label": source_label,
                        "timepoint": source_label.split("_D")[0],
                        "protocol_day": int(source_label.split("_D")[1]) if "_D" in source_label else None,
                        "clinical_date": dates[source_label], "instrument": instrument,
                        "measure": measure, "raw_value_json": json.dumps(source_value(value), ensure_ascii=False),
                        "source_data_type": getattr(cell, "data_type", None), "formula": formula,
                        "method_id": METHOD_ID}
                if activity:
                    if value is not None:
                        calendar.append({**base, "event_kind": "protocol_activity", "date_status": "source_column_date" if dates[source_label] else "unresolved_source_date"})
                    continue
                for side, numeric, status in value_parts(value, bilateral=instrument == "Box and Blocks"):
                    if formula and value is None:
                        status = "formula_cache_missing"
                    token = f"{reader.participant_key}|{record['source_id']}|{cell_name}|{side}"
                    observations.append({**base, "observation_id": "clinical_"+hashlib.sha256(token.encode()).hexdigest()[:24],
                                         "anatomical_side": side, "value": numeric, "value_status": status,
                                         "unit": None, "instrument_version": None, "interpretation_status": "source_preserved_not_clinically_validated"})
                    if status in {"uninterpreted_text", "partial_side_value", "formula_cache_missing"}:
                        issues.append({**base, "code": "clinical_"+status, "details": "Preserved original cell; no guessed scalar/side/date"})
    finally:
        values.close()
        formulas.close()
    return observations, calendar, issues, provenance
