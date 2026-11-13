"""Fugl-Meyer and Box and Blocks per visit, from each patient's clinical workbook.

Reads only the visit-label row, the date row under it and the scale rows (never the header cells
that identify a person). One workbook per patient: <workbooks>/<ID>/<name>.xlsx.

Pairing (folder-label rule): workbook T0 -> Xsens T1; T5, T6, ... -> the same label. Full scales are
monthly, so T1-T4 (weekly) carry no Fugl-Meyer.
  FMA   the first row starting with "Fugl" (Fugl-Meyer A-D), a plain number
  BBT   the first row starting with "Box", the paretic hand's value from cells like "0 (L) / 24 (R)";
        at T0, when the cell has no side, the T1_d1 cell is used (flagged)
A visit is kept when it has at least one of the two scores.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import re

import openpyxl
import pandas as pd

LABEL = re.compile(r"T\d+(_d\d+)?$", re.I)
MISSING = {"", "o", "x", "na", "?", "none"}
SCALES = {"FMA": r"^Fugl", "BBT": r"^Box"}
HAND = {"left": "L", "right": "R"}


def workbook_visit(column: str) -> str | None:
    number = re.fullmatch(r"T(\d+)", column)
    if not number:
        return None
    n = int(number.group(1))
    return "T1" if n == 0 else f"T{n}" if n >= 5 else None


def plain_number(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.lower() in MISSING:
        return None
    match = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*(?:/\s*100)?", text)
    return float(match.group(1).replace(",", ".")) if match else None


def sided(value) -> dict[str, float | None]:
    """'0 (L) / 24 (R)', "58'' (R), 46'' (L)", '0(L)/' -> {'L': ..., 'R': ...}; '?' -> None."""
    out = {}
    for number, side in re.findall(r"(\d+(?:[.,]\d+)?|\?)\s*(?:''|\"|'')?\s*\(\s*([LR])\s*\)", str(value)):
        out[side] = None if number == "?" else float(number.replace(",", "."))
    lone = re.match(r"^\s*(\d+)\s*\(\s*([LR])\s*\)\s*/\s*$", str(value))
    if lone:
        out[lone.group(2)] = float(lone.group(1))
    return out


def workbook_of(folder: Path) -> Path:
    found = [p for p in folder.glob("*.xlsx") if not p.name.startswith(("~", "."))]
    if len(found) != 1:
        raise ValueError(f"{folder.name}: expected one clinical workbook, found {len(found)}")
    return found[0]


def read_workbook(path: Path) -> tuple[dict, dict]:
    """Visit dates and the raw scale cells, by workbook column label."""
    sheet = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    rows = [list(r) for r in sheet.iter_rows(values_only=True)]
    header = next((i for i, r in enumerate(rows)
                   if sum(isinstance(c, str) and bool(LABEL.fullmatch(c.strip())) for c in r) > 5), None)
    if header is None:
        raise ValueError(f"{path.name}: no row of visit labels")
    columns = {j: str(c).strip().upper().replace("_D", "_d") for j, c in enumerate(rows[header])
               if isinstance(c, str) and LABEL.fullmatch(c.strip())}
    dates = {}
    for j, label in columns.items():
        value = rows[header + 1][j] if header + 1 < len(rows) and j < len(rows[header + 1]) else None
        dates[label] = (value.date().isoformat() if isinstance(value, datetime)
                        else value.isoformat() if isinstance(value, date) else None)
    raw = {}
    for row in rows[header + 2:]:
        name = str(row[0]).strip() if row and row[0] is not None else ""
        for key, pattern in SCALES.items():
            if name and re.search(pattern, name, re.I) and key not in raw:
                raw[key] = {label: (row[j] if j < len(row) else None) for j, label in columns.items()}
    return dates, raw


def read_scales(workbooks: Path, sides: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """One row per patient and paired visit (patients in code order, visits in number order)."""
    records, notes = [], []
    for patient in sorted(sides):
        folder = Path(workbooks) / patient
        if not folder.is_dir():
            notes.append(f"{patient}: no workbook folder, left out")
            continue
        dates, raw = read_workbook(workbook_of(folder))
        hand = HAND[sides[patient]]
        fma, bbt = raw.get("FMA", {}), raw.get("BBT", {})
        visits = sorted(((workbook_visit(c), c) for c in dates if workbook_visit(c)), key=lambda v: int(v[0][1:]))
        for visit, column in visits:
            flags = []
            cell = bbt.get(column)
            if column == "T0" and not sided(cell) and sided(bbt.get("T1_d1")):
                flags.append("BBT from T1_d1 (T0 cell has no side)")
                cell = bbt.get("T1_d1")
            box = sided(cell)
            if cell not in (None, "") and not box and plain_number(cell) is None and str(cell).strip().lower() not in MISSING:
                flags.append("BBT unparsed")
            record = {"patient": patient, "visit": visit, "workbook_column": column, "workbook_date": dates.get(column),
                      "FMA": plain_number(fma.get(column)), "BBT": box.get(hand), "flags": "; ".join(flags)}
            if record["FMA"] is not None or record["BBT"] is not None:
                records.append(record)
    frame = pd.DataFrame(records, columns=["patient", "visit", "workbook_column", "workbook_date", "FMA", "BBT", "flags"])
    return frame.astype({"FMA": float, "BBT": float}), notes
