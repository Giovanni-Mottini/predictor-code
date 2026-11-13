"""Open Xsens workbooks whose shared strings hold a bare '&'.

MVN writes the original file name into the shared strings without escaping it, so a
recording saved as e.g. "..._B&B_..." yields malformed XML and openpyxl refuses the
workbook. Only that case is repaired: if xl/sharedStrings.xml does not parse, every '&'
that does not start an XML entity is escaped, the part must then parse, and every other
part of the package is copied byte for byte. The source file is never changed; the
repair happens in memory and is reported to the caller.
"""
from __future__ import annotations

import io
import re
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

import openpyxl

PART = "xl/sharedStrings.xml"
BARE_AMPERSAND = re.compile(rb"&(?!(?:amp|lt|gt|quot|apos|#[0-9]+|#x[0-9A-Fa-f]+);)")
RULE = "bare ampersand in shared strings escaped as &amp; (in memory; source unchanged)"


def repaired_copy(path) -> tuple[io.BytesIO, dict] | None:
    """In-memory package with the shared strings repaired, or None if not this defect."""
    with ZipFile(path) as archive:
        if PART not in archive.namelist():
            return None
        payload = archive.read(PART)
        try:
            ET.fromstring(payload)
            return None
        except ET.ParseError:
            pass
        fixed, count = BARE_AMPERSAND.subn(b"&amp;", payload)
        if not count:
            return None
        ET.fromstring(fixed)  # still malformed: a different defect, let it raise
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as out:
            for info in archive.infolist():
                out.writestr(info, fixed if info.filename == PART else archive.read(info.filename))
    buffer.seek(0)
    return buffer, {"part": PART, "rule": RULE, "replacements": count}


def load_workbook(path):
    """Read-only workbook (formulas kept) and the list of repairs it needed."""
    try:
        return openpyxl.load_workbook(path, read_only=True, data_only=False), []
    except SyntaxError:  # xml.etree ParseError and lxml XMLSyntaxError both derive from it
        repaired = repaired_copy(path)
        if repaired is None:
            raise
        buffer, repair = repaired
        return openpyxl.load_workbook(buffer, read_only=True, data_only=False), [repair]


def processing_quality(path) -> str | None:
    """The "Processing Quality" of an Xsens export (HD, Normal, Live), or None if unreadable."""
    try:
        book, _ = load_workbook(path)
        try:
            sheet = book["General Information"]
            sheet.reset_dimensions()  # some exports declare dimensions smaller than their data
            for row in sheet.iter_rows(values_only=True):
                if row and row[0] is not None and str(row[0]).strip() == "Processing Quality":
                    return str(row[1]).strip() if len(row) > 1 and row[1] is not None else None
        finally:
            book.close()
    except (OSError, KeyError, ValueError, SyntaxError, BadZipFile):
        return None
    return None
