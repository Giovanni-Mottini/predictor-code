"""Checksum-verified Xsens descriptors and preserved source tables."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from shared_code import SCHEMA_VERSION
from shared_code.reader import DataContractError, SubjectReader, contained_path, document


def open_recording(subject: SubjectReader, record: dict) -> "XsensRecording":
    if record not in subject.manifest["records"]:
        raise DataContractError("record_not_in_manifest", "Record is not from this manifest")
    if record.get("modality") != "xsens" or record.get("disposition") != "mapped":
        raise DataContractError("record_not_selected", "Expected a mapped Xsens record")
    if record.get("formatter_status") != "converted":
        raise DataContractError("record_not_converted", "Selected recording is not converted")
    relative = (record.get("formatter_report") or {}).get("descriptor_path")
    target = record.get("normalized_target")
    if not relative or not target:
        raise DataContractError("descriptor_missing", "Recording descriptor path is missing")
    path = contained_path(subject.root, relative)
    if path != contained_path(subject.root, str(Path(target) / "recording.json")):
        raise DataContractError("descriptor_path_mismatch", "Descriptor does not match canonical target")
    descriptor, digest = document(path)
    if descriptor.get("schema_version") != SCHEMA_VERSION:
        raise DataContractError("unsupported_schema", "Unsupported recording schema")
    for key in ("source_id", "source_path"):
        if descriptor.get(key) != record.get(key):
            raise DataContractError("descriptor_identity_mismatch", f"Recording {key} disagrees with manifest")
    if not record.get("sha256") or descriptor.get("source_sha256") != record["sha256"]:
        raise DataContractError("source_revision_mismatch", "Manifest and recording content revisions disagree")
    if descriptor.get("import_policy_id") != subject.manifest.get("import_policy", {}).get("policy_id"):
        raise DataContractError("import_policy_mismatch", "Recording import policy differs from manifest")
    if descriptor.get("all_values_roundtrip_verified") is not True:
        raise DataContractError("conversion_not_verified", "Recording lacks complete round-trip verification")
    for key in ("expected_repetitions", "expected_marker_count"):
        if descriptor.get(key) != record.get(key):
            raise DataContractError("expectation_mismatch", f"Recording {key} disagrees with manifest")
    return XsensRecording(subject, record, path, descriptor, digest)


@dataclass
class XsensRecording:
    subject: SubjectReader
    record: dict
    descriptor_path: Path
    descriptor: dict
    descriptor_sha256: str
    checked_tables: dict = field(default_factory=dict)

    def table_descriptor(self, sheet: str) -> dict:
        candidates = [t for t in self.descriptor.get("tables", []) if t.get("source_sheet") == sheet]
        if len(candidates) != 1:
            raise DataContractError("sheet_missing_or_ambiguous", f"Expected one table for {sheet!r}")
        return candidates[0]

    def _verified_payload(self, sheet: str, expected_format: str) -> tuple[dict, bytes]:
        table = self.table_descriptor(sheet)
        if table.get("format") != expected_format or table.get("roundtrip_verified") is not True:
            raise DataContractError("table_not_verified", f"Unverified or wrong-format table: {sheet}")
        path = contained_path(self.descriptor_path.parent, table["file"])
        # Hash the exact bytes subsequently decoded, avoiding a check/read race.
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != table.get("sha256"):
            raise DataContractError("table_checksum_mismatch", f"Changed or damaged table: {sheet}")
        self.checked_tables[sheet] = {"file": str(path.relative_to(self.subject.root)),
                                      "sha256": digest, "size_bytes": len(payload)}
        return table, payload

    def read_table(self, sheet: str, columns: list[str] | None = None) -> pa.Table:
        desc, payload = self._verified_payload(sheet, "parquet")
        parquet = pq.ParquetFile(pa.BufferReader(payload))
        actual = parquet.schema_arrow
        declared = [c["name"] for c in desc["columns"]]
        if actual.names != declared or len(actual.names) != len(set(actual.names)):
            raise DataContractError("table_columns_mismatch", f"Column contract mismatch: {sheet}")
        if columns is not None and any(c not in actual.names for c in columns):
            raise DataContractError("column_missing", f"Requested column absent in {sheet}")
        result = parquet.read(columns=columns)
        if result.num_rows != desc["row_count"]:
            raise DataContractError("table_rows_mismatch", f"Row count mismatch: {sheet}")
        return result

    def read_general_information(self) -> dict:
        desc, payload = self._verified_payload("General Information", "json")
        result = json.loads(payload)
        if result.get("source_sheet") != "General Information" or len(result["rows"]) != desc["row_count"]:
            raise DataContractError("metadata_rows_mismatch", "General Information contract mismatch")
        return result

    def provenance(self) -> dict:
        return {"subject_path": str(self.subject.root),
                "subject_metadata_sha256": self.subject.metadata_sha256,
                "descriptor_path": str(self.descriptor_path.relative_to(self.subject.root)),
                "descriptor_sha256": self.descriptor_sha256,
                "source_id": self.record["source_id"], "source_path": self.record["source_path"],
                "source_sha256": self.descriptor["source_sha256"],
                "raw_source_rechecked": False, "checked_tables": self.checked_tables}
