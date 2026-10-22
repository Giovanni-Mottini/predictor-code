"""Bounded, checksum-verified Garmin reads from selected manifest descriptors."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re

import pandas as pd
import pyarrow.parquet as pq

from shared_code import SCHEMA_VERSION
from shared_code.reader import DataContractError, SubjectReader, contained_path, document


# Fitrockr exports from October 2026 name the health-log local clock "Selected
# Time Zone" (set to the user's own zone). Read them under the earlier names.
SELECTED_ZONE = {
    "calendar_date_selected_time_zone": "calendar_date_local",
    "start_time_selected_time_zone": "start_time_local",
    "end_time_selected_time_zone": "end_time_local",
    "selected_time_zone": "time_zone_local",
    "selected_time_zone_s": "time_zone_s",
}


def local_names(names):
    """Source column -> pipeline column; identity unless only the new names exist."""
    return {c: SELECTED_ZONE.get(c, c) if SELECTED_ZONE.get(c) not in names else c for c in names}


def fingerprint(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@dataclass
class GarminExport:
    subject: SubjectReader
    record: dict
    descriptor_path: Path
    descriptor: dict
    descriptor_sha256: str
    checked_tables: dict = field(default_factory=dict)

    def iter_all_rows(self, table: dict, *, batch_size: int = 16384):
        """Checksum-verified whole-table batches, including undated observations.

        Used by advisory dataset QC, not by feature extraction. Consumers must
        exhaust the iterator before publishing results: the final fingerprint
        check rejects concurrent replacement/modification of the input.
        """
        if (table not in self.descriptor["tables"] or table.get("roundtrip_verified") is not True
                or table.get("format") != "parquet"):
            raise DataContractError("table_not_verified", "Expected a verified Garmin Parquet table")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise DataContractError("invalid_batch_size", "Use a positive integer batch size")
        path = contained_path(self.descriptor_path.parent, table["file"])
        offset, dates = 0, {}
        with path.open("rb") as stream:
            before = fingerprint(os.fstat(stream.fileno()))
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != table.get("sha256"):
                raise DataContractError("table_checksum_mismatch", f"Changed table: {table['source_table']}")
            stream.seek(0)
            parquet = pq.ParquetFile(stream)
            names = [c["name"] for c in table["columns"]]
            if parquet.schema_arrow.names != names or len(names) != len(set(names)):
                raise DataContractError("table_columns_mismatch", "Garmin column contract mismatch")
            if parquet.metadata.num_rows != table["row_count"]:
                raise DataContractError("table_rows_mismatch", "Garmin row count mismatch")
            columns = [c for c in names if not c.startswith(("user_", "team_", "group_"))]
            for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
                data = batch.to_pandas()
                data["source_row"] = data.index.to_numpy() + offset
                data["source_id"] = self.record["source_id"]
                data["source_table"] = table["source_table"]
                offset += len(data)
                for name in ("start_time_local", "end_time_local", "start_time_utc"):
                    if name not in data:
                        continue
                    text = data[name].astype("string")
                    valid = text.str.match(r"^\d{4}-\d{2}-\d{2}T", na=False)
                    state = dates.setdefault(name, {"missing_or_noniso_prefix_count": 0, "min_observed_text": None, "max_observed_text": None})
                    state["missing_or_noniso_prefix_count"] += int((~valid).sum())
                    if valid.any():
                        lo, hi = text[valid].min(), text[valid].max()
                        state["min_observed_text"] = min(state["min_observed_text"], lo) if state["min_observed_text"] else lo
                        state["max_observed_text"] = max(state["max_observed_text"], hi) if state["max_observed_text"] else hi
                yield data
            if fingerprint(os.fstat(stream.fileno())) != before or fingerprint(path.stat()) != before:
                raise DataContractError("table_changed_during_read", "Garmin table changed during decoding")
        self.checked_tables[table["source_table"]] = {
            "file": str(path.relative_to(self.subject.root)), "sha256": digest,
            "source_csv_sha256": table.get("source_csv_sha256"), "size_bytes": before[2],
            "total_rows": offset, "selected_rows": offset, "selection": "entire_table_no_date_filter",
            "observed_timestamps": dates,
        }

    def read_period(
        self,
        table: dict,
        *,
        start: str,
        end: str,
        date_column: str = "start_time_local",
        max_rows: int = 2_000_000,
    ) -> pd.DataFrame:
        """Select local calendar dates in [start, end), retaining CSV record positions.

        Hash the entire Parquet through the same open handle used for decoding.
        Decode batches rather than materializing the full export. ISO date-prefix
        filtering bounds retained memory; dates are strictly parsed by scientific QC.
        Malformed/null timestamps in scanned batches are separately counted.
        """
        if (
            table not in self.descriptor["tables"]
            or table.get("roundtrip_verified") is not True
        ):
            raise DataContractError(
                "table_not_verified", "Expected a verified descriptor table"
            )
        if table.get("format") != "parquet":
            raise DataContractError(
                "table_not_verified", "Garmin table must be Parquet"
            )
        path = contained_path(self.descriptor_path.parent, table["file"])
        frames, retained, offset, malformed, scanned, skipped = [], 0, 0, 0, 0, 0
        with path.open("rb") as stream:
            before = fingerprint(os.fstat(stream.fileno()))
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != table.get("sha256"):
                raise DataContractError(
                    "table_checksum_mismatch", f"Changed table: {table['source_table']}"
                )
            stream.seek(0)
            parquet = pq.ParquetFile(stream)
            names = [c["name"] for c in table["columns"]]
            if parquet.schema_arrow.names != names or len(names) != len(set(names)):
                raise DataContractError(
                    "table_columns_mismatch", "Garmin column contract mismatch"
                )
            if parquet.metadata.num_rows != table["row_count"]:
                raise DataContractError(
                    "table_rows_mismatch", "Garmin row count mismatch"
                )
            rename = local_names(names)
            source_date = next((c for c, n in rename.items() if n == date_column), date_column)
            if source_date not in names:
                raise DataContractError("column_missing", f"Missing {date_column}")
            # Identity is taken from subject.json, never personal CSV columns.
            columns = [
                c for c in names if not c.startswith(("user_", "team_", "group_"))
            ]
            for group_index in range(parquet.num_row_groups):
                group = parquet.metadata.row_group(group_index)
                stats = group.column(names.index(source_date)).statistics
                # Only prune with ISO-looking string bounds; no chronology/sort assumption.
                bounds = (stats.min, stats.max) if stats and stats.has_min_max else ()
                can_prune = len(bounds) == 2 and all(
                    isinstance(v, str) and re.match(r"^\d{4}-\d{2}-\d{2}T", v)
                    for v in bounds
                )
                if can_prune and (bounds[1][:10] < start or bounds[0][:10] >= end):
                    offset += group.num_rows
                    skipped += 1
                    continue
                for batch in parquet.iter_batches(
                    batch_size=16384, columns=columns, row_groups=[group_index]
                ):
                    frame = batch.to_pandas().rename(columns=rename)
                    text = frame[date_column].astype("string")
                    valid_prefix = text.str.match(r"^\d{4}-\d{2}-\d{2}T", na=False)
                    malformed += int((~valid_prefix).sum())
                    mask = (
                        valid_prefix
                        & text.str.slice(0, 10).ge(start)
                        & text.str.slice(0, 10).lt(end)
                    )
                    selected = frame.loc[mask].copy()
                    selected["source_row"] = selected.index.to_numpy() + offset
                    selected["source_id"] = self.record["source_id"]
                    selected["source_table"] = table["source_table"]
                    retained += len(selected)
                    if retained > max_rows:
                        raise DataContractError(
                            "period_too_large",
                            f"Selection exceeds {max_rows} rows; narrow the dates",
                        )
                    if len(selected):
                        frames.append(selected)
                    offset += len(frame)
                    scanned += len(frame)
            if (
                fingerprint(os.fstat(stream.fileno())) != before
                or fingerprint(path.stat()) != before
            ):
                raise DataContractError(
                    "table_changed_during_read", "Garmin table changed during decoding"
                )
        self.checked_tables[table["source_table"]] = {
            "file": str(path.relative_to(self.subject.root)),
            "sha256": digest,
            "source_csv_sha256": table.get("source_csv_sha256"),
            "size_bytes": before[2],
            "total_rows": offset,
            "selected_rows": retained,
            "unassignable_date_rows_in_scanned_groups": malformed,
            "decoded_rows": scanned,
            "pruned_row_groups": skipped,
            "selection_date_column": date_column,
            "start_date": start,
            "end_date_exclusive": end,
        }
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame(
            columns=[*(rename[c] for c in columns), "source_row", "source_id", "source_table"]
        )

    def provenance(self):
        return {
            "subject_path": str(self.subject.root),
            "subject_metadata_sha256": self.subject.metadata_sha256,
            "descriptor_path": str(self.descriptor_path.relative_to(self.subject.root)),
            "descriptor_sha256": self.descriptor_sha256,
            "source_id": self.record["source_id"],
            "source_path": self.record["source_path"],
            "source_sha256": self.record["sha256"],
            "raw_source_rechecked": False,
            "selection_reason": self.record.get("selection_reason"),
            "selection_assumption": self.record.get("selection_assumption"),
            "checked_tables": self.checked_tables,
        }


def open_export(subject: SubjectReader, record: dict) -> GarminExport:
    if record not in subject.manifest["records"]:
        raise DataContractError(
            "record_not_in_manifest", "Record is not from this manifest"
        )
    if record.get("modality") != "garmin" or record.get("disposition") != "mapped":
        raise DataContractError(
            "record_not_selected", "Expected a selected Garmin export"
        )
    if record.get("formatter_status") != "converted":
        raise DataContractError(
            "record_not_converted", "Selected Garmin export is not converted"
        )
    relative = (record.get("formatter_report") or {}).get("descriptor_path")
    target = record.get("normalized_target")
    if not relative or not target:
        raise DataContractError(
            "descriptor_missing", "Garmin descriptor path is missing"
        )
    path = contained_path(subject.root, relative)
    if path != contained_path(subject.root, str(Path(target) / "export.json")):
        raise DataContractError(
            "descriptor_path_mismatch", "Garmin descriptor disagrees with target"
        )
    descriptor, digest = document(path)
    if descriptor.get("schema_version") != SCHEMA_VERSION:
        raise DataContractError(
            "unsupported_schema", "Unsupported Garmin descriptor schema"
        )
    for key in ("source_id", "source_path"):
        if descriptor.get(key) != record.get(key):
            raise DataContractError(
                "descriptor_identity_mismatch", f"Garmin {key} disagrees with manifest"
            )
    if not record.get("sha256") or descriptor.get("source_sha256") != record["sha256"]:
        raise DataContractError(
            "source_revision_mismatch", "Garmin revision disagrees with manifest"
        )
    if descriptor.get("import_policy_id") != subject.manifest.get(
        "import_policy", {}
    ).get("policy_id"):
        raise DataContractError(
            "import_policy_mismatch", "Garmin import policy disagrees with manifest"
        )
    if (
        descriptor.get("complete_selected_tables") is not True
        or descriptor.get("all_values_roundtrip_verified") is not True
    ):
        raise DataContractError(
            "conversion_not_verified", "Garmin export is incomplete or unverified"
        )
    return GarminExport(subject, record, path, descriptor, digest)


def check_snapshot(subject: SubjectReader, exports: list[GarminExport]):
    for name, digest in subject.metadata_sha256.items():
        if hashlib.sha256((subject.root / name).read_bytes()).hexdigest() != digest:
            raise DataContractError("metadata_changed_during_read", f"Changed {name}")
    for export in exports:
        if (
            hashlib.sha256(export.descriptor_path.read_bytes()).hexdigest()
            != export.descriptor_sha256
        ):
            raise DataContractError(
                "metadata_changed_during_read", "Garmin descriptor changed"
            )
