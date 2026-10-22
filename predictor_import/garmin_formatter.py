"""Streaming Garmin CSV/ZIP conversion; no timestamp alignment or scientific QC."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import codecs
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import time
from zipfile import BadZipFile, ZipFile

import pyarrow as pa
import pyarrow.parquet as pq

from . import SCHEMA_VERSION
from .adapter import atomic_json
from .formatter import checksum, feed_rows, fingerprint
from .inventory import now
from .inspection import csv_delimiter
from .content_cache import link_table, load_table, save_table, stream_checksum
from .storage_names import garmin_table_names, garmin_targets

FORMATTER_VERSION = "garmin-parquet-0.2.0"
CHUNK_ROWS = 16384
INTEGER = re.compile(r"[+-]?(?:0|[1-9][0-9]*)\Z")
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
TEXT_WORDS = {"id", "name", "names", "email", "status", "qualifier", "type", "source", "tracker", "device", "code", "gender", "country", "city", "note", "unit", "units"}
UNITS = {"s", "ms", "sec", "msec", "%", "bpm", "kcal", "m", "m/s", "min/km", "spm", "breaths/min", "celsius", "strokes/min", "rounds/min"}


def column_names(headers: list[str]) -> list[str]:
    """Keep a positional source map, including repeated headers in sleep.csv."""
    result, used = [], set()
    for index, header in enumerate(headers):
        base = re.sub(r"[^\w]+", "_", header.casefold()).strip("_") or "column"
        if base[0].isdigit():
            base = "date_" + base if re.fullmatch(r"\d{4}-\d{2}-\d{2}", header) else "column_" + base
        name = base
        while name in used:
            name += f"__{index + 1}"
        result.append(name)
        used.add(name)
    return result


def text_reason(header: str) -> str | None:
    words = set(re.findall(r"[a-z]+", header.casefold()))
    if words & TEXT_WORDS:
        return "identifier_or_categorical_header"
    if words & {"date", "local", "utc", "timezone"} or ("zone" in words and "time" in words and "s" not in words and "sec" not in words):
        return "source_time_text_preserved"
    return None


@contextmanager
def csv_rows(open_stream, delimiter: str | None = None, expected_headers=None):
    with open_stream() as raw:
        # Reopen after bounded dialect detection; ZIP streams need no extraction.
        if delimiter is None:
            # An incomplete multibyte character at the sample boundary is not
            # malformed CSV. The full read below still decodes strictly.
            prefix = codecs.getincrementaldecoder("utf-8-sig")().decode(raw.read(65536), final=False)
            delimiter = csv_delimiter(prefix, expected_headers)
    with open_stream() as raw:
        with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
            rows = csv.reader(text, delimiter=delimiter, strict=True)
            headers = next(rows, [])
            if not headers or any(not header for header in headers):
                raise ValueError("Missing or empty CSV column header")
            yield headers, rows, delimiter


def checked_rows(rows, width):
    for row_number, row in enumerate(rows, start=2):
        if len(row) != width:
            raise ValueError(f"CSV record {row_number} has {len(row)} fields; expected {width}")
        yield row


def profile_table(open_stream, expected_headers: list[str]) -> dict:
    with csv_rows(open_stream, expected_headers=expected_headers) as (headers, rows, delimiter):
        if headers != expected_headers:
            raise ValueError("CSV headers changed since inventory")
        columns = [{"kind": "int64", "reason": text_reason(h), "null_count": 0,
                    "count": 0, "min_integer": None, "max_integer": None,
                    "nonfinite_tokens": 0} for h in headers]
        digest, count = hashlib.sha256(), 0
        for row in checked_rows(rows, len(headers)):
            feed_rows(digest, [row])
            count += 1
            for value, column in zip(row, columns):
                if value == "":
                    column["null_count"] += 1
                    continue
                column["count"] += 1
                if value.casefold() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
                    column["nonfinite_tokens"] += 1
                if column["reason"]:
                    continue
                if INTEGER.fullmatch(value):
                    integer = int(value)
                    if not -(2**63) <= integer < 2**63:
                        column["reason"] = "integer_outside_int64_preserved_as_text"
                    else:
                        column["min_integer"] = integer if column["min_integer"] is None else min(integer, column["min_integer"])
                        column["max_integer"] = integer if column["max_integer"] is None else max(integer, column["max_integer"])
                elif NUMBER.fullmatch(value):
                    numeric = float(value)
                    if not math.isfinite(numeric) or Decimal(str(numeric)) != Decimal(value):
                        column["reason"] = "numeric_precision_preserved_as_text"
                    elif re.match(r"[+-]?0\d", value):
                        column["reason"] = "leading_zero_format_preserved_as_text"
                    else:
                        column["kind"] = "float64"
                else:
                    column["reason"] = "nonnumeric_values_preserved_as_text"
        for column in columns:
            if not column["count"]:
                column["reason"] = "empty_column_type_unknown"
            if column["kind"] == "float64" and any(v is not None and abs(v) > 2**53 for v in (column["min_integer"], column["max_integer"])):
                column["reason"] = "mixed_numeric_precision_preserved_as_text"
            if column["reason"]:
                column["kind"] = "string"
        return {"headers": headers, "names": column_names(headers), "columns": columns,
                "row_count": count, "delimiter": delimiter, "source_cells_sha256": digest.hexdigest()}


def convert_table(open_stream, expected_headers: list[str], output: Path, source_table: str) -> dict:
    start = time.perf_counter()
    profile = profile_table(open_stream, expected_headers)
    schema = pa.schema([pa.field(name, {"string": pa.string(), "int64": pa.int64(), "float64": pa.float64()}[column["kind"]])
                        for name, column in zip(profile["names"], profile["columns"])])
    cell_digest, typed_digest, count = hashlib.sha256(), hashlib.sha256(), 0
    with pq.ParquetWriter(output, schema, compression="zstd") as writer:
        with csv_rows(open_stream, profile["delimiter"]) as (headers, rows, _):
            if headers != profile["headers"]:
                raise ValueError("CSV headers changed during conversion")
            pending = []
            for row in checked_rows(rows, len(schema)):
                feed_rows(cell_digest, [row])
                values = [None if value == "" else int(value) if column["kind"] == "int64" else float(value) if column["kind"] == "float64" else value
                          for value, column in zip(row, profile["columns"])]
                feed_rows(typed_digest, [values])
                count += 1
                pending.append(values)
                if len(pending) == CHUNK_ROWS:
                    writer.write_table(pa.Table.from_arrays([pa.array([r[i] for r in pending], type=f.type) for i, f in enumerate(schema)], schema=schema))
                    pending.clear()
            if pending:
                writer.write_table(pa.Table.from_arrays([pa.array([r[i] for r in pending], type=f.type) for i, f in enumerate(schema)], schema=schema))
    if count != profile["row_count"] or cell_digest.hexdigest() != profile["source_cells_sha256"]:
        raise ValueError("CSV cells changed during conversion")
    parquet, output_digest = pq.ParquetFile(output), hashlib.sha256()
    for batch in parquet.iter_batches(batch_size=CHUNK_ROWS):
        feed_rows(output_digest, zip(*(column.to_pylist() for column in batch.columns)))
    if parquet.metadata.num_rows != count or output_digest.digest() != typed_digest.digest():
        raise ValueError("Parquet value round-trip mismatch")
    columns = []
    warnings = []
    for index, (header, name, column) in enumerate(zip(headers, profile["names"], profile["columns"])):
        suffix = re.search(r"\(([^()]*)\)\s*$", header)
        unit = suffix.group(1) if suffix and suffix.group(1).casefold() in UNITS else None
        columns.append({"name": name, "source_header": header, "source_index": index, "dtype": column["kind"],
                        "unit": unit, "null_count": column["null_count"], "type_reason": column["reason"],
                        "nonfinite_token_count": column["nonfinite_tokens"]})
        if column["nonfinite_tokens"]:
            warnings.append(f"nonfinite_tokens_preserved:{name}")
        if column["reason"] and "precision" in column["reason"]:
            warnings.append(f"numeric_precision_preserved_as_text:{name}")
    if len(headers) != len(set(headers)):
        warnings.append("repeated_source_headers_disambiguated")
    return {"source_table": source_table, "file": output.name, "format": "parquet", "row_count": count,
            "columns": columns, "source_headers": headers, "csv_delimiter": profile["delimiter"],
            "null_policy": "empty_csv_field_only", "source_cells_sha256": cell_digest.hexdigest(),
            "values_sha256": typed_digest.hexdigest(), "sha256": checksum(output),
            "roundtrip_verified": True, "warnings": warnings, "conversion_seconds": time.perf_counter() - start}


def selected_tables(source: Path, record: dict, archive: ZipFile | None) -> dict:
    if archive is None:
        return {source.name: {"columns": record["columns"], "open": lambda: source.open("rb")}}
    infos = archive.infolist()
    names = [info.filename for info in infos]
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or (path.parts and ":" in path.parts[0]):
            raise ValueError("Unsafe ZIP member path")
    if len(names) != len(set(names)):
        raise ValueError("Repeated ZIP member paths")
    known = {m["path"]: m for m in record["archive_members"]}
    tables = {}
    for name in record["selected_archive_members"] or []:
        info, evidence = archive.getinfo(name), known[name]
        if info.is_dir() or info.file_size != evidence["size_bytes"] or f"{info.CRC:08x}" != evidence["crc32"]:
            raise ValueError("ZIP member changed since inventory")
        tables[name] = {"columns": evidence["columns"], "open": lambda name=name: archive.open(name)}
    if not tables:
        raise ValueError("No selected Garmin tables")
    return tables


def cached_tables(target: Path, source_sha: str, policy_id: str, selection: list[str], record: dict) -> dict:
    try:
        descriptor = json.loads((target / "export.json").read_text())
        if (descriptor["source_sha256"], descriptor["formatter_version"], descriptor["import_policy_id"], descriptor["selected_source_tables"]) != (source_sha, FORMATTER_VERSION, policy_id, selection):
            return {}
        if (descriptor["source_id"], descriptor["source_path"]) != (record["source_id"], record["source_path"]):
            return {}
        valid = {}
        for table in descriptor["tables"]:
            path = target / table["file"]
            if table["source_table"] in selection and path.resolve().is_relative_to(target.resolve()) and not path.is_symlink() and checksum(path) == table["sha256"]:
                valid[table["source_table"]] = table
        return valid
    except (OSError, KeyError, ValueError, TypeError):
        return {}


def convert_record(source: Path, record: dict, target: Path, policy_id: str, *, table_names: list[str] | None = None, progress=None) -> dict:
    start = time.perf_counter()
    before = fingerprint(source)
    if before != (record["size_bytes"], record["mtime_ns"]):
        raise ValueError("Source changed since inventory; refresh before converting")
    try:
        previous = json.loads((target / "export.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    if previous.get("source_id") and previous["source_id"] != record["source_id"]:
        raise ValueError("Garmin destination belongs to a different source; use a distinct export name")
    source_sha = checksum(source)
    context = ZipFile(source) if record["artifact_kind"] == "garmin_archive" else nullcontext(None)
    with context as archive:
        sources = selected_tables(source, record, archive)
        selection = sorted(sources)
        filenames = garmin_table_names(selection)
        requested = {name for name in selection if table_names is None or name in table_names or PurePosixPath(name).name in table_names}
        if table_names and any(not any(name == wanted or PurePosixPath(name).name == wanted for name in selection) for wanted in table_names):
            raise ValueError("Requested table not found in selected Garmin source")
        valid = cached_tables(target, source_sha, policy_id, selection, record)
        cache_root = target.parent / ".table_cache"
        if not cache_root.resolve().is_relative_to(target.parent.resolve()):
            raise ValueError("Content cache escapes output root")
        if requested <= valid.keys() and all(valid[name].get("source_csv_sha256") for name in requested) and fingerprint(source) == before:
            return {"status": "cached", "source_id": record["source_id"], "descriptor": json.loads((target / "export.json").read_text()), "tables_reused": len(requested), "tables_rebuilt": 0}
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".format-", dir=target.parent))
        try:
            tables, pending_cache = [], []
            rebuilt = reused = 0
            for name in selection:
                if name in valid:
                    table = dict(valid[name])
                    previous_file = target / table["file"]
                    table["file"] = filenames[name]
                    raw_sha = table.get("source_csv_sha256") or stream_checksum(sources[name]["open"])
                    table["source_csv_sha256"] = raw_sha
                    link_table(previous_file, temporary / table["file"])
                    tables.append(table)
                    pending_cache.append((raw_sha, table, temporary / table["file"]))
                    if name in requested:
                        reused += 1
                elif name in requested:
                    raw_sha = stream_checksum(sources[name]["open"])
                    shared = load_table(cache_root, FORMATTER_VERSION, raw_sha, sources[name]["columns"])
                    output = temporary / filenames[name]
                    if shared:
                        previous_table, previous_file = shared
                        table = {**previous_table, "source_table": name, "file": output.name}
                        link_table(previous_file, output)
                        reused += 1
                    else:
                        table = convert_table(sources[name]["open"], sources[name]["columns"], output, name)
                        if stream_checksum(sources[name]["open"]) != raw_sha:
                            raise ValueError("CSV content changed during conversion")
                        rebuilt += 1
                    table["source_csv_sha256"] = raw_sha
                    tables.append(table)
                    pending_cache.append((raw_sha, table, output))
                    if progress:
                        progress(name, table["row_count"])
            after_sha = checksum(source)
            if fingerprint(source) != before or after_sha != source_sha:
                raise ValueError("Source changed during conversion")
            for raw_sha, table, file in pending_cache:
                save_table(cache_root, FORMATTER_VERSION, raw_sha, table, file)
            descriptor = {"schema_version": SCHEMA_VERSION, "formatter_version": FORMATTER_VERSION,
                "import_policy_id": policy_id, "source_id": record["source_id"], "source_path": record["source_path"],
                "source_sha256": source_sha, "source_size_bytes": before[0], "source_mtime_ns": before[1],
                "created_at": now(), "compression": "zstd", "tables": tables, "selected_source_tables": selection,
                "complete_selected_tables": len(tables) == len(selection), "timestamp_policy": "preserve_source_columns_without_timezone_conversion",
                "numeric_policy": "int64_or_float64_with_decimal_text_roundtrip_guard_otherwise_string",
                "crc_policy": "converted_archive_members_read_to_eof" if archive else "not_applicable",
                "warnings": sorted({warning for table in tables for warning in table["warnings"]}),
                "all_values_roundtrip_verified": True, "ready_for_features": False,
                "output_size_bytes": sum((temporary / table["file"]).stat().st_size for table in tables),
                "conversion_seconds": time.perf_counter() - start}
            atomic_json(temporary / "export.json", descriptor)
            backup = target.with_name(target.name + ".previous")
            if backup.exists():
                raise ValueError("Unresolved previous conversion backup; inspect before replacing")
            if target.exists():
                target.rename(backup)
            try:
                temporary.rename(target)
            except OSError:
                if backup.exists():
                    backup.rename(target)
                raise
            if backup.exists():
                shutil.rmtree(backup)
            return {"status": "converted" if rebuilt else "cached", "source_id": record["source_id"], "descriptor": descriptor,
                    "tables_reused": reused, "tables_rebuilt": rebuilt}
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def format_subject(subject_path: Path, *, source_ids=None, limit=None, table_names=None, progress=None) -> dict:
    subject_path = subject_path.resolve()
    manifest_path = subject_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Restage metadata with the current schema before formatting")
    root = Path(manifest["source_root"]).resolve()
    if subject_path == root or subject_path.is_relative_to(root) or root.is_relative_to(subject_path):
        raise ValueError("Formatted output must be separate from raw sources")
    known_ids = {r["source_id"] for r in manifest["records"]}
    if source_ids and set(source_ids) - known_ids:
        raise ValueError("Requested source ID not found in subject manifest")
    candidates = [r for r in manifest["records"] if r["disposition"] == "mapped" and r["artifact_kind"] in {"garmin_archive", "garmin_table"}
                  and (not source_ids or r["source_id"] in source_ids)]
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")
    from .layout import migrate_subject
    targets = garmin_targets(manifest["records"])
    results, migrations = [], []
    for record in candidates[:limit]:
        try:
            migration = migrate_subject(subject_path, modality="garmin",
                                        targets={record["source_id"]: targets[record["source_id"]]})
            if migration["records_changed"]:
                migrations.append(migration)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                record = next(r for r in manifest["records"] if r["source_id"] == record["source_id"])
            if not record["normalized_target"]:
                raise ValueError("Export has no canonical destination")
            target, source = subject_path / record["normalized_target"], root / record["source_path"]
            if target.resolve() == subject_path or not target.resolve().is_relative_to(subject_path) or source.is_symlink() or not source.resolve().is_relative_to(root):
                raise ValueError("Source or destination escapes its root")
            result = convert_record(source, record, target, manifest["import_policy"]["policy_id"], table_names=table_names, progress=progress)
            descriptor = result.pop("descriptor")
            record["sha256"] = descriptor["source_sha256"]
            complete = descriptor["complete_selected_tables"]
            record["formatter_status"] = "converted" if complete else "not_converted"
            record["formatter_report"] = {"descriptor_path": record["normalized_target"] + "/export.json", "warnings": descriptor["warnings"], "complete_selected_tables": complete}
            result.update({"complete_selected_tables": complete, "available_table_count": len(descriptor["tables"])})
        except (OSError, ValueError, KeyError, UnicodeError, csv.Error, InvalidOperation, pa.ArrowException, BadZipFile, RuntimeError) as error:
            record["formatter_status"] = "failed"
            record["formatter_report"] = {"error": str(error)}
            result = {"status": "failed", "source_id": record["source_id"], "error": str(error)}
        results.append(result)
        manifest["adapter_stage"] = "metadata_and_tables"
        atomic_json(manifest_path, manifest)
    report = {"schema_version": SCHEMA_VERSION, "participant_key": manifest["participant_key"], "results": results,
              "converted": sum(r["status"] == "converted" for r in results), "cached": sum(r["status"] == "cached" for r in results),
              "failed": sum(r["status"] == "failed" for r in results),
              "complete_exports": sum(r.get("complete_selected_tables", False) for r in results),
              "partial_exports": sum(r["status"] != "failed" and not r["complete_selected_tables"] for r in results),
              "ready_for_feature_extraction": False}
    report["layout_migrations"] = migrations
    atomic_json(subject_path / "garmin_format_report.json", report)
    return report
