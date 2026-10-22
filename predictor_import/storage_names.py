"""Readable derived filenames; source and recording IDs remain metadata."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
import re
import unicodedata


def slug(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "worksheet"


def xsens_recording_name(record: dict) -> str:
    activity = slug(record.get("activity") or "recording")
    code = record.get("original_limb_code")
    if code in {"dx", "sx"}:
        arm = {"dx": "right", "sx": "left"}[code]
    else:
        arm = record.get("limb_condition") or record.get("anatomical_side") or "unspecified_arm"
    name = f"{activity}_{slug(arm)}"
    if record.get("repetition") is not None:
        name += f"_rep{int(record['repetition'])}"
    return name


def xsens_targets(records: list[dict]) -> dict[str, str]:
    """Disambiguate only actual collisions within a participant/session."""
    groups = defaultdict(list)
    for record in records:
        if (record.get("artifact_kind") == "xsens_workbook"
                and record.get("disposition") == "mapped" and record.get("session_id")):
            session = record["session_id"]
            if Path(session).name != session or session in {".", ".."}:
                raise ValueError("Unsafe Xsens session label")
            groups[f"xsens/{session}/{xsens_recording_name(record)}"].append(record)
    return unique_targets(groups)


def unique_targets(groups: dict[str, list[dict]]) -> dict[str, str]:
    """Keep readable bases unless selected sources genuinely collide."""
    targets = {}
    for base, group in groups.items():
        width = 8
        while len({r["source_id"][4:4 + width] for r in group}) != len(group):
            width += 4
            if width > 24:
                raise ValueError("Duplicate source identifiers")
        for record in group:
            suffix = f"__{record['source_id'][4:4 + width]}" if len(group) > 1 else ""
            targets[record["source_id"]] = base + suffix
    return targets


def source_period(source_path: str) -> str | None:
    """Preserve explicit filename/folder date labels, without inferring coverage."""
    for part in reversed(PurePosixPath(source_path).parts):
        match = re.search(r"(?<!\d)(\d{2}\.\d{2}\.\d{4})-(\d{2}\.\d{2}\.\d{4})(?!\d)", part)
        if match:
            try:
                dates = [datetime.strptime(value, "%d.%m.%Y").date() for value in match.groups()]
            except ValueError:
                continue
            return f"{dates[0].isoformat()}_to_{dates[1].isoformat()}"
    return None


def garmin_export_name(record: dict) -> str:
    source = PurePosixPath(record["source_path"])
    period = source_period(record["source_path"])
    if record["artifact_kind"] == "garmin_table":
        return "table_" + slug(source.stem) + (f"_{period}" if period else "")
    if period:
        return f"export_{period}"
    stem = source.name
    for extension in (".zip", ".csv"):
        if stem.lower().endswith(extension):
            stem = stem[:-len(extension)]
    label = slug(stem)
    participant = slug(source.parts[1]) if len(source.parts) > 1 else ""
    aliases = {participant, participant.removeprefix("sano_"), "export", "data", "garmin"}
    if label in aliases:
        return "export"
    return label if label.startswith("export_") else f"export_{label}"


def garmin_targets(records: list[dict]) -> dict[str, str]:
    groups = defaultdict(list)
    for record in records:
        if (record.get("artifact_kind") in {"garmin_archive", "garmin_table"}
                and record.get("disposition") == "mapped"):
            groups[f"garmin/{garmin_export_name(record)}"].append(record)
    return unique_targets(groups)


def garmin_table_names(source_tables: list[str]) -> dict[str, str]:
    """Name CSV tables by basename; full source paths disambiguate collisions."""
    if len(set(source_tables)) != len(source_tables):
        raise ValueError("Duplicate Garmin source table paths")
    groups = defaultdict(list)
    for source in source_tables:
        groups[slug(PurePosixPath(source).stem)].append(source)
    return {source: f"{base}{f'__{index:02d}' if len(group) > 1 else ''}.parquet"
            for base, group in groups.items()
            for index, source in enumerate(sorted(group), 1)}


def xsens_table_names(titles: list[str]) -> dict[str, str]:
    """Worksheet names survive reordering; sanitized collisions get suffixes."""
    if len(set(titles)) != len(titles):
        raise ValueError("Duplicate Xsens worksheet titles")
    groups = defaultdict(list)
    names = {}
    for title in titles:
        if title == "General Information":
            names[title] = "general_information.json"
        else:
            groups[slug(title)].append(title)
    for base, group in groups.items():
        for index, title in enumerate(sorted(group), 1):
            suffix = f"__{index:02d}" if len(group) > 1 else ""
            names[title] = f"{base}{suffix}.parquet"
    return names
