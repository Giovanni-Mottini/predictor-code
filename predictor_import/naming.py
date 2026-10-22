"""Explicit source naming rules. Unrecognized names are never repaired."""
from __future__ import annotations

import re

COHORTS = ("SANI", "PAZIENTI", "CONTROLLO PAZIENTI")
ACTIVITY = r"(?P<activity>task[1-4]|mot[12]|b&b)"
SIDE = r"(?P<side>dx|sx|sim)"
PATTERNS = (
    ("clinical_code", rf"(?P<subject>.+?)_(?P<code>00|01|0|1)_{ACTIVITY}_t(?P<t>\d+)(?:_d(?P<day>\d+))?"),
    ("timepoint_first", rf"(?P<subject>.+?)[_-]t(?P<t>\d+)_{ACTIVITY}_{SIDE}"),
    ("activity_first", rf"(?P<subject>.+?)_{ACTIVITY}_t(?P<t>\d+)_{SIDE}"),
    ("individual_repetition", rf"(?P<subject>.+?)_{ACTIVITY}_rip(?P<rep>\d+)_{SIDE}"),
    ("activity_side", rf"(?P<subject>.+?)_{ACTIVITY}_{SIDE}"),
)


def filename_subject(subject_folder: str, cohort: str) -> str:
    """Only documented folder aliases; do not merge IDs across cohorts."""
    value = subject_folder.upper()
    if cohort == "SANI" and value.startswith("SANO_"):
        value = value[5:]
    if cohort == "PAZIENTI" and value.endswith("_DROPOUT"):
        value = value[:-8]
    if cohort == "CONTROLLO PAZIENTI" and re.fullmatch(r"CTRLFG_\d+", value):
        value = value.replace("_", "")
    return value


def parse_xsens_name(name: str) -> dict | None:
    if not name.lower().endswith(".xlsx"):
        return None
    for variant, expression in PATTERNS:
        match = re.fullmatch(expression, name[:-5], re.IGNORECASE)
        if match:
            groups = match.groupdict()
            activity = groups["activity"].lower().replace("b&b", "box_and_blocks")
            return {
                "naming_variant": variant,
                "filename_subject_id": groups["subject"].upper(),
                "activity": activity,
                "task": int(activity[4:]) if activity.startswith("task") else None,
                "timepoint": f"T{int(groups['t'])}" if groups.get("t") is not None else None,
                "protocol_day": int(groups["day"]) if groups.get("day") else None,
                "repetition": int(groups["rep"]) if groups.get("rep") else None,
                "original_limb_code": groups.get("code") or groups.get("side", "").lower(),
                "anatomical_side": {"dx": "right", "sx": "left"}.get(groups.get("side", "").lower()),
                "representation": "individual_repetition" if groups.get("rep") else "full_recording",
            }
    return None
