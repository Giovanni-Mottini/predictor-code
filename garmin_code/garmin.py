"""Garmin preparation and window features, shared by notebooks and the CLI.

No raw data edits, first-wave eligibility sheets, forced timezones or EMA joins.
Values and their eligibility/review masks are separate objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from shared_code.reader import DataContractError, SubjectReader
from shared_code.progress import ProgressReporter
from .garmin_reader import check_snapshot, open_export
from .garmin_hrv import ADVANCED_HRV, EPOCH_COLUMNS, advanced_block, continuity_mask, method as hrv_method
from .garmin_hrv_notebook import block as notebook_block, method as notebook_method

METHOD_ID = "garmin-local-windows-explicit-bbi-v4"
STREAM_FILES = {
    "health": "detailed-health-log.csv",
    "stress": "stress.csv",
    "respiration": "respiration.csv",
    "bbi": "bbi-(raw).csv",
    "sleep": "sleep.csv",
    "rhr": "daily-health-log.csv",
    "heart_rate": "heart-rate.csv",
}
HEALTH_FEATURES = (
    "calories_kcal",
    "total_calories_kcal",
    "active_calories_kcal",
    "steps",
    "mean_motion_intensity",
    "met_avg",
    "mvpa_mins",
    "moderate_intensity_s",
    "vigorous_intensity_s",
    "stress_qualifier",
    # Fitrockr's own 5-minute HRV, kept beside the in-house BBI computation.
    "rmssd",
    "sdnn",
    "sd1_sd2",
    "lf_hf",
)
FITROCKR_HRV = ("rmssd", "sdnn", "sd1_sd2", "lf_hf")
FEATURES = {
    "health": HEALTH_FEATURES,
    "stress": ("stress_level_value", "body_battery"),
    "respiration": ("respiration_breaths_min",),
    "bbi": ("bbi_ms",),
    "sleep": ("sleep_score_value",),
    "rhr": ("resting_heart_rate_avg",),
    "heart_rate": ("heart_rate_bpm",),
}
HRV_FIELDS = (
    "n_beats",
    "n_adjacent_pairs",
    "SDNN",
    "RMSSD",
    "SDSD",
    "NN50",
    "pNN50",
    "NN20",
    "pNN20",
)
# Kept as an empty compatibility constant for earlier inspection notebooks.
DEFERRED_HRV = ()


@dataclass(frozen=True)
class GarminConfig:
    window_minutes: int = 30
    calorie_rate_kcal_min: float = 25.0
    bbi_min_ms: float = 230.0
    bbi_max_ms: float = 1550.0
    max_pair_gap_s: float = 5.0
    max_selected_rows: int = 2_000_000
    hrv_entropy_beats: int = 1000
    hrv_clock_tolerance_ms: float = 50.0
    doubled_bin_ratio: float = 1.05
    # Beat screens and the HRV method stay switchable until the professor decides.
    hrv_method: str = "notebook"
    bbi_screens: bool = False
    exclude_second_stream: bool = False
    bbi_local_median_pct: float | None = None

    def __post_init__(self):
        if self.hrv_method not in {"notebook", "segments"}:
            raise DataContractError("invalid_config", "HRV method must be notebook or segments")
        # Each window belongs to exactly one six-hour block and one local day.
        if (
            not isinstance(self.window_minutes, int)
            or self.window_minutes <= 0
            or 360 % self.window_minutes
        ):
            raise DataContractError(
                "invalid_window", "Window minutes must be a positive divisor of 360"
            )
        for value in (
            self.calorie_rate_kcal_min,
            self.bbi_min_ms,
            self.bbi_max_ms,
            self.max_pair_gap_s,
            self.hrv_clock_tolerance_ms,
        ):
            if not np.isfinite(value) or value <= 0:
                raise DataContractError(
                    "invalid_config", "QC parameters must be finite and positive"
                )
        if self.bbi_min_ms >= self.bbi_max_ms or self.max_selected_rows < 1:
            raise DataContractError("invalid_config", "Invalid BBI range or row limit")
        if not np.isfinite(self.doubled_bin_ratio) or self.doubled_bin_ratio <= 1:
            raise DataContractError("invalid_config", "Doubled-stream ratio must exceed 1")
        if self.bbi_local_median_pct is not None and (not np.isfinite(self.bbi_local_median_pct) or self.bbi_local_median_pct <= 0):
            raise DataContractError("invalid_config", "Local-median screen must be positive or unset")
        if (type(self.hrv_entropy_beats) is not int
                or not 1000 <= self.hrv_entropy_beats <= 10000):
            raise DataContractError("invalid_config", "Entropy epochs require 1000–10000 consecutive beats")

    def method(self):
        value = {
            "method_id": METHOD_ID,
            "config": self.__dict__,
            "time_basis": "source local wall clock; UTC validates ordering and interval membership",
            "date_range": "explicit local calendar dates [start, end)",
            "window_rule": "measurement start in [start, end); means, never sums",
            "daily_rule": "RHR start date, ambiguous days null; sleep: sessions <2h apart merged (mean score), longest merged session per local end date, UTC span incl. awake",
            "bbi_source": "raw BBI only; Enhanced BBI is not read (it fills detection gaps with synthetic long intervals)",
            "hrv_rule": "notebook: all block beats in time order, see notebook_hrv; segments: SDNN sample std; differences only original adjacent rows, same source/table, positive UTC delta <= max_pair_gap_s and equal to the following BBI within hrv_clock_tolerance_ms, no declared gap",
            "notebook_hrv": notebook_method(),
            "bbi_screens": "when bbi_screens: 230-1550 ms range, zero-wear intervals and MET>=3 exclude beats; otherwise every timestamp-valid raw beat is used",
            "advanced_hrv": hrv_method(),
            "bbi_range_origin": "inherited 04_bbi.ipynb, configurable screening range, not clinical validation",
            "calorie_rule": "separate source fields; no correction; negative excluded; high rate retained for review, omitted from normalization",
            "calorie_threshold_origin": "inherited exploration script, heuristic, not validated physiological limit",
            "coverage_rule": "observed valid health interval seconds / window seconds; missing wear remains null",
            "stream_screens": {
                "stress": "finite scale [0,100] and positive finite heart-rate reference at the same UTC timestamp",
                "body_battery": "finite scale [0,100], source status in [1,2,3,125]; independent of stress eligibility",
                "respiration": "finite positive value and source status 0",
                "bbi_rest": "measured wearing>0 and not MET>=3 among eligible MET; unknown MET is kept and counted (info)",
                "bbi_second_stream": "when exclude_second_stream: 5-min UTC bins whose in-range raw BBI sum exceeds doubled_bin_ratio x 300 s, plus one bin either side, exclude BBI, stress and minute HR",
                "bbi_local_median": "exclude beats deviating > bbi_local_median_pct from the centred 31-beat median (unset disables)",
                "rhr": "positive finite value; inherited open (30,120) screen is review only",
                "sleep": "consistent repeated session summary, score [0,100], positive UTC session duration <=24h",
                "origins": "inherited preparation notebooks; computational screens, not clinical validation",
            },
            "zero_coded_missing": {
                "met_avg": "0 excluded (MET>=1 by definition); also excluded in 15-min quarters containing a worn interval with missing calories",
                "total_calories": "worn days where >90% of positive totals equal active (resting energy absent) excluded; elsewhere total<=active is exercise, kept for review",
                "dropped_record": "worn interval with total 0 outside such days: total, active calories and steps excluded together",
                "steps": "excluded when no worn interval in the selected period has positive steps",
                "fitrockr_hrv": "values <=0 excluded; same wear and MET>=3 screen as BBI; sd2_sd1 = 1/sd1_sd2",
            },
            "stress_qualifier_mapping": {"REST": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3},
            "units": {
                "source_features": "preserved descriptor units; unknown units remain unknown",
                "SDNN_RMSSD_SDSD": "ms derived from declared BBI ms",
                "NN20_NN50_n_beats_n_adjacent_pairs": "count",
                "pNN20_pNN50": "% of accepted adjacent pairs",
                **hrv_method()["units"],
                "sleep_session_hours": "h of UTC session span, including awake time",
                "normalization": "dimensionless within-participant range",
            },
            "deferred_hrv_fields": list(DEFERRED_HRV),
            "normalization": "within participant and selected period; finite non-review features only; constant observed features -> 0; missing stays missing; descriptive, not a train/test fit",
        }
        value["method_sha256"] = hashlib.sha256(
            json.dumps(value, sort_keys=True).encode()
        ).hexdigest()
        return value


@dataclass
class PreparedStream:
    name: str
    data: pd.DataFrame
    values: pd.DataFrame
    eligible: pd.DataFrame
    review: pd.DataFrame
    row_eligible: pd.Series
    findings: list[dict] = field(default_factory=list)


def numeric(series):
    return pd.to_numeric(series, errors="coerce").astype(float)


def within_intervals(points, starts, ends):
    """Membership in a union of UTC [start, end) intervals, including overlaps."""
    if len(starts) == 0:
        return np.zeros(len(points), dtype=bool)
    order = np.argsort(starts)
    starts, ends = (
        np.asarray(starts)[order],
        np.maximum.accumulate(np.asarray(ends)[order]),
    )
    at = np.searchsorted(starts, np.asarray(points), side="right") - 1
    return (at >= 0) & (np.asarray(points) < ends[np.maximum(at, 0)])


class GarminSubject:
    def __init__(
        self, root: Path, start: str, end: str, config: GarminConfig | None = None
    ):
        self.reader = SubjectReader(root)
        try:
            first, last = date.fromisoformat(start), date.fromisoformat(end)
        except ValueError as exc:
            raise DataContractError(
                "invalid_dates", "Use ISO calendar dates YYYY-MM-DD"
            ) from exc
        if (
            first.isoformat() != start
            or last.isoformat() != end
            or first >= last
            or (last - first).days > 366
        ):
            raise DataContractError(
                "invalid_dates", "Require canonical dates and 0 < period <= 366 days"
            )
        self.start, self.end = start, end
        self.config = config or GarminConfig()
        self.exports = [
            open_export(self.reader, r) for r in self.reader.select(modality="garmin")
        ]
        if not self.exports:
            raise DataContractError("empty_garmin_selection", "No mapped Garmin export")
        self.prepared: dict[str, PreparedStream] = {}
        self.hrv_blocks = pd.DataFrame()
        self.hrv_epochs = pd.DataFrame(columns=EPOCH_COLUMNS)
        self._doubled = None

    @property
    def participant_key(self):
        return self.reader.participant_key

    def doubled_bins(self):
        """5-min UTC bins holding more beat time than clock time: a second beat stream.

        One heart cannot exceed 300 s of intervals per 5 min; normal bins stay below
        1.003. Always read from raw BBI, which carries the second stream reliably.
        """
        if self._doubled is None:
            raw = self._load("bbi")
            utc = pd.to_datetime(raw["start_time_utc"], format="ISO8601", errors="coerce", utc=True).dt.tz_localize(None)
            bbi = numeric(raw["bbi_ms"])
            ok = utc.notna() & bbi.between(self.config.bbi_min_ms, self.config.bbi_max_ms)
            load = bbi[ok].groupby(utc[ok].dt.floor("5min")).sum() / 300_000
            bins = pd.DatetimeIndex(load.index[load.gt(self.config.doubled_bin_ratio)])
            step = pd.Timedelta(minutes=5)
            self._doubled = bins.union(bins - step).union(bins + step)
        return self._doubled

    def _load(self, name, filename=None):
        filename = filename or STREAM_FILES[name]
        candidates = [
            (export, table)
            for export in self.exports
            for table in export.descriptor["tables"]
            if Path(table["source_table"]).name == filename
        ]
        if len(candidates) != 1:
            raise DataContractError(
                "table_missing_or_ambiguous",
                f"Expected one selected {filename}; found {len(candidates)}; no source fallback",
            )
        export, table = candidates[0]
        data = export.read_period(
            table,
            start=self.start,
            end=self.end,
            date_column="end_time_local" if name == "sleep" else "start_time_local",
            max_rows=self.config.max_selected_rows,
        )
        data["participant_key"] = self.participant_key
        required = {"start_time_local", "start_time_utc", "time_zone_s"}
        if name in {"health", "sleep"}:
            required |= {"end_time_local", "end_time_utc"}
        required |= (
            set(FEATURES[name]) - set(HEALTH_FEATURES)
            if name == "health"
            else set(FEATURES[name])
        )
        if not required <= set(data):
            raise DataContractError(
                "column_missing", f"{name}: missing {sorted(required-set(data))}"
            )
        declared = {c["name"]: c for c in table["columns"]}
        for column, unit in {
            "bbi_ms": "ms",
            "respiration_breaths_min": "breaths/min",
            "calories_kcal": "kcal",
            "total_calories_kcal": "kcal",
            "active_calories_kcal": "kcal",
            "time_zone_s": "s",
        }.items():
            if column in declared and declared[column].get("unit") != unit:
                raise DataContractError(
                    "unit_mismatch", f"{name}/{column}: expected declared {unit}"
                )
        return data

    def prepare(self, name: str) -> PreparedStream:
        if name in self.prepared:
            return self.prepared[name]
        if name not in STREAM_FILES:
            raise ValueError(f"Unknown Garmin stream {name}")
        data = self._load(name)
        data["_local"] = pd.to_datetime(
            data["start_time_local"], format="ISO8601", errors="coerce"
        )
        if isinstance(data["_local"].dtype, pd.DatetimeTZDtype):
            raise DataContractError(
                "local_time_not_naive",
                "Source local clock must not contain an assumed timezone",
            )
        data["_utc"] = pd.to_datetime(
            data["start_time_utc"], format="ISO8601", errors="coerce", utc=True
        ).dt.tz_localize(None)
        zone = numeric(data["time_zone_s"])
        seconds = (data["_local"] - data["_utc"]).dt.total_seconds()
        base = (
            data["_local"].notna()
            & data["_utc"].notna()
            & zone.notna()
            & seconds.sub(zone).abs().le(0.001)
        )
        values = pd.DataFrame(index=data.index)
        for col in FEATURES[name]:
            if col not in data:
                continue
            values[col] = (
                data[col].map({"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 3.0, "REST": 0.0})
                if col == "stress_qualifier"
                else numeric(data[col])
            )
        eligible = values.notna() & np.isfinite(values)
        review = pd.DataFrame(False, index=data.index, columns=values.columns)
        result = PreparedStream(name, data, values, eligible, review, base.copy())

        def flag(code, mask, feature=None, severity="exclude", rule=""):
            mask = pd.Series(mask, index=data.index).fillna(False).astype(bool)
            if not mask.any():
                return
            rows = data.loc[mask]
            examples = []
            for i, row in rows.head(5).iterrows():
                example = {
                    "source_id": row["source_id"],
                    "source_table": row["source_table"],
                    "source_row": int(row["source_row"]),
                    "source_csv_record": int(row["source_row"]) + 1,
                    "start_time_local": row["start_time_local"],
                    "start_time_utc": row["start_time_utc"],
                }
                if feature in data:
                    v = row[feature]
                    example["raw_value"] = None if pd.isna(v) else str(v)
                examples.append(example)
            result.findings.append(
                {
                    "code": code,
                    "feature": feature,
                    "severity": severity,
                    "count": int(mask.sum()),
                    "rule": rule,
                    "examples": examples,
                }
            )
            if severity == "review" and feature in review:
                review.loc[mask, feature] = True
            elif severity == "exclude":
                if feature in eligible:
                    eligible.loc[mask, feature] = False
                else:
                    result.row_eligible.loc[mask] = False

        flag(
            "invalid_or_inconsistent_timestamp",
            ~base,
            rule="local - UTC must equal declared offset; no correction",
        )
        epoch_col = (
            "start_time_ms"
            if "start_time_ms" in data
            else "start_time_s"
            if "start_time_s" in data
            else None
        )
        if epoch_col:
            epoch = numeric(data[epoch_col])
            parsed = pd.to_datetime(
                epoch, unit="ms" if epoch_col.endswith("ms") else "s", errors="coerce"
            )
            flag(
                "epoch_timestamp_mismatch",
                parsed.isna()
                | (parsed - data["_utc"]).dt.total_seconds().abs().gt(0.001),
            )
        for col in values:
            flag("missing_or_nonfinite_value", ~eligible[col], col)

        if name == "sleep":
            # Phase rows repeat a session summary. Collapse only equal summaries.
            data["_end_utc"] = pd.to_datetime(
                data["end_time_utc"], format="ISO8601", errors="coerce", utc=True
            ).dt.tz_localize(None)
            keys = ["_utc", "_end_utc"]
            summary = [*keys, "start_time_local", "end_time_local", "sleep_score_value"]
        else:
            keys = ["_utc"]
            summary = [
                c
                for c in data
                if not c.startswith("_")
                and c
                not in {"source_id", "source_row", "source_table", "participant_key"}
            ]
        equal = data.duplicated(subset=summary, keep="first")
        flag(
            "equivalent_summary_duplicate"
            if name == "sleep"
            else "exact_observation_duplicate",
            equal,
            severity="exclude",
            rule="keep first equivalent observation; raw rows remain available",
        )
        active = data.loc[result.row_eligible]
        conflict = active.duplicated(subset=keys, keep=False)
        flag("conflicting_utc_observation", data.index.isin(active.index[conflict]))
        active = data.loc[result.row_eligible]
        local_conflict = active.duplicated(subset=["_local"], keep=False)
        flag(
            "ambiguous_local_clock",
            data.index.isin(active.index[local_conflict]),
            rule="distinct UTC observations share local clock; do not combine",
        )

        if name == "health":
            for col in ("wearing_time", "met_avg"):
                if col not in data:
                    raise DataContractError("column_missing", f"Health requires {col}")
            wear = numeric(data["wearing_time"])
            flag(
                "invalid_wearing_percentage", ~np.isfinite(wear) | ~wear.between(0, 100)
            )
            # Zero-wear rows remain eligible for measured wear and coverage.
            for col in values:
                flag("zero_wear_measurement", wear.eq(0), col)
                if col != "stress_qualifier":
                    flag("negative_measurement", values[col].lt(0), col)
            duration = (
                pd.to_datetime(
                    data["end_time_utc"], format="ISO8601", errors="coerce", utc=True
                ).dt.tz_localize(None)
                - data["_utc"]
            ).dt.total_seconds()
            local_end = pd.to_datetime(
                data["end_time_local"], format="ISO8601", errors="coerce"
            )
            local_duration = (local_end - data["_local"]).dt.total_seconds()
            data["_end_local"], data["_duration_s"] = local_end, duration
            flag(
                "invalid_health_interval",
                duration.isna()
                | duration.le(0)
                | duration.gt(300)
                | local_duration.ne(duration),
                rule="positive UTC interval <=300s and matching local span; transition intervals excluded",
            )
            active = data.loc[result.row_eligible].sort_values("_utc")
            overlap = active["_utc"].lt(
                (active["_utc"] + pd.to_timedelta(active["_duration_s"], unit="s"))
                .cummax()
                .shift()
            )
            # Exclude both members of overlapping intervals instead of overcounting.
            bad = set(active.index[overlap])
            for idx in active.index[overlap]:
                start_time = active.loc[idx, "_utc"]
                bad.update(
                    active.index[
                        (active["_utc"] < start_time)
                        & (
                            active["_utc"]
                            + pd.to_timedelta(active["_duration_s"], unit="s")
                            > start_time
                        )
                    ]
                )
            flag("overlapping_health_intervals", data.index.isin(bad))
            for col in ("calories_kcal", "total_calories_kcal", "active_calories_kcal"):
                if col in values:
                    flag(
                        "high_calorie_rate",
                        values[col].gt(
                            self.config.calorie_rate_kcal_min * duration / 60
                        ),
                        col,
                        "review",
                        f"source kcal > {self.config.calorie_rate_kcal_min:g} * UTC duration minutes; exploratory",
                    )
            # Fitrockr writes 0 where it could not compute a value. Garmin's own
            # dailies carry the same gaps, so they are upstream, not ours.
            worn = wear.gt(0) & result.row_eligible
            calorie_gap = pd.Series(False, index=data.index)
            if {"total_calories_kcal", "active_calories_kcal"} <= set(values):
                total, active = values["total_calories_kcal"], values["active_calories_kcal"]
                day = data["_local"].dt.date
                positive = worn & total.gt(0)
                share = (positive & total.le(active)).groupby(day).sum() / positive.groupby(day).sum()
                no_resting_day = day.map(share.gt(0.9)).fillna(False).astype(bool) & worn
                flag("resting_energy_missing_day", no_resting_day, "total_calories_kcal",
                     rule="worn day where >90% of positive totals equal active calories; total is incomplete")
                dropped = worn & ~no_resting_day & total.eq(0)
                for col in ("total_calories_kcal", "active_calories_kcal", "steps"):
                    if col in values:
                        flag("calorie_step_record_missing", dropped, col,
                             rule="worn interval with 0 total kcal: calories, steps and distance arrive together and are all 0")
                # Outside such days total<=active marks exercise (HR ~105 vs ~70 bpm), not a gap.
                flag("total_not_above_active", worn & ~no_resting_day & total.gt(0) & total.le(active),
                     "total_calories_kcal", "info", "worn total <= active on a day with resting energy; kept")
                calorie_gap = no_resting_day | dropped
            if "met_avg" in values:
                quarter = data["_local"].dt.floor("15min")
                gap_quarter = quarter.map(calorie_gap.groupby(quarter).any()).fillna(False).astype(bool)
                flag("met_missing_coded_as_zero", values["met_avg"].eq(0), "met_avg",
                     rule="MET is total/resting energy and >=1 by definition; 0 means not computed")
                flag("met_from_incomplete_calories", gap_quarter & values["met_avg"].gt(0), "met_avg",
                     rule="15-min MET quarter contains a worn interval with missing calories")
                # The export's own formula: it reproduces 100% of valid exported quarters
                # in the current format and in old-format quarters (report 2026-10-21).
                data["_met_recomputed"] = False
                if {"total_calories_kcal", "active_calories_kcal"} <= set(values):
                    good = worn & ~no_resting_day & total.gt(active)
                    parts = pd.DataFrame({"good": good, "total": total.where(good), "resting": (total - active).where(good)})
                    q = parts.groupby(quarter).agg(rows=("good", "size"), good=("good", "sum"),
                                                    total=("total", "sum"), resting=("resting", "sum"))
                    full = q[(q.rows == 3) & (q.good == 3)]
                    rebuilt = quarter.map(full.total / full.resting)
                    target = values["met_avg"].eq(0) & rebuilt.notna()
                    values.loc[target, "met_avg"] = rebuilt[target]
                    eligible.loc[target, "met_avg"] = True
                    data["_met_recomputed"] = target
                    flag("met_recomputed", target, "met_avg", "info",
                         "zero-coded MET recomputed as sum(total)/sum(total-active) over a 15-min quarter of 3 intact intervals")
            if "steps" in values and worn.any() and not (worn & values["steps"].gt(0)).any():
                flag("steps_never_recorded", worn, "steps",
                     rule="no worn interval in the selected period has positive steps; 0 is not a measurement")
            for col in FITROCKR_HRV:
                if col in values:
                    flag("missing_coded_as_zero", values[col].le(0), col, rule="Fitrockr HRV <=0 is not computed")
            if "met_avg" in values and self.config.bbi_screens:
                exercise = eligible["met_avg"] & values["met_avg"].ge(3)
                for col in FITROCKR_HRV:
                    if col in values:
                        flag("outside_observed_rest", exercise, col, rule="same MET>=3 screen as BBI")
            if "sd1_sd2" in values:
                values["sd2_sd1"] = 1 / values["sd1_sd2"].where(values["sd1_sd2"].gt(0))
                eligible["sd2_sd1"] = eligible["sd1_sd2"] & values["sd2_sd1"].notna()
                review["sd2_sd1"] = False
        elif name in {"stress", "respiration", "bbi"}:
            health = self.prepare("health")
            h = health.data.loc[health.row_eligible]
            zero = h.loc[numeric(h["wearing_time"]).eq(0)]
            points = data["_utc"].to_numpy(dtype="datetime64[ns]")
            starts = zero["_utc"].to_numpy(dtype="datetime64[ns]")
            ends = starts + pd.to_timedelta(zero["_duration_s"], unit="s").to_numpy()
            if name != "bbi" or self.config.bbi_screens:
                flag("zero_wear_interval", within_intervals(points, starts, ends))
            if name == "stress":
                reference = self.prepare("heart_rate")
                r = reference.data.loc[
                    reference.row_eligible
                    & reference.eligible["heart_rate_bpm"]
                    & reference.values["heart_rate_bpm"].gt(0)
                ]
                flag(
                    "stress_without_heart_rate",
                    ~data["_utc"].isin(r["_utc"]),
                    "stress_level_value",
                )
                if "body_battery_status" not in data:
                    flag(
                        "missing_body_battery_status",
                        np.ones(len(data), dtype=bool),
                        "body_battery",
                    )
                else:
                    flag(
                        "invalid_body_battery_status",
                        ~numeric(data["body_battery_status"]).isin([1, 2, 3, 125]),
                        "body_battery",
                    )
                flag(
                    "stress_out_of_scale",
                    ~values["stress_level_value"].between(0, 100),
                    "stress_level_value",
                )
                flag(
                    "body_battery_out_of_scale",
                    ~values["body_battery"].between(0, 100),
                    "body_battery",
                )
            elif name == "respiration":
                if "status" not in data:
                    flag(
                        "missing_respiration_status",
                        np.ones(len(data), dtype=bool),
                        "respiration_breaths_min",
                    )
                else:
                    flag(
                        "invalid_respiration_status",
                        ~numeric(data["status"]).eq(0),
                        "respiration_breaths_min",
                    )
                flag(
                    "nonpositive_respiration",
                    values["respiration_breaths_min"].le(0),
                    "respiration_breaths_min",
                )
            elif self.config.bbi_screens:
                # Zero-coded MET is unknown, not rest: keep those beats but count them.
                met_known = health.eligible.loc[h.index, "met_avg"]
                met = health.values.loc[h.index, "met_avg"]
                worn = numeric(h["wearing_time"]).gt(0)

                def inside(rows):
                    rstart = rows["_utc"].to_numpy(dtype="datetime64[ns]")
                    rend = rstart + pd.to_timedelta(rows["_duration_s"], unit="s").to_numpy()
                    return within_intervals(points, rstart, rend)

                flag(
                    "outside_observed_rest",
                    ~inside(h.loc[worn & ~(met_known & met.ge(3))]),
                    "bbi_ms",
                    rule="observed wearing>0 and not MET>=3 among eligible MET",
                )
                flag("rest_unverified_met_unknown", inside(h.loc[worn & ~met_known]), "bbi_ms", "info",
                     "MET missing or zero-coded; beat kept because MET>=3 covers ~1% of worn time")
                flag(
                    "bbi_outside_screen",
                    ~values["bbi_ms"].between(
                        self.config.bbi_min_ms, self.config.bbi_max_ms
                    ),
                    "bbi_ms",
                )
        elif name == "sleep":
            data["_end_local"] = pd.to_datetime(
                data["end_time_local"], format="ISO8601", errors="coerce"
            )
            endutc = pd.to_datetime(
                data["end_time_utc"], format="ISO8601", errors="coerce", utc=True
            ).dt.tz_localize(None)
            data["_duration_s"] = (endutc - data["_utc"]).dt.total_seconds()
            flag(
                "invalid_sleep_interval",
                data["_end_local"].isna()
                | endutc.isna()
                | data["_duration_s"].le(0)
                | data["_duration_s"].gt(86400),
            )
            flag(
                "sleep_score_out_of_scale",
                ~values["sleep_score_value"].between(0, 100),
                "sleep_score_value",
            )
        elif name == "rhr":
            flag(
                "nonpositive_rhr",
                values["resting_heart_rate_avg"].le(0),
                "resting_heart_rate_avg",
            )
            flag(
                "rhr_outside_inherited_screen",
                values["resting_heart_rate_avg"].le(30)
                | values["resting_heart_rate_avg"].ge(120),
                "resting_heart_rate_avg",
                "review",
                "inherited open (30,120) screen; not clinical validity",
            )

        target = {"bbi": "bbi_ms", "stress": "stress_level_value", "heart_rate": "heart_rate_bpm"}.get(name)
        if target:
            # Detected always; excluded only when exclude_second_stream is set.
            flag("second_beat_stream", data["_utc"].dt.floor("5min").isin(self.doubled_bins()), target,
                 "exclude" if self.config.exclude_second_stream else "info",
                 f"raw BBI sum > {self.config.doubled_bin_ratio:g} x 300 s in the 5-min bin or a neighbour")
        if name == "bbi" and self.config.bbi_local_median_pct:
            ok = result.row_eligible & eligible["bbi_ms"]
            beats = values.loc[data.loc[ok].sort_values("_utc").index, "bbi_ms"]
            median = beats.rolling(31, center=True).median()
            flag("bbi_local_median_outlier",
                 ((beats - median).abs() > median * self.config.bbi_local_median_pct / 100).reindex(data.index, fill_value=False),
                 "bbi_ms", rule=f"> {self.config.bbi_local_median_pct:g}% from centred 31-beat median")

        eligible.loc[~result.row_eligible, :] = False
        self.prepared[name] = result
        check_snapshot(self.reader, self.exports)
        return result

    def report(self):
        check_snapshot(self.reader, self.exports)
        return {
            "participant_key": self.participant_key,
            "start_date": self.start,
            "end_date_exclusive": self.end,
            "download_completeness": self.reader.subject.get(
                "download_completeness", "unknown"
            ),
            "streams": {
                name: {
                    "selected_rows": len(p.data),
                    "eligible_counts": {c: int(p.eligible[c].sum()) for c in p.values},
                    "review_counts": {
                        c: int((p.review[c] & p.eligible[c]).sum()) for c in p.values
                    },
                    "findings": p.findings,
                }
                for name, p in self.prepared.items()
            },
            "provenance": [e.provenance() for e in self.exports],
            "hrv": {
                "blocks": len(self.hrv_blocks),
                "epochs": len(self.hrv_epochs),
                "computed_blocks": {
                    name: int(self.hrv_blocks[name].notna().sum()) for name in ADVANCED_HRV
                } if len(self.hrv_blocks) else {},
                "reviewed_blocks": {
                    name: int((self.hrv_blocks[name].notna() & self.hrv_blocks[f"{name}_review"]).sum())
                    for name in ADVANCED_HRV
                } if len(self.hrv_blocks) else {},
            },
            "ready_for_features": False,
        }


def time_domain(bbi: np.ndarray, adjacent: np.ndarray):
    diffs = np.diff(bbi)[adjacent]
    result = {c: np.nan for c in HRV_FIELDS}
    result.update(n_beats=len(bbi), n_adjacent_pairs=len(diffs))
    if len(bbi) > 1:
        result["SDNN"] = float(np.std(bbi, ddof=1))
    if len(diffs):
        result.update(
            RMSSD=float(np.sqrt(np.mean(diffs ** 2))),
            NN50=int((np.abs(diffs) > 50).sum()),
            NN20=int((np.abs(diffs) > 20).sum()),
        )
        result["pNN50"], result["pNN20"] = (
            result["NN50"] * 100 / len(diffs),
            result["NN20"] * 100 / len(diffs),
        )
    if len(diffs) > 1:
        result["SDSD"] = float(np.std(diffs, ddof=1))
    return result


def build_windows(run: GarminSubject, *, progress: ProgressReporter | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return provisional features, review-aware normalized values, feature names."""
    freq = f"{run.config.window_minutes}min"
    starts = pd.date_range(run.start, run.end, freq=freq, inclusive="left")
    windows = pd.DataFrame(
        {
            "participant_key": run.participant_key,
            "window_start": starts,
            "window_end": starts + pd.Timedelta(minutes=run.config.window_minutes),
        }
    )
    windows["date"] = windows["window_start"].dt.strftime("%Y-%m-%d")
    windows["circadian_cycle"] = windows["window_start"].dt.hour // 6
    windows["time_basis"] = "source_local_wall_clock"
    names = []
    for name in ("health", "stress", "respiration"):
        if progress:
            progress.update(f"{run.participant_key}: preparing/aggregating {name} stream")
        p = run.prepare(name)
        group = p.data["_local"].dt.floor(freq)
        for col in p.values:
            feature = f"{col}_mean"
            names.append(feature)
            usable = p.values[col].where(p.eligible[col])
            windows[feature] = windows["window_start"].map(usable.groupby(group).mean())
            windows[f"{feature}_n"] = (
                windows["window_start"]
                .map(p.eligible[col].groupby(group).sum())
                .fillna(0)
                .astype(int)
            )
            windows[f"{feature}_review"] = (
                windows["window_start"]
                .map((p.review[col] & p.eligible[col]).groupby(group).any())
                .fillna(False)
                .astype(bool)
            )
        if "_met_recomputed" in p.data:
            windows["met_avg_mean_recomputed_n"] = (
                windows["window_start"].map((p.data["_met_recomputed"] & p.eligible["met_avg"]).groupby(group).sum())
                .fillna(0).astype(int)
            )
    if progress:
        progress.update(f"{run.participant_key}: calculating health coverage for {len(windows)} windows")
    h = run.prepare("health")
    observed = h.data.loc[h.row_eligible]
    coverage, wear = [], []
    for index, (start, end) in enumerate(zip(windows["window_start"], windows["window_end"]), 1):
        part = observed.loc[
            observed["_local"].lt(end) & observed["_end_local"].gt(start)
        ]
        overlap = (
            part["_end_local"].clip(upper=end) - part["_local"].clip(lower=start)
        ).dt.total_seconds()
        coverage.append(float(overlap.sum() / (run.config.window_minutes * 60) * 100))
        wear.append(
            float(
                (overlap * numeric(part["wearing_time"]) / 100).sum()
                / overlap.sum()
                * 100
            )
            if overlap.sum()
            else np.nan
        )
        if progress and (index % max(1, len(windows) // 10) == 0 or index == len(windows)):
            progress.update(f"{run.participant_key}: health coverage windows processed", completed=index, total=len(windows))
    windows["health_coverage_pct"], windows["wearing_time_pct_observed"] = (
        coverage,
        wear,
    )
    if progress:
        progress.update(f"{run.participant_key}: preparing BBI stream for HRV")
    p = run.prepare("bbi")
    block = p.data["_local"].dt.floor("6h")
    hrv = []
    epochs = []
    block_starts = pd.date_range(run.start, run.end, freq="6h", inclusive="left")
    for index, start in enumerate(block_starts, 1):
        if progress:
            progress.update(f"{run.participant_key}: calculating HRV block starting {start}",
                            completed=index - 1, total=len(block_starts))
        idx = p.data.index[block.eq(start) & p.eligible["bbi_ms"]]
        selected = p.data.loc[idx].sort_values("_utc")
        bbi = p.values.loc[selected.index, "bbi_ms"].to_numpy()
        gaps = selected["_utc"].diff().dt.total_seconds().to_numpy()[1:]
        if run.config.hrv_method == "notebook":
            hrv.append({"hrv_block_start": start, **notebook_block(bbi),
                        "hrv_pairs_excluded": 0, "hrv_max_retained_gap_s": float(np.max(gaps)) if len(gaps) else np.nan,
                        "hrv_clock_mismatch_pairs": 0, "hrv_segments": 1 if len(bbi) else 0,
                        "hrv_longest_segment_s": float(gaps.sum()) if len(gaps) else np.nan})
            evidence = []
        else:
            adjacent = (
                (np.diff(selected["source_row"].to_numpy(dtype=int)) == 1)
                & (gaps > 0)
                & (gaps <= run.config.max_pair_gap_s)
            )
            if len(selected) > 1:
                adjacent &= (
                    selected["source_id"].to_numpy()[1:]
                    == selected["source_id"].to_numpy()[:-1]
                )
                # A missed beat or an interleaved row leaves a time step unequal to the interval.
                adjacent &= np.abs(gaps - bbi[1:] / 1000) <= run.config.hrv_clock_tolerance_ms / 1000 + 1e-12
            adjacent, _ = continuity_mask(selected, adjacent)
            advanced, evidence = advanced_block(selected, bbi, adjacent, run.config)
            epochs.extend({"hrv_block_start": start, **row} for row in evidence)
            hrv.append(
                {
                    "hrv_block_start": start,
                    **time_domain(bbi, adjacent),
                    **advanced,
                    "hrv_pairs_excluded": int(len(adjacent) - adjacent.sum()),
                    "hrv_max_retained_gap_s": float(np.max(gaps)) if len(gaps) else np.nan,
                }
            )
        if progress:
            progress.update(f"{run.participant_key}: HRV block processed; {len(bbi)} retained beat(s), {len(evidence)} epoch(s)",
                            completed=index, total=len(block_starts))
    run.hrv_blocks = pd.DataFrame(hrv)
    run.hrv_blocks.insert(0, "participant_key", run.participant_key)
    run.hrv_epochs = pd.DataFrame(epochs, columns=EPOCH_COLUMNS)
    hrv = run.hrv_blocks.set_index("hrv_block_start")
    windows["hrv_block_start"] = windows["window_start"].dt.floor("6h")
    windows = windows.join(hrv.drop(columns="participant_key"), on="hrv_block_start")
    names.extend((*HRV_FIELDS, *ADVANCED_HRV))
    for name, col, feature in (
        ("rhr", "resting_heart_rate_avg", "rhr_avg"),
        ("sleep", "sleep_score_value", "sleep_score"),
    ):
        if progress:
            progress.update(f"{run.participant_key}: preparing/aggregating daily {name} stream")
        p = run.prepare(name)
        day = p.data["_end_local" if name == "sleep" else "_local"].dt.strftime(
            "%Y-%m-%d"
        )
        active = p.row_eligible
        counts = active.groupby(day).sum()
        multiple_days = counts.index[counts.gt(1)].tolist()
        if multiple_days and not any(
            f["code"] == "multiple_daily_sessions" for f in p.findings
        ):
            p.findings.append(
                {
                    "code": "multiple_daily_sessions",
                    "feature": col,
                    "severity": "review",
                    "count": len(multiple_days),
                    "rule": "sleep: merge sessions <2h apart, keep the longest per local end date"
                            if name == "sleep" else "one unambiguous session per local date; daily values null",
                    "examples": [
                        {"local_date": d, "session_count": int(counts[d])}
                        for d in multiple_days[:5]
                    ],
                }
            )
        if name == "sleep":
            nights = sleep_nights(p)
            valid = nights["score"]
        else:
            valid = (
                p.values[col]
                .where(p.eligible[col])
                .groupby(day)
                .first()
                .where(counts.eq(1))
            )
        windows[feature] = windows["date"].map(valid)
        windows[f"{name}_session_count"] = (
            windows["date"].map(counts).fillna(0).astype(int)
        )
        windows[f"{feature}_review"] = (
            windows["date"]
            .map((p.review[col] & p.eligible[col]).groupby(day).any())
            .fillna(False)
            .astype(bool)
        )
        names.append(feature)
        if name == "sleep":
            windows["sleep_session_hours"] = windows["date"].map(nights["hours"])
            names.append("sleep_session_hours")
    windows["method_id"] = METHOD_ID
    windows["method_sha256"] = run.config.method()["method_sha256"]
    if progress:
        progress.update(f"{run.participant_key}: normalizing {len(windows)} windows and verifying input snapshot")
    normalized, scaling = normalize(windows, names)
    normalized.attrs["scaling"] = scaling
    check_snapshot(run.reader, run.exports)
    return windows, normalized, names


def sleep_nights(p: PreparedStream) -> pd.DataFrame:
    """Notebook merge of sessions <2 h apart (first start, last end, mean score),
    then the longest merged session per local end date; naps are dropped."""
    s = p.data.loc[p.row_eligible, ["_utc", "_end_utc", "_end_local"]].assign(
        score=p.values["sleep_score_value"].where(p.eligible["sleep_score_value"])).sort_values("_utc")
    if s.empty:
        return pd.DataFrame(columns=["score", "hours"])
    gap_h = (s["_utc"] - s["_end_utc"].cummax().shift()).dt.total_seconds() / 3600
    group = (~gap_h.between(0, 2, inclusive="left")).cumsum()
    merged = s.groupby(group).agg(start=("_utc", "first"), end=("_end_utc", "max"),
                                  end_local=("_end_local", "max"), score=("score", "mean"))
    merged["hours"] = (merged["end"] - merged["start"]).dt.total_seconds() / 3600
    merged["day"] = merged["end_local"].dt.strftime("%Y-%m-%d")
    return merged.sort_values("hours").groupby("day").tail(1).set_index("day")[["score", "hours"]]


def normalize(frame: pd.DataFrame, features: list[str]):
    output = frame.copy()
    parameters = []
    for col in features:
        output[col] = output[col].astype(float)
    for key, group in frame.groupby("participant_key", sort=False):
        for col in features:
            series = group[col].astype(float)
            if f"{col}_review" in group:
                series = series.mask(group[f"{col}_review"])
            series = series.where(np.isfinite(series))
            lo, hi = series.min(), series.max()
            output.loc[group.index, col] = (
                (series - lo) / (hi - lo)
                if hi > lo
                else series.where(series.isna(), 0.0)
            )
            parameters.append(
                {
                    "participant_key": key,
                    "feature": col,
                    "observations": int(series.notna().sum()),
                    "min": None if pd.isna(lo) else float(lo),
                    "max": None if pd.isna(hi) else float(hi),
                }
            )
    return output, parameters
