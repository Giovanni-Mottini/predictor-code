"""Feature importance on people the model has not seen: Gini and held-out shuffling.

Same settings as the analysis behind the paper (feature_analysis_code, 13 Nov 2026), restricted
to the two comparisons the paper ranks:

  paretic_vs_non_paretic   patients, one arm's repetition -> which arm; the task is an input;
                           each patient weighted once; leave one patient out
  healthy_vs_simulated     volunteers, healthy repetitions (both arms) vs the simulated arm;
                           balanced class weights; 5 folds grouped by person

Per fold, one Random Forest (100 trees, seed 42) is trained on the training people:
  gini            its impurity importance of each feature (the task inputs are not ranked)
  held-out drop   on the test people, the feature is shuffled 20 times inside each person,
                  visit and task, and the loss of balanced accuracy is averaged
Both are then averaged over the folds. Rank 1 = most important (pandas rank, method "min").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from .tables import ARM_FEATURES, task_rows

SEED, TREES, FOLDS, REPEATS = 42, 100, 5, 20
ROW_KEY = ["person", "visit", "task", "repetition", "side", "condition"]
STRATA = ["person", "visit", "task"]     # a feature is shuffled only among these rows


@dataclass(frozen=True)
class Comparison:
    key: str
    title: str
    header: tuple[str, str]              # two lines of the LaTeX column header
    legend: str                          # figure legend
    cohort: str                          # patients or volunteers
    classes: tuple[str, str]             # sorted; the second is the positive class
    weight: str                          # person (each person counts once) or balanced (class weights)
    split: str                           # leave_one_person_out or groupkfold
    task_inputs: bool                    # the task (one-hot) is a model input
    color: str
    marker: str


COMPARISONS = {c.key: c for c in (
    Comparison("paretic_vs_non_paretic", "Paretic vs non-paretic", ("Paretic vs.", "non-paretic"), "Paretic vs non-paretic arm",
               "patients", ("non_paretic", "paretic"), "person", "leave_one_person_out", True, "#D55E00", "o"),
    Comparison("healthy_vs_simulated", "Healthy vs simulated", ("Healthy vs.", "simulated"), "Healthy vs simulated",
               "volunteers", ("healthy", "simulated"), "balanced", "groupkfold", False, "#0072B2", "D"),
)}
VOLUNTEER_CONDITIONS = {"non_paretic": "healthy", "simulated": "simulated"}


def comparison_rows(table: pd.DataFrame, comparison: Comparison) -> pd.DataFrame:
    """Task repetitions of one cohort, checked, without blanks, in a fixed order (the order changes the forest)."""
    rows = task_rows(table)
    if comparison.cohort == "volunteers":
        unknown = set(rows["condition"]) - set(VOLUNTEER_CONDITIONS)
        if unknown:
            raise ValueError(f"Volunteer rows must be healthy or simulated, found {sorted(unknown)}")
        rows = rows.assign(condition=rows["condition"].map(VOLUNTEER_CONDITIONS))
    else:
        if not rows["condition"].isin(comparison.classes).all():
            raise ValueError("Every patient row needs a paretic or non_paretic label")
        for person, group in rows.groupby("person"):
            sides = group.groupby("condition")["side"].unique()
            if len(sides) != 2 or any(len(s) != 1 for s in sides) or sides["paretic"][0] == sides["non_paretic"][0]:
                raise ValueError(f"{person}: arm labels are not one paretic side and the opposite non-paretic side")
    if not rows["side"].isin(["left", "right"]).all():
        raise ValueError("Every row needs a resolved side")
    if rows.duplicated(ROW_KEY).any():
        raise ValueError(f"{int(rows.duplicated(ROW_KEY).sum())} duplicate repetitions")
    recordings = rows.drop_duplicates("recording_id").dropna(subset=["source_sha256"])
    if recordings["source_sha256"].duplicated().any():
        raise ValueError("The same source file appears under more than one recording")
    rows = rows.dropna(subset=list(ARM_FEATURES))
    if rows.duplicated(list(ARM_FEATURES)).any():
        raise ValueError("Two repetitions have identical feature values")
    rows = rows[rows["condition"].isin(comparison.classes)]
    return rows.sort_values(ROW_KEY).reset_index(drop=True)


def design(rows: pd.DataFrame, comparison: Comparison):
    X = rows[list(ARM_FEATURES)].to_numpy(float)
    if comparison.task_inputs:
        X = np.column_stack([X, np.eye(4)[rows["task"].to_numpy(int) - 1]])
    return X, rows["condition"].to_numpy(), rows["person"].to_numpy()


def make_folds(X, y, people, split: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if split == "leave_one_person_out":
        return [(p, np.flatnonzero(people != p), np.flatnonzero(people == p)) for p in sorted(set(people))]
    if split == "groupkfold":
        return [(f"fold{i}", train, test) for i, (train, test) in enumerate(GroupKFold(n_splits=FOLDS).split(X, y, people))]
    raise ValueError(f"Unknown split {split}")


def fit(X, y, people, weight: str) -> RandomForestClassifier:
    forest = RandomForestClassifier(n_estimators=TREES, random_state=SEED, class_weight="balanced" if weight == "balanced" else None)
    if weight == "person":
        counts = pd.Series(people).map(pd.Series(people).value_counts()).to_numpy()
        values = 1 / counts
        return forest.fit(X, y, sample_weight=values / values.mean())
    return forest.fit(X, y)


def strata_of(rows: pd.DataFrame) -> np.ndarray:
    """Shuffle cells. Stops if every cell holds one class: every drop would then be zero by construction."""
    labels = rows[STRATA].astype(str).agg("|".join, axis=1).to_numpy()
    if pd.Series(rows["condition"].to_numpy()).groupby(labels).nunique().max() < 2:
        raise ValueError("Every shuffle cell holds one class, so importance would be zero by construction")
    return labels


def stratified_order(strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    order = np.arange(len(strata))
    for stratum in np.unique(strata):
        members = np.flatnonzero(strata == stratum)
        order[members] = rng.permutation(members)
    return order


def _fold(X, y, people, strata, comparison, index, fold, repeats):
    name, train, test = fold
    forest = fit(X[train], y[train], people[train], comparison.weight)
    Xt, truth, cells = X[test], y[test], strata[test]
    base = balanced_accuracy_score(truth, forest.predict(Xt))
    rows = []
    for column, feature in enumerate(ARM_FEATURES):
        drops = []
        for repeat in range(repeats):
            order = stratified_order(cells, np.random.default_rng([SEED, index, column, repeat]))
            shuffled = Xt.copy()
            shuffled[:, column] = Xt[order, column]
            drops.append(base - balanced_accuracy_score(truth, forest.predict(shuffled)))
        rows.append({"comparison": comparison.key, "fold": name, "feature": feature,
                     "gini": float(forest.feature_importances_[column]), "held_out_drop": float(np.mean(drops))})
    summary = {"comparison": comparison.key, "fold": name, "train_rows": len(train), "test_rows": len(test),
               "test_people": len(set(people[test])), "balanced_accuracy": float(base)}
    return rows, summary


def importance(rows: pd.DataFrame, comparison: Comparison, repeats: int = REPEATS, jobs: int = -1):
    """Per-fold Gini and held-out drop, and one line per fold with the unshuffled balanced accuracy."""
    X, y, people = design(rows, comparison)
    strata = strata_of(rows)
    folds = make_folds(X, y, people, comparison.split)
    for name, train, test in folds:
        if set(people[train]) & set(people[test]):
            raise AssertionError(f"Fold {name} has a person on both sides")
    parts = Parallel(n_jobs=jobs)(delayed(_fold)(X, y, people, strata, comparison, i, fold, repeats)
                                  for i, fold in enumerate(folds))
    return (pd.DataFrame([row for rows_, _ in parts for row in rows_]), pd.DataFrame([summary for _, summary in parts]))


def summarize(by_fold: pd.DataFrame) -> pd.DataFrame:
    """Mean over folds and ranks, per comparison."""
    out = []
    for key, group in by_fold.groupby("comparison", sort=False):
        mean = group.groupby("feature")[["gini", "held_out_drop"]].mean().reindex(list(ARM_FEATURES))
        mean["gini_rank"] = mean["gini"].rank(ascending=False, method="min").astype(int)
        mean["held_out_rank"] = mean["held_out_drop"].rank(ascending=False, method="min").astype(int)
        mean["no_effect"] = mean["held_out_drop"] <= 0
        out.append(mean.reset_index().assign(comparison=key))
    columns = ["comparison", "feature", "gini", "gini_rank", "held_out_drop", "held_out_rank", "no_effect"]
    return pd.concat(out, ignore_index=True)[columns]
