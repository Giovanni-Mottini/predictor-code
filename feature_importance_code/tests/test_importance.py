import numpy as np
import pandas as pd
import pytest

from feature_importance_code.importance import (COMPARISONS, comparison_rows, design, importance, make_folds,
                                                stratified_order, summarize)
from feature_importance_code.outputs import rank_table
from feature_importance_code.tables import ARM_FEATURES, labels, read_tables
from feature_importance_code.tests.synthetic import task_table

PATIENTS = ["FG91", "FG92", "FG93", "FG94", "FG95", "FG96"]


def _table(tmp_path, frame):
    frame.to_csv(tmp_path / "features.csv", index=False)
    return read_tables([tmp_path], "mps")


def test_the_informative_feature_ranks_first(tmp_path):
    table = _table(tmp_path, task_table(PATIENTS, signal="rom_elbow_bending", effect=8.0))
    comparison = COMPARISONS["paretic_vs_non_paretic"]
    by_fold, folds = importance(comparison_rows(table, comparison), comparison, repeats=3, jobs=1)
    summary = summarize(by_fold)
    first = summary.set_index("feature").loc["rom_elbow_bending"]
    assert first["gini_rank"] == 1 and first["held_out_rank"] == 1
    assert len(folds) == len(PATIENTS) and folds["test_people"].eq(1).all()
    noise = summary[summary["feature"] != "rom_elbow_bending"]
    assert noise["held_out_drop"].abs().max() < 0.05


def test_volunteers_compare_healthy_with_simulated(tmp_path):
    frame = task_table([f"SANO_BA09{i}" for i in range(1, 7)], cohort="SANI", conditions=("non_paretic", "simulated"),
                       visits=("unlabelled",), signal="sparc_forearm", effect=8.0)
    table = _table(tmp_path, frame)
    comparison = COMPARISONS["healthy_vs_simulated"]
    rows = comparison_rows(table, comparison)
    assert set(rows["condition"]) == {"healthy", "simulated"} and rows["person"].str.startswith("BA").all()
    by_fold, folds = importance(rows, comparison, repeats=2, jobs=1)
    assert len(folds) == 5
    assert summarize(by_fold).set_index("feature").loc["sparc_forearm", "gini_rank"] == 1


def test_folds_never_share_a_person():
    people = np.repeat(np.array(PATIENTS), 10)
    X, y = np.zeros((60, 2)), np.tile(["a", "b"], 30)
    for split in ("leave_one_person_out", "groupkfold"):
        for _, train, test in make_folds(X, y, people, split):
            assert not set(people[train]) & set(people[test])


def test_shuffling_stays_inside_each_cell():
    strata = np.array(["a", "a", "b", "b", "b", "c"])
    order = stratified_order(strata, np.random.default_rng(3))
    assert sorted(order) == list(range(6))
    assert (strata[order] == strata).all()


def test_task_is_an_input_only_for_patients(tmp_path):
    table = _table(tmp_path, task_table(PATIENTS[:2]))
    X, y, people = design(comparison_rows(table, COMPARISONS["paretic_vs_non_paretic"]), COMPARISONS["paretic_vs_non_paretic"])
    assert X.shape[1] == len(ARM_FEATURES) + 4 and X[:, -4:].sum(axis=1).tolist() == [1.0] * len(X)


def test_bad_tables_stop_the_run(tmp_path):
    frame = task_table(PATIENTS[:2])
    twice = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        comparison_rows(_table(tmp_path, twice), COMPARISONS["paretic_vs_non_paretic"])
    same_side = frame.assign(anatomical_side="left")
    with pytest.raises(ValueError, match="opposite"):
        comparison_rows(_table(tmp_path, same_side), COMPARISONS["paretic_vs_non_paretic"])


def test_ranks_share_the_lower_number_and_mark_no_effect():
    by_fold = pd.DataFrame({"comparison": "paretic_vs_non_paretic", "fold": "FG91", "feature": list(ARM_FEATURES),
                            "gini": [0.2, 0.2] + [0.05] * 12, "held_out_drop": [0.1, -0.01] + [0.0] * 12})
    summary = summarize(by_fold)
    assert summary["gini_rank"].tolist()[:3] == [1, 1, 3]
    assert summary.set_index("feature").loc["vel_mean_shoulder", "no_effect"]
    tex = rank_table(summary, ["paretic_vs_non_paretic"], labels("length"))
    assert r"Normalized trunk speed & \textbf{1} & \textbf{1} \\" in tex
    assert r"Normalized shoulder speed & \textbf{1} & 14$^{\dagger}$ \\" in tex
