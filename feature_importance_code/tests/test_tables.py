import pandas as pd
import pytest

from feature_importance_code.tables import ARM_FEATURES, LEVERS, feature_files, labels, paretic_sides, read_tables, task_rows
from feature_importance_code.tests.synthetic import task_table, whole_rows, write_run


def test_arm_features_follow_xsens_code():
    xsens = pytest.importorskip("xsens_code.xsens_features")
    assert ARM_FEATURES == tuple(f for f in xsens.FEATURE_NAMES if f != "dtw_hand")
    assert set(LEVERS.values()) == set(xsens.LEVER_FIELDS)


def test_run_folder_is_found(tmp_path):
    write_run(tmp_path, task_table(["FG91", "FG92"]))
    assert [f.parts[-3] for f in feature_files(tmp_path)] == ["FG91", "FG92"]
    with pytest.raises(ValueError, match="no features.csv"):
        feature_files(tmp_path / "runs")


def test_length_divides_by_the_lever(tmp_path):
    frame = task_table(["FG91"], levers=True)
    frame.to_csv(tmp_path / "features.csv", index=False)
    table = read_tables([tmp_path], "length")
    for speed, lever in LEVERS.items():
        assert table[speed].to_numpy() == pytest.approx(frame[speed].to_numpy() / frame[lever].to_numpy())
    assert table["sparc_forearm"].tolist() == pytest.approx(frame["sparc_forearm"].tolist())


def test_length_refuses_tables_without_levers(tmp_path):
    task_table(["FG91"]).to_csv(tmp_path / "features.csv", index=False)
    with pytest.raises(ValueError, match="already-normalized"):
        read_tables([tmp_path], "length")
    stored = read_tables([tmp_path], "already-normalized")
    assert stored["vel_mean_forearm"].tolist() == pytest.approx(task_table(["FG91"])["vel_mean_forearm"].tolist())


def test_whole_recordings_are_kept_apart(tmp_path):
    frame = pd.concat([task_table(["FG91"], whole=True), whole_rows(["FG91"])], ignore_index=True)
    frame.to_csv(tmp_path / "features.csv", index=False)
    table = read_tables([tmp_path], "mps")
    assert table["whole"].sum() == 4 and set(table.loc[table["whole"], "activity"]) == {"mot1"}
    tasks = task_rows(table)
    assert len(tasks) == 64 and tasks["task"].dtype.kind == "i"


def test_labels_name_divided_speeds():
    assert labels("length")["vel_mean_forearm"] == "Normalized forearm speed"
    assert labels("already-normalized")["vel_mean_upper_arm"] == "Normalized upper-arm speed"
    assert labels("mps")["vel_mean_forearm"] == "Forearm speed"
    assert labels("length")["rom_elbow_bending"] == "Elbow range"


def test_paretic_side_comes_from_the_tables(tmp_path):
    task_table(["FG91", "FG92"]).to_csv(tmp_path / "features.csv", index=False)
    assert paretic_sides(read_tables([tmp_path], "mps")) == {"FG91": "right", "FG92": "right"}
