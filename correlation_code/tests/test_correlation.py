"""correlation_code on synthetic data. Made-up codes (FG91-FG95) and dates (2030) only."""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest

from correlation_code.__main__ import main
from correlation_code.robot import ROWS, export_files, read_export, read_sessions, robot_visits
from correlation_code.scales import plain_number, read_scales, sided, workbook_of, workbook_visit
from correlation_code.sources import pair, pooled_whole_recording_visits, task_visits, visit_dates, whole_recording_visits
from correlation_code.statistics import benjamini_hochberg, between, correlate, holm, rmcorr, shuffle_within
from feature_importance_code.tables import ARM_FEATURES, read_tables
from feature_importance_code.tests.synthetic import task_table, whole_rows, write_run

PEOPLE = ["FG91", "FG92", "FG93", "FG94", "FG95"]


# --- statistics ---------------------------------------------------------------------------------------------

def test_holm_and_benjamini_hochberg_match_hand_computed_values():
    p = [0.01, 0.04, 0.03, 0.005]
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])
    assert holm([0.6, 0.7]).max() == 1.0


@pytest.mark.filterwarnings("ignore:divide by zero encountered in arctanh")   # r = 1 exactly: CI is (1, 1)
def test_rmcorr_uses_within_patient_change_only():
    g = pd.Series(np.repeat(["a", "b", "c"], 4))
    x = pd.Series(np.tile([1.0, 2.0, 3.0, 4.0], 3))
    offsets = g.map({"a": 30.0, "b": 0.0, "c": -30.0})          # between-patient trend opposite to the within one
    result = rmcorr(x, 2 * x + offsets, g)
    assert result["r"] == pytest.approx(1.0) and result["df"] == 12 - 3 - 1
    rng = np.random.default_rng(0)
    y = 2 * x + offsets + pd.Series(rng.normal(0, 1, 12))
    xc, yc = x - x.groupby(g).transform("mean"), y - y.groupby(g).transform("mean")
    assert rmcorr(x, y, g)["r"] == pytest.approx(np.corrcoef(xc, yc)[0, 1])


def test_rmcorr_drops_patients_with_one_visit_and_between_needs_four_patients():
    g = pd.Series(["a", "a", "b", "b", "c"])
    x = pd.Series([1.0, 2.0, 1.0, 3.0, 9.0])
    assert rmcorr(x, x, g)["n"] == 4 and rmcorr(x, x, g)["k"] == 2
    assert np.isnan(between(x, x, g)["r"])
    g4 = pd.Series(["a", "b", "c", "d"])
    assert between(pd.Series([1.0, 2, 3, 4]), pd.Series([2.0, 4, 5, 9]), g4)["r"] == pytest.approx(1.0)


def test_shuffles_stay_within_each_patient():
    g = np.array(list("aaabbbbcc"))
    index = shuffle_within(g, np.random.default_rng(1))
    assert sorted(index) == list(range(len(g))) and (g[index] == g).all()


def test_correlate_is_reproducible_and_independent_of_jobs():
    rng = np.random.default_rng(2)
    visits = pd.DataFrame({"patient": np.repeat(PEOPLE, 3), "FMA": rng.normal(30, 5, 15), "BBT": rng.normal(20, 5, 15)})
    visits["f1"] = visits["FMA"] + rng.normal(0, 1, 15)
    visits["f2"] = rng.normal(0, 1, 15)
    one = correlate(visits, ["f1", "f2"], ["FMA", "BBT"], seed=7, permutations=99, jobs=1)
    two = correlate(visits, ["f1", "f2"], ["FMA", "BBT"], seed=7, permutations=99, jobs=2)
    pd.testing.assert_frame_equal(one, two)
    assert list(zip(one["scale"], one["feature"])) == [("FMA", "f1"), ("FMA", "f2"), ("BBT", "f1"), ("BBT", "f2")]
    assert one.loc[0, "rmcorr"] > 0.8 and one.loc[0, "rmcorr_p_shuffle"] <= 0.02
    assert (one["rmcorr_p_holm"] >= one["rmcorr_p_shuffle"]).all()


# --- scales -------------------------------------------------------------------------------------------------

def clinical_workbook(path: Path, *, fma: dict, bbt: dict) -> Path:
    """Header cells, then the visit-label row, the date row and the scale rows, like the clinical workbooks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["T0", "T1_d1", "T1_d2", "T2", "T3", "T4", "T5", "T6"]
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["Codice", "XX"])
    sheet.append(["Visita", *labels])
    sheet.append(["Data", *[datetime(2030, 1, 1 + i) for i in range(len(labels))]])
    sheet.append(["Fugl-Meyer A-D", *[fma.get(label) for label in labels]])
    sheet.append(["Fugl-Meyer totale", *[999 for _ in labels]])                  # only the first Fugl row counts
    sheet.append(["Box and Block Test", *[bbt.get(label) for label in labels]])
    book.save(path)
    return path


def test_visit_labels_numbers_and_sided_cells():
    assert [workbook_visit(c) for c in ("T0", "T1", "T4", "T5", "T12", "T1_d1")] == ["T1", None, None, "T5", "T12", None]
    assert [plain_number(v) for v in (45, "45,5", " 34/100", "x", "na", "abc", None)] == [45.0, 45.5, 34.0, None, None, None, None]
    assert sided("0 (L) / 24 (R)") == {"L": 0.0, "R": 24.0}
    assert sided("58'' (R), 46'' (L)") == {"R": 58.0, "L": 46.0}
    assert sided("0(L)/") == {"L": 0.0}
    assert sided("? (L) / 12 (R)") == {"L": None, "R": 12.0}
    assert sided("24") == {}


def test_scales_pair_t0_with_t1_take_the_paretic_hand_and_fall_back_to_t1_d1(tmp_path):
    clinical_workbook(tmp_path / "FG91/FG91.xlsx", fma={"T0": 20, "T5": "31", "T6": "x"},
                      bbt={"T0": "7", "T1_d1": "3 (L) / 7 (R)", "T5": "5 (L) / 12 (R)", "T6": "6 (L) / 15 (R)"})
    clinical_workbook(tmp_path / "FG92/FG92.xlsx", fma={"T5": 40}, bbt={"T5": "30 (L) / 2 (R)"})
    scales, notes = read_scales(tmp_path, {"FG91": "right", "FG92": "left", "FG93": "left"})
    assert notes == ["FG93: no workbook folder, left out"]
    rows = {(r.patient, r.visit): r for r in scales.itertuples()}
    assert set(rows) == {("FG91", "T1"), ("FG91", "T5"), ("FG91", "T6"), ("FG92", "T5")}
    assert (rows["FG91", "T1"].FMA, rows["FG91", "T1"].BBT) == (20.0, 7.0)
    assert rows["FG91", "T1"].flags == "BBT from T1_d1 (T0 cell has no side)"
    assert rows["FG91", "T1"].workbook_date == "2030-01-01"
    assert (rows["FG91", "T5"].FMA, rows["FG91", "T5"].BBT) == (31.0, 12.0)
    assert np.isnan(rows["FG91", "T6"].FMA) and rows["FG91", "T6"].BBT == 15.0
    assert rows["FG92", "T5"].BBT == 30.0                                         # left hand is paretic


def test_one_workbook_per_patient_folder(tmp_path):
    clinical_workbook(tmp_path / "FG91/a.xlsx", fma={}, bbt={})
    clinical_workbook(tmp_path / "FG91/b.xlsx", fma={}, bbt={})
    with pytest.raises(ValueError, match="expected one clinical workbook"):
        workbook_of(tmp_path / "FG91")


# --- visits from the feature tables -------------------------------------------------------------------------

def test_task_visits_z_score_each_task_and_need_two_repetitions_and_two_tasks(tmp_path):
    frame = task_table(PEOPLE, visits=("T1", "T5", "T6"))
    keep = ~((frame["participant_key"] == "PAZIENTI/FG91") & (frame["timepoint"] == "T5") & (frame["task"] >= 2)
             & (frame["repetition"] >= 2) & (frame["limb_condition"] == "paretic"))
    table = read_tables([write_run(tmp_path, frame[keep])], "mps")
    visits = task_visits(table).set_index(["patient", "visit"])
    assert visits.loc[("FG91", "T5"), "n_tasks"] == 1 and visits.loc[("FG91", "T5"), list(ARM_FEATURES)].isna().all()
    assert visits.loc[("FG92", "T5"), "n_tasks"] == 4
    full = visits[visits["n_tasks"] == 4]
    assert len(full) == len(PEOPLE) * 3 - 1
    assert abs(visits[list(ARM_FEATURES)].mean()).max() < 0.2                   # z-scores centre each task on 0


def test_whole_recordings_paretic_only_and_xsens_dates(tmp_path):
    frame = pd.concat([task_table(PEOPLE[:2], visits=("T1",), whole=True), whole_rows(PEOPLE[:2], visits=("T1",))])
    table = read_tables([write_run(tmp_path, frame)], "mps")
    visits = whole_recording_visits(table, "mot1")
    assert list(visits["patient"]) == ["FG91", "FG92"] and visits["visit"].eq("T1").all()
    dates = visit_dates(table).set_index("patient")["xsens_date"]
    assert dates.to_dict() == {"FG91": "2030-01-10", "FG92": "2030-01-11"}
    doubled = pd.concat([table, table[table["whole"] & (table["condition"] == "paretic")].head(1)])
    with pytest.raises(ValueError, match="More than one paretic mot1"):
        whole_recording_visits(doubled, "mot1")
    scales = pd.DataFrame({"patient": ["FG92", "FG91", "FG93"], "visit": ["T1", "T1", "T1"], "FMA": [1.0, 2.0, 3.0]})
    assert list(pair(scales, visits)["patient"]) == ["FG92", "FG91"]           # scale order, inner join


def test_pooled_exercises_z_score_each_exercise_and_need_both(tmp_path):
    frame = pd.concat([whole_rows(PEOPLE, "mot1", visits=("T1", "T5")), whole_rows(PEOPLE, "mot2", visits=("T1", "T5"), seed=3)])
    keep = ~((frame["activity"] == "mot2") & (frame["participant_key"] == "PAZIENTI/FG91") & (frame["timepoint"] == "T5"))
    table = read_tables([write_run(tmp_path, frame[keep])], "mps")
    pooled = pooled_whole_recording_visits(table).set_index(["patient", "visit"])
    assert len(pooled) == len(PEOPLE) * 2 and pooled.loc[("FG91", "T5"), "n_exercises"] == 1
    assert pooled.loc[("FG91", "T5"), list(ARM_FEATURES)].isna().all()
    f = "sparc_forearm"
    one, two = (whole_recording_visits(table, a).set_index(["patient", "visit"])[f] for a in ("mot1", "mot2"))
    z = lambda s: (s - s.mean()) / s.std()
    assert pooled.loc[("FG93", "T1"), f] == pytest.approx((z(one)[("FG93", "T1")] + z(two)[("FG93", "T1")]) / 2)


# --- robot exports ------------------------------------------------------------------------------------------

def export_csv(folder: Path, kind: str, day: str, time="10-30-00-000", overall=50.0, settings=(2, 4, 2)) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    values = {name: str(overall + 0.1 * i) for i, name in enumerate(ROWS)}
    values |= {"scores.params": f"{overall / 200},{overall / 100}", "scores.overall": str(overall),
               "weight": str(settings[0]), "viscosity": str(settings[1]), "stiffness": str(settings[2])}
    path = folder / f"csv_data_{kind}_{day}_{time}Z.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
        writer.writerow(["Name", "Value"])
        writer.writerows([name, values[name]] for name in ROWS)
    return path


def test_read_export_splits_params_and_reads_settings(tmp_path):
    record = read_export(export_csv(tmp_path, "TraIns", "2030-01-10", overall=50.0, settings=(5, 4, 16)))
    assert (record["kind"], record["day"], record["start_utc"]) == ("TraIns", "2030-01-10", "2030-01-10 10:30:00.000")
    assert (record["params_1"], record["params_2"], record["overall"]) == (0.25, 0.5, 50.0)
    assert (record["weight"], record["viscosity"], record["stiffness"]) == (5.0, 4.0, 16.0)
    bad = tmp_path / "csv_data_Evaluation_2030-01-10_10-30-00-000Z.csv"
    bad.write_text("Name,Value\nscores.overall,1\n")
    with pytest.raises(ValueError, match="unexpected rows"):
        read_export(bad)


def test_sessions_from_both_layouts_copies_count_once_and_conflicts_stop(tmp_path):
    patients = tmp_path / "PAZIENTI"
    export_csv(patients / "FG91/MOTORE/fg91_export/T1", "Evaluation", "2030-01-10")
    export_csv(patients / "FG91/MOTORE/fg91_export/T2", "Evaluation", "2030-01-10")     # same session, two folders
    single = tmp_path / "rebuilt/fg92_export"
    export_csv(single / "T1", "Evaluation", "2030-01-11")
    assert {p for p, _ in export_files([patients, single])} == {"FG91", "FG92"}
    sessions = read_sessions([patients, single])
    assert len(sessions) == 2 and sessions.set_index("patient").loc["FG91", "copies"] == 2
    export_csv(patients / "FG91/MOTORE/fg91_export/T3", "Evaluation", "2030-01-10", overall=99.0)
    with pytest.raises(ValueError, match="two different export files"):
        read_sessions([patients])


def test_robot_visits_pair_by_xsens_day(tmp_path):
    folder = tmp_path / "fg91_export"
    export_csv(folder, "Evaluation", "2030-01-10", time="09-00-00-000", overall=10.0)
    export_csv(folder, "Evaluation", "2030-01-10", time="15-00-00-000", overall=20.0)
    export_csv(folder, "TraIns", "2030-01-10")
    sessions = read_sessions([folder])
    dates = pd.DataFrame({"patient": ["FG91", "FG91"], "visit": ["T1", "T5"], "xsens_date": ["2030-01-10", "2030-02-10"]})
    visits, pairing = robot_visits(sessions, dates, "Evaluation")
    assert list(visits["visit"]) == ["T1"] and visits.loc[0, "overall"] == 10.0
    assert list(pairing["status"]) == ["used the first of the day", "no session on the Xsens day"]
    with pytest.raises(ValueError, match="Unknown session type"):
        robot_visits(sessions, dates, "Inseguimento")


# --- the three commands -------------------------------------------------------------------------------------

@pytest.fixture
def study(tmp_path):
    """Five patients, visits T1/T5/T6 with tasks, mot1 and mot2, workbooks and robot exports on the Xsens days."""
    visits = ("T1", "T5", "T6")
    frame = pd.concat([task_table(PEOPLE, visits=visits, levers=True, whole=True),
                       whole_rows(PEOPLE, "mot1", visits=visits, levers=True),
                       whole_rows(PEOPLE, "mot2", visits=visits, levers=True, seed=3)])
    run = write_run(tmp_path / "run", frame)
    workbooks = tmp_path / "PAZIENTI"
    for i, person in enumerate(PEOPLE):
        clinical_workbook(workbooks / person / f"{person}.xlsx", fma={"T0": 20 + i, "T5": 25 + 2 * i, "T6": 30 + i},
                          bbt={"T0": f"1 (L) / {5 + i} (R)", "T5": f"1 (L) / {8 + i} (R)", "T6": f"1 (L) / {9 + 2 * i} (R)"})
        for v, visit in enumerate(visits):
            export_csv(workbooks / person / "MOTORE" / f"{person.lower()}_export" / visit, "Evaluation",
                       f"2030-0{1 + v}-1{i}", overall=40.0 + i + v)
    return run, workbooks, tmp_path


def test_three_studies_write_their_tables_and_refuse_a_used_folder(study, capsys):
    run, workbooks, tmp = study
    common = ["--features", str(run), "--workbooks", str(workbooks), "--permutations", "49", "--jobs", "1"]
    assert main(["tasks", *common, "--speed", "length", "--output", str(tmp / "out/tasks")]) == 0
    assert main(["xsens-motore", *common, "--speed", "length", "--output", str(tmp / "out/motore")]) == 0
    assert main(["robot", *common, "--exports", str(workbooks), "--output", str(tmp / "out/robot")]) == 0
    tasks = pd.read_csv(tmp / "out/tasks/tasks_results.csv")
    assert len(tasks) == 2 * len(ARM_FEATURES) and tasks["visits"].max() == 15 and tasks["patients"].max() == 5
    assert "Normalized forearm speed" in set(tasks["label"])
    for name in ("mot1", "mot2", "mot_pooled"):
        assert len(pd.read_csv(tmp / f"out/motore/{name}_visits.csv")) == 15
    pairing = pd.read_csv(tmp / "out/robot/robot_evaluation_pairing.csv")
    assert pairing["status"].eq("used").sum() == 15
    assert pd.read_csv(tmp / "out/robot/robot_trains_pairing.csv")["status"].eq("no session on the Xsens day").all()
    info = json.loads((tmp / "out/tasks/run.json").read_text())
    assert info["permutations"] == 49 and len(info["inputs"]["workbooks"]) == 5
    assert (tmp / "out/tasks/tasks_table.tex").read_text().startswith("\\begin{table")
    assert main(["tasks", *common, "--speed", "length", "--output", str(tmp / "out/tasks")]) == 2
    assert "not empty" in capsys.readouterr().err


def test_motore_study_needs_whole_recordings(study, capsys):
    run, workbooks, tmp = study
    frame = pd.concat(pd.read_csv(f) for f in run.glob("runs/*/*/xsens/features.csv"))
    tasks_only = write_run(tmp / "tasks_only", frame[frame["window_kind"] != "whole_recording"].drop(columns=["window_kind", "activity"]))
    code = main(["xsens-motore", "--features", str(tasks_only), "--workbooks", str(workbooks), "--speed", "length",
                 "--permutations", "9", "--jobs", "1", "--output", str(tmp / "out/m")])
    assert code == 2 and "no mot1/mot2 rows" in capsys.readouterr().err
