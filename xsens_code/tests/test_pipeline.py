"""Synthetic Dati PREDICTOR-to-deliverables checks, with no private dataset dependency."""
import csv
import hashlib
import json
import statistics

import openpyxl
import pytest

from predictor_import.tests.test_import import workbook
from shared_code.reader import DataContractError
from xsens_code.__main__ import main
from xsens_code.export import AVG_HEADERS, CSV_FIELDS, HEADERS, export_outputs
from xsens_code.xsens_features import FEATURE_NAMES, METHOD_ID
from xsens_code.tests.test_xsens_features import full_subject, synthetic_recording  # noqa: F401  (fixture)

LAYOUT = {'Features_Xsens_Task_1-4/Pazienti/FG91/Xsens_tasks/Xsens_tasks.xlsx',
          'Features_Xsens_Task_1-4/Pazienti_Paretic_AVG/FG91/Xsens_tasks/Xsens_tasks.xlsx',
          'features b&b/fg91_b&b.csv', 'features XSens Motore/fg91.csv'}


def sheets(path):
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return {name: list(book[name].values) for name in book.sheetnames}
    finally:
        book.close()


def csv_rows(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def patient_raw(tmp_path):
    """FG91: T1 complete (4 tasks x 2 arms), B&B and mot1 on both arms, mot2 paretic only; T2 one task only."""
    raw = tmp_path / 'Dati PREDICTOR'
    t1, t2 = raw / 'PAZIENTI/FG91/XSens/T1', raw / 'PAZIENTI/FG91/XSens/T2'
    for code, side in (('00', 'dx'), ('01', 'sx')):
        for task in range(1, 5):
            synthetic_recording(t1 / f'FG91_{code}_TASK{task}_T1.xlsx', side)
        synthetic_recording(t1 / f'FG91_{code}_B&B_T1.xlsx', side, markers=())
        synthetic_recording(t1 / f'FG91_{code}_mot1_T1.xlsx', side, markers=(1100,))  # never checked
        synthetic_recording(t2 / f'FG91_{code}_TASK1_T2.xlsx', side)
    synthetic_recording(t1 / 'FG91_01_mot2_T1.xlsx', 'sx', markers=())
    workbook(raw / 'PAZIENTI/FG91/FG91_VAL.xlsx', metadata_id='FG91', hemisphere='Right')  # left arm paretic
    return raw


def invocation(raw, prepared, output, run):
    return ['pipeline', str(raw), '--subject', 'PAZIENTI/FG91', '--standardized', str(prepared),
            '--output', str(output), '--work-directory', str(run), '--no-progress']


def test_full_run_has_the_features_example_layout_and_reruns_from_cache(patient_raw, tmp_path):
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in patient_raw.rglob('*.xlsx')}
    prepared, output, run = tmp_path / 'prepared', tmp_path / 'OUTPUT', tmp_path / 'run'
    assert main(invocation(patient_raw, prepared, output, run)) == 1  # mot2 has no partner: DTW blank
    assert {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} == LAYOUT

    tasks = sheets(output / 'Features_Xsens_Task_1-4/Pazienti/FG91/Xsens_tasks/Xsens_tasks.xlsx')
    assert list(tasks) == ['T1']  # T2 has one task only: left out
    assert tuple(tasks['T1'][0]) == HEADERS and len(tasks['T1']) == 33
    assert all(v is not None for row in tasks['T1'][1:] for v in row)
    average = sheets(output / 'Features_Xsens_Task_1-4/Pazienti_Paretic_AVG/FG91/Xsens_tasks/Xsens_tasks.xlsx')
    assert tuple(average['T1'][0]) == AVG_HEADERS and len(average['T1']) == 5
    for task, row in zip(range(1, 5), average['T1'][1:]):
        reps = [r for r in tasks['T1'][1:] if r[1] == task and r[4] == 'paretic']
        assert row[:4] == ('FG91', task, 'sx', 'paretic') and len(reps) == 4
        assert list(row[4:]) == pytest.approx([statistics.mean(r[6 + i] for r in reps) for i in range(15)], rel=1e-12)

    bb = csv_rows(output / 'features b&b/fg91_b&b.csv')
    motore = csv_rows(output / 'features XSens Motore/fg91.csv')
    for rows in (bb, motore):
        assert tuple(rows[0]) == CSV_FIELDS
        assert all((r['subject'], r['timepoint'], r['arto'], r['arto_eff'], r['label'], r['n_samples']) ==
                   ('FG91', '1', 'sx', 'sx', 'malato', '240') for r in rows)
    assert [r['task'] for r in bb] == ['B&B'] and [r['task'] for r in motore] == ['1', '2']
    assert bb[0]['dtw_hand'] != '' and motore[0]['dtw_hand'] != '' and motore[1]['dtw_hand'] == ''
    assert all((r['n_degenerate'], r['degenerate_features']) == ('0', 'none') for r in bb + motore)
    with (run / 'runs/PAZIENTI/FG91/xsens/features.csv').open(newline='') as stream:
        source = {(r['activity'], r['limb_condition']): r for r in csv.DictReader(stream) if r['window_kind'] == 'whole_recording'}
    for row, activity in ((bb[0], 'box_and_blocks'), (motore[0], 'mot1'), (motore[1], 'mot2')):
        expected = source[activity, 'paretic']
        assert all(row[f] == expected[f] for f in (*FEATURE_NAMES, 'duration_s'))  # full precision, unchanged

    receipt = json.loads((run / 'run.json').read_text())
    assert receipt['method_id'] == METHOD_ID
    assert {f['kind'] for f in receipt['files']} == {'tasks', 'tasks_avg', 'bb', 'motore'}
    assert receipt['incomplete_task_visits'] == [{'participant_key': 'PAZIENTI/FG91', 'visit': 'T2', 'reasons': ['24 of 32 rows missing']}]
    assert {f['kind'] for f in receipt['speed_length']['files']} == {'tasks', 'tasks_avg', 'bb', 'motore'}
    assert receipt['speed_length']['incomplete_task_visits'] == receipt['incomplete_task_visits']

    with pytest.raises(SystemExit) as error:  # nothing is ever overwritten
        main(invocation(patient_raw, prepared, output, tmp_path / 'refused-run'))
    assert error.value.code == 2 and not (tmp_path / 'refused-run').exists()
    taken = tmp_path / 'OUTPUT-3_SPEED_LENGTH/features b&b/fg91_b&b.csv'  # the second folder is checked too
    taken.parent.mkdir(parents=True)
    taken.write_text('keep')
    with pytest.raises(SystemExit) as error:
        main(invocation(patient_raw, prepared, tmp_path / 'OUTPUT-3', tmp_path / 'refused-run-3'))
    assert error.value.code == 2 and not (tmp_path / 'OUTPUT-3').exists() and taken.read_text() == 'keep'
    assert main(invocation(patient_raw, prepared, tmp_path / 'OUTPUT-2', tmp_path / 'run-2')) == 1
    preparation = json.loads((tmp_path / 'run-2/preparation/batch_report.json').read_text())
    assert preparation['summary']['conversion_statuses'] == {'cached': 15}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    feature_directory = run / 'runs/PAZIENTI/FG91/xsens'
    with pytest.raises(DataContractError, match='already exists'):
        export_outputs([feature_directory], output)
    (feature_directory / 'features.csv').write_text('modified')
    with pytest.raises(DataContractError, match='checksum'):
        export_outputs([feature_directory], tmp_path / 'tampered-output')
    assert not (tmp_path / 'tampered-output').exists()


def test_speed_length_folder_divides_only_the_four_speeds_by_the_recording_lever(patient_raw, tmp_path):
    output, run = tmp_path / 'OUTPUT', tmp_path / 'run'
    main(invocation(patient_raw, tmp_path / 'prepared', output, run))
    length = tmp_path / 'OUTPUT_SPEED_LENGTH'
    assert {str(p.relative_to(length)) for p in length.rglob('*') if p.is_file()} == LAYOUT
    with (run / 'runs/PAZIENTI/FG91/xsens/features.csv').open(newline='') as stream:
        source = {(r['timepoint'], r['activity'], r['repetition'], r['anatomical_side']): r for r in csv.DictReader(stream)}
    arm = {'dx': 'right', 'sx': 'left'}

    def same_except_speeds(ms_values, length_values, row):
        for name, ms, value in zip(FEATURE_NAMES, ms_values, length_values):
            if name.startswith('vel_mean_'):
                lever = float(row[f"lever_{name.removeprefix('vel_mean_')}_m"])
                assert lever > 0 and value == pytest.approx(float(ms) / lever, rel=1e-15)
            else:
                assert value == ms

    task_path = 'Features_Xsens_Task_1-4/Pazienti/FG91/Xsens_tasks/Xsens_tasks.xlsx'
    ms_tasks, length_tasks = sheets(output / task_path), sheets(length / task_path)
    assert list(length_tasks) == ['T1'] and length_tasks['T1'][0] == ms_tasks['T1'][0]
    for ms_row, length_row in zip(ms_tasks['T1'][1:], length_tasks['T1'][1:]):
        assert length_row[:6] == ms_row[:6]
        same_except_speeds(ms_row[6:], length_row[6:], source['T1', f'task{ms_row[1]}', str(ms_row[2]), arm[ms_row[3]]])
    average = sheets(length / task_path.replace('/Pazienti/', '/Pazienti_Paretic_AVG/'))
    for task, row in zip(range(1, 5), average['T1'][1:]):
        reps = [r for r in length_tasks['T1'][1:] if r[1] == task and r[4] == 'paretic']
        assert list(row[4:]) == pytest.approx([statistics.mean(r[6 + i] for r in reps) for i in range(15)], rel=1e-12)
    for name in ('features b&b/fg91_b&b.csv', 'features XSens Motore/fg91.csv'):
        ms_rows, length_rows = csv_rows(output / name), csv_rows(length / name)
        assert len(length_rows) == len(ms_rows) and list(length_rows[0]) == list(CSV_FIELDS)
        for ms_row, length_row in zip(ms_rows, length_rows):
            assert all(length_row[f] == ms_row[f] for f in CSV_FIELDS if f not in FEATURE_NAMES)
            activity = {'B&B': 'box_and_blocks', '1': 'mot1', '2': 'mot2'}[ms_row['task']]
            same_except_speeds([ms_row[f] for f in FEATURE_NAMES],
                               [float(length_row[f]) if f.startswith('vel_mean_') else length_row[f] for f in FEATURE_NAMES],
                               source['T1', activity, '', arm[ms_row['arto_eff']]])


def test_side_override_shows_workbook_arm_in_arto_and_used_arm_in_arto_eff(patient_raw, tmp_path, monkeypatch):
    from predictor_import import adapter
    monkeypatch.setitem(adapter.SIDE_OVERRIDES, 'PAZIENTI/FG91', {'paretic_side': 'right', 'rule': 'test', 'reason': 'test'})
    output = tmp_path / 'OUTPUT'
    main(invocation(patient_raw, tmp_path / 'prepared', output, tmp_path / 'run'))
    rows = csv_rows(output / 'features b&b/fg91_b&b.csv') + csv_rows(output / 'features XSens Motore/fg91.csv')
    assert {(r['arto'], r['arto_eff']) for r in rows} == {('sx', 'dx')}
    average = sheets(output / 'Features_Xsens_Task_1-4/Pazienti_Paretic_AVG/FG91/Xsens_tasks/Xsens_tasks.xlsx')
    assert {row[2] for row in average['T1'][1:]} == {'dx'}
