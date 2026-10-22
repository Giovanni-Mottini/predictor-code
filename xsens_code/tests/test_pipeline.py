"""Synthetic Dati PREDICTOR-to-XLSX checks, with no private dataset dependency."""
import hashlib
import json
from pathlib import Path
import shutil

import openpyxl
import pytest

from predictor_import.tests.test_import import workbook
from shared_code.reader import DataContractError
from xsens_code.__main__ import main
from xsens_code.export import HEADERS, export_workbooks
from xsens_code.xsens_features import METHOD_ID
from xsens_code.tests.test_xsens_features import full_subject


@pytest.fixture
def patient_raw(full_subject, tmp_path):
    manifest = json.loads((full_subject / 'manifest.json').read_text())
    original = Path(manifest['source_root']) / 'SANI/SANO_FG91/XSens'
    raw = tmp_path / 'Dati PREDICTOR'
    destination = raw / 'PAZIENTI/FG91/XSens/T1'
    destination.mkdir(parents=True)
    for side, code in [('dx', '00'), ('sx', '01')]:
        shutil.copy2(original / f'FG91_t1_task1_{side}.xlsx', destination / f'FG91_{code}_TASK1_T1.xlsx')
    workbook(raw / 'PAZIENTI/FG91/FG91_VAL.xlsx', metadata_id='FG91', hemisphere='Right')
    return raw


def test_raw_to_workbooks_and_cached_rerun_preserve_sources(patient_raw, tmp_path):
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in patient_raw.rglob('*.xlsx')}
    prepared = tmp_path / 'prepared'
    def invocation(output, run):
        return ['pipeline', str(patient_raw), '--subject', 'PAZIENTI/FG91', '--standardized', str(prepared),
                '--output', str(output), '--work-directory', str(run), '--no-progress']
    output, run = tmp_path / 'OUTPUT', tmp_path / 'run'
    assert main(invocation(output, run)) == 0
    path = output / 'PAZIENTI/FG91/Xsens_tasks/Xsens_tasks.xlsx'
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    assert book.sheetnames == ['T1']
    rows = list(book['T1'].values)
    book.close()
    assert tuple(rows[0]) == HEADERS and len(rows) == 9
    assert {row[3] for row in rows[1:]} == {'dx', 'sx'}
    assert {row[4] for row in rows[1:]} == {'healthy', 'paretic'}
    assert all(row[HEADERS.index('rom_elbow_bending')] is not None for row in rows[1:])
    assert json.loads((run / 'run.json').read_text())['method_id'] == METHOD_ID
    assert not list(run.rglob('*.md')) and not list(run.rglob('report.json'))
    with pytest.raises(SystemExit) as error:
        main(invocation(output, tmp_path / 'refused-run'))
    assert error.value.code == 2 and not (tmp_path / 'refused-run').exists()
    assert main(invocation(tmp_path / 'OUTPUT-2', tmp_path / 'run-2')) == 0
    preparation = json.loads((tmp_path / 'run-2/preparation/batch_report.json').read_text())
    assert preparation['summary']['conversion_statuses'] == {'cached': 2}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    feature_directory = run / 'runs/PAZIENTI/FG91/xsens'
    with pytest.raises(DataContractError, match='already exists'):
        export_workbooks([feature_directory], output)
    (feature_directory / 'features.csv').write_text('modified')
    with pytest.raises(DataContractError, match='checksum'):
        export_workbooks([feature_directory], tmp_path / 'tampered-output')
    assert not (tmp_path / 'tampered-output').exists()
