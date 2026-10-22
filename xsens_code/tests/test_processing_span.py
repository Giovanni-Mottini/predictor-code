"""Verify recovery boundaries and original-source provenance independently."""
import hashlib
from pathlib import Path

import numpy as np
import openpyxl
import pytest
from scipy.signal import butter, sosfiltfilt

from predictor_import.adapter import make_plan, write_plan
from predictor_import.formatter import format_subject
from predictor_import.inventory import scan
from xsens_code.tests.test_xsens_features import full_subject
from xsens_code.tests.test_xsens_reader import opened, replace_column, rewrite_table
from xsens_code.xsens_features import extract_xsens


@pytest.mark.parametrize('gap', ['leading', 'trailing', 'interior', 'inside_window'])
def test_edge_gaps_recover_only_with_valid_margins_and_preserve_source_coordinates(full_subject, gap):
    recording = opened(full_subject)
    raw = recording.subject.manifest['source_root']
    source = Path(raw) / recording.record['source_path']
    book = openpyxl.load_workbook(source)
    for row, frame in enumerate((1040, 1080, 1120, 1160, 1200), 2):
        book['Markers'].cell(row, 1).value = frame
    book.save(source)
    write_plan(make_plan(scan(Path(raw))), full_subject.parents[1])
    format_subject(full_subject)
    positions = {'leading': [0, 1], 'trailing': [238, 239], 'interior': [5], 'inside_window': [50]}[gap]
    def change(table):
        values = table['Right Shoulder x'].to_pylist()
        for i in positions:
            values[i] = None
        return replace_column(table, 'Right Shoulder x', values)
    rewrite_table(full_subject, 'Segment Velocity', change)
    paths = [p for p in full_subject.rglob('*') if p.is_file()]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    recording = opened(full_subject)
    result = extract_xsens(recording).report
    assert [(r['row_start'], r['row_end_exclusive']) for r in result['feature_rows']] == [(40,80),(80,120),(120,160),(160,200)]
    if gap in ('leading', 'trailing'):
        start, end = (2, 240) if gap == 'leading' else (0, 238)
        xyz = np.column_stack([recording.read_table('Segment Velocity')[f'Right Shoulder {a}'].to_pylist() for a in 'xyz']).astype(float)
        expected = np.linalg.norm(sosfiltfilt(butter(4, 6, fs=60, output='sos'), xyz[start:end], axis=0, padlen=12), axis=1)
        for row in result['feature_rows']:
            assert row['vel_mean_shoulder'] == pytest.approx(expected[row['row_start']-start:row['row_end_exclusive']-start].mean(), abs=1e-10)
            evidence = row['feature_evidence']['rom_shoulder_flex_ext']
            assert evidence['min_frame'] == 1000 + evidence['min_row_index']
            assert evidence['max_frame'] == 1000 + evidence['max_row_index']
        assert result['frame_axis']['row_count'] == 240
        assert result['provenance']['finite_span_processing']['source_row_start'] == start
        assert any(f['code'] == 'leading_or_trailing_missing_samples_outside_windows' for f in result['flags'])
    else:
        assert all(r['vel_mean_shoulder'] is None for r in result['feature_rows'])
        assert all(r['vel_mean_trunk'] is not None for r in result['feature_rows'])
        assert not any(f['code'] == 'leading_or_trailing_missing_samples_outside_windows' for f in result['flags'])
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == expected for p, expected in before.items())
