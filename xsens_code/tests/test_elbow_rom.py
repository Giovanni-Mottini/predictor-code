"""Independent geometric and source-dependency checks for production elbow ROM."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from shared_code.reader import DataContractError
from xsens_code.elbow_rom import bending_values, ELBOW_FEATURE
from xsens_code.xsens_features import extract_xsens, FEATURE_NAMES
from xsens_code.tests.test_xsens_features import full_subject
from xsens_code.tests.test_xsens_reader import opened, rewrite_table, replace_column


def test_bending_is_invariant_to_global_rotation_pronation_quaternion_sign_and_scale():
    angles = np.array([0., 30., 60., 90., 120.])
    upper = Rotation.from_euler('xyz', [[20, 30, 40]] * len(angles), degrees=True)
    forearm = upper * Rotation.from_euler('z', angles[:, None], degrees=True) * Rotation.from_euler('y', angles[:, None] / 2, degrees=True)
    u, f = upper.as_quat()[:, [3, 0, 1, 2]], forearm.as_quat()[:, [3, 0, 1, 2]]
    u[2] *= -.9
    original = u.copy()
    values, _, findings, _ = bending_values(u, f, list(range(len(angles))))
    np.testing.assert_allclose(values, angles, atol=1e-12)
    np.testing.assert_array_equal(u, original)
    assert findings[0]['events'][0]['frame'] == 2


@pytest.mark.parametrize('bad', [[0, 0, 0, 0], [np.nan, 0, 0, 0], [np.inf, 0, 0, 0]])
def test_invalid_quaternions_are_rejected(bad):
    q = np.tile([1., 0, 0, 0], (3, 1))
    broken = q.copy()
    broken[1] = bad
    with pytest.raises(DataContractError):
        bending_values(broken, q, [0, 1, 2])


def test_euler_elbow_is_not_a_processing_dependency(full_subject):
    baseline = extract_xsens(opened(full_subject)).report
    assert ELBOW_FEATURE in FEATURE_NAMES and 'rom_elbow_flex_ext' not in FEATURE_NAMES
    recording = opened(full_subject)
    descriptor = recording.table_descriptor('Joint Angles ZXY')
    descriptor['columns'] = [c for c in descriptor['columns'] if c['name'] not in {'Right Elbow Flexion/Extension', 'Left Elbow Flexion/Extension'}]
    from shared_code.io import atomic_json
    atomic_json(recording.descriptor_path, recording.descriptor)
    rewrite_table(full_subject, 'Joint Angles ZXY', lambda table: table.drop([
        'Right Elbow Flexion/Extension', 'Left Elbow Flexion/Extension']))
    result = extract_xsens(opened(full_subject)).report
    for before, after in zip(baseline['feature_rows'], result['feature_rows']):
        for feature in FEATURE_NAMES:
            assert before[feature] == after[feature]
    assert result['channels'][ELBOW_FEATURE]['status'] == 'computed'


def test_zero_arm_quaternion_excludes_elbow_only(full_subject):
    baseline = extract_xsens(opened(full_subject)).report
    def damage(table):
        for i in range(4):
            values = table[f'Right Upper Arm q{i}'].to_pylist()
            values[80] = 0.
            table = replace_column(table, f'Right Upper Arm q{i}', values)
        return table
    rewrite_table(full_subject, 'Segment Orientation - Quat', damage)
    result = extract_xsens(opened(full_subject)).report
    assert result['channels'][ELBOW_FEATURE]['status'] == 'excluded'
    for before, after in zip(baseline['feature_rows'], result['feature_rows']):
        assert after[ELBOW_FEATURE] is None
        for feature in FEATURE_NAMES:
            if feature != ELBOW_FEATURE:
                assert before[feature] == after[feature]
