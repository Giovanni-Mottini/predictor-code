"""Unsigned elbow bending ROM from normalized upper-arm/forearm body +Y axes."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from shared_code.reader import DataContractError
from .rom import filter_angle, range_evidence
from .xsens import frame_axis

ELBOW_FEATURE = "rom_elbow_bending"
SPEC = {
    'method_id': 'xsens-elbow-longitudinal-axis-bending-normalized-v1',
    'output_label': 'rom_elbow_bending',
    'source_sheet': 'Segment Orientation - Quat', 'quaternion_order': 'wxyz',
    'segments': ['Upper Arm', 'Forearm'], 'axis': 'segment body +Y for both segments',
    'angle': 'atan2(norm(cross(u,f)), dot(u,f)) in degrees',
    'range': 'max minus min after order-4 Butterworth filtfilt at the configured cutoff (default 6 Hz); original repetition windows',
    'normalization': 'Every finite nonzero quaternion divided by its norm in derived arrays only',
    'norm_minimum': 1e-12, 'norm_review_tolerance': 1e-3,
    'norm_review': 'Record frames and original components; retain computed elbow with requires_review for whole channel',
    'geometry_check': 'Compare quaternion +Y with position bone axes and report disagreement; no fitted axes',
    'singularity': 'No ZXY Euler decomposition; unsigned bending remains defined for nonzero segment axes',
    'limitations': ['Unsigned; includes sideways angulation; cannot distinguish hyperextension.',
                    'Quaternion normalization repairs magnitude only, not incorrect orientation.',
                    'Same-model position agreement is not independent physical validation.'],
    'raw_data_modified': False, 'scientifically_validated': False,
    'adoption': 'Production default approved by user on 2026-10-13',
}


def normalized_rotation(values, name, frames):
    q = np.asarray(values, dtype=float)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all():
        raise DataContractError('elbow_quaternion_nonfinite', f'{name}: invalid quaternion array')
    norms = np.linalg.norm(q, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= SPEC['norm_minimum']):
        raise DataContractError('elbow_quaternion_zero_norm', f'{name}: zero/near-zero quaternion norm')
    bad = np.flatnonzero(abs(norms-1) > SPEC['norm_review_tolerance'])
    findings = []
    if len(bad):
        findings.append({'code': 'elbow_quaternion_norm_normalized_requires_review', 'severity': 'warning',
                         'segment': name, 'count': len(bad),
                         'events': [{'frame': int(frames[i]), 'source_excel_row': int(i)+2,
                                     'norm_before': float(norms[i]), 'raw_wxyz': q[i].tolist()} for i in bad],
                         'treatment': 'normalized in derived arrays; all elbow repetitions retained with review'})
    return Rotation.from_quat((q/norms[:, None])[:, [1,2,3,0]]), findings, [float(norms.min()),float(norms.max())]


def bending_values(upper, forearm, frames):
    rotations, findings, ranges = [], [], []
    for q, name in ((upper,'Upper Arm'),(forearm,'Forearm')):
        rotation, flags, bounds = normalized_rotation(q,name,frames)
        rotations.append(rotation); findings.extend(flags); ranges.append(bounds)
    axes = [r.apply(np.tile([0.,1.,0.],(len(frames),1))) for r in rotations]
    cross = np.linalg.norm(np.cross(*axes),axis=1)
    dot = np.einsum('ij,ij->i',*axes)
    return np.rad2deg(np.arctan2(cross,dot)), axes, findings, ranges


def extract_elbow_bending(recording, frames, windows, cutoff_hz, sampling_rate_hz):
    channel = {'source_sheet': SPEC['source_sheet'], 'source_unit': 'dimensionless', 'effective_unit': 'deg',
               'status': 'computed', 'findings': [], 'method_id': SPEC['method_id'],
               'definition': SPEC['output_label']}
    filtered = None
    try:
        quat = recording.read_table(SPEC['source_sheet'])
        if frames != frame_axis(quat,SPEC['source_sheet']):
            raise DataContractError('elbow_frame_mismatch','Angle and quaternion frames differ')
        side = recording.record['anatomical_side']
        arrays = [np.column_stack([quat[f'{side.title()} {s} q{i}'].to_pylist() for i in range(4)])
                  for s in SPEC['segments']]
        values, axes, flags, ranges = bending_values(*arrays,frames)
        channel['findings'].extend(flags)
        channel['quaternion_norm_ranges'] = ranges
        channel['quaternions_normalized'] = True
        # Independent geometric consistency check, advisory because both come from the same model.
        try:
            pos = recording.read_table('Segment Position')
            if frames != frame_axis(pos,'Segment Position'):
                raise ValueError('Position frames differ')
            points = [np.column_stack([pos[f'{side.title()} {s} {axis}'].to_pylist() for axis in 'xyz'])
                      for s in ('Upper Arm','Forearm','Hand')]
            vectors = [points[1]-points[0],points[2]-points[1]]
            norms = [np.linalg.norm(v,axis=1) for v in vectors]
            if any(not np.isfinite(v).all() for v in vectors) or any(np.any(n<=1e-12) for n in norms):
                raise ValueError('Invalid position segment direction')
            vectors = [v/n[:,None] for v,n in zip(vectors,norms)]
            position_angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(*vectors),axis=1),np.einsum('ij,ij->i',*vectors)))
            error = float(np.max(abs(position_angle-values)))
            direction = 1 if side=='left' else -1
            agreement = [float(np.min(direction*np.einsum('ij,ij->i',a,v))) for a,v in zip(axes,vectors)]
            channel['geometry_check'] = {'maximum_angle_error_deg': error, 'minimum_axis_agreement': agreement}
            if error > .2 or min(agreement)<.99:
                channel['findings'].append({'code':'elbow_geometry_disagreement','severity':'warning',
                                            'maximum_angle_error_deg':error,'minimum_axis_agreement':agreement})
        except (DataContractError,ValueError,KeyError,TypeError) as exc:
            channel['findings'].append({'code':'elbow_geometry_check_unavailable','severity':'warning','message':str(exc)})
        if any(f['severity']=='warning' for f in channel['findings']):
            channel['status']='requires_review'
        filtered=filter_angle(values,cutoff_hz,sampling_rate_hz)
        if not np.isfinite(filtered).all():
            raise DataContractError('elbow_filter_nonfinite','Bending filter produced nonfinite values')
    except (DataContractError,OSError,ValueError,KeyError,TypeError) as exc:
        filtered=None
        channel['status']='excluded'
        channel['findings'].append({'code':getattr(exc,'code','elbow_bending_unavailable'),'severity':'error',
                                    'message':str(exc),'treatment':'elbow only excluded; other features retained'})
    rows = []
    for window in windows:
        evidence = range_evidence(filtered, window, frames) if filtered is not None else {'value': None}
        evidence['definition'] = ELBOW_FEATURE
        status = 'requires_review' if filtered is not None and window.n_samples == 1 else channel['status']
        rows.append({'value': evidence['value'], 'status': status, 'evidence': evidence})
    return channel, rows
