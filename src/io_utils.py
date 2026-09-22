"""Portable inputs, stable observation order, and explicit output boundaries."""
from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import numpy as np
import pandas as pd
from .settings import ALTITUDES, EXPECTED_CLASSES, METHODS, PROFILE_COLS

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIERS = {'FieldID': str, 'ID': str, 'Type': str, 'TreeKey': str, 'SensorToken': str}


def input_path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def output_dir(value):
    """All generated output belongs to this independent release directory."""
    path = input_path(value)
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError('Output must be a subdirectory of this release directory.')
    # Protect source, reference metadata, and pre-existing local documents.
    if path.relative_to(ROOT).parts[0] in {'src', 'data', '_validation', '.git'}:
        raise ValueError('Choose a results directory, not source or reference metadata.')
    path.mkdir(parents=True, exist_ok=True)
    return path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_run_record(out, command, inputs=(), details=None):
    packages = {}
    for package in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'laspy', 'matplotlib', 'torch']:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    record = {'command': command, 'python': platform.python_version(), 'packages': packages,
              'inputs': [{'file': Path(p).name, 'sha256': digest(p)} for p in inputs],
              'settings': details or {}}
    (out / f'{command}_run.json').write_text(json.dumps(record, indent=2), encoding='utf-8')


def lidar_files(folder):
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in {'.las', '.laz'})


def discover(data_root):
    tasks = []
    rows = []
    for alt in ALTITUDES:
        candidates = [data_root / f'{alt:03d}', data_root / str(alt)]
        available = [(p, lidar_files(p)) for p in dict.fromkeys(candidates) if p.is_dir()]
        available = [(p, files) for p, files in available if files]
        if len(available) != 1:
            raise ValueError(f'Expected one {alt}-m folder under {data_root}; found {len(available)}.')
        folder, files = available[0]
        tasks.extend((str(p), alt, 'L1') for p in files)
        rows.append({'Source': 'L1', 'Altitude': alt, 'Folder': folder.name, 'NFiles': len(files)})
    files = lidar_files(data_root / 'A120')
    tasks.extend((str(p), 120, 'H300') for p in files)
    rows.append({'Source': 'H300', 'Altitude': 120, 'Folder': 'A120', 'NFiles': len(files)})
    return tasks, pd.DataFrame(rows)


def canonical_order(df, manifest=None):
    """Reproduce archived training-row order, independent of worker completion."""
    manifest = manifest or ROOT / 'data' / 'observation_order.csv'
    order = pd.read_csv(manifest, dtype=IDENTIFIERS)
    keys = ['Source', 'Altitude', 'TreeKey']
    order['_rank'] = np.arange(len(order))
    result = df.merge(order[keys + ['_rank']], on=keys, how='left', validate='one_to_one')
    return result.sort_values(['_rank'] + keys, na_position='last', kind='stable').drop(columns='_rank').reset_index(drop=True)


def validate_features(df):
    required = {'Source', 'File', 'Class', 'FieldID', 'ID', 'Type', 'Altitude', 'TreeKey'}
    required.update(f for spec in METHODS.values() for f in spec['features'])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'Missing feature-table columns: {missing}')
    if df.empty:
        raise ValueError('The feature table is empty.')
    if df[list(required & {'Source','File','FieldID','ID','Type','TreeKey','Class','Altitude'})].isna().any().any():
        raise ValueError('Missing observation identifiers or labels.')
    if not df['Class'].isin(EXPECTED_CLASSES).all():
        raise ValueError('Expected finalized class labels 0, 1, 2, 3, 4; no implicit remapping is performed.')
    if not df['Altitude'].isin(ALTITUDES).all():
        raise ValueError('Unexpected altitude in feature table.')
    if not df['Source'].isin(['L1','H300']).all():
        raise ValueError('Source must be L1 or H300.')
    if df.duplicated(['Source', 'Altitude', 'TreeKey']).any():
        raise ValueError('Duplicate sensor/altitude/tree observations.')
    expected = df['FieldID'].astype(str) + '|' + df['ID'].astype(str) + '|' + df['Type'].astype(str)
    if not expected.equals(df['TreeKey'].astype(str)):
        raise ValueError('TreeKey does not agree with field, tree ID and type.')
    if (df.groupby('TreeKey')['Class'].nunique() > 1).any():
        raise ValueError('A biological tree has conflicting labels across observations.')
    feature_cols = sorted({f for spec in METHODS.values() for f in spec['features']})
    numeric = df[feature_cols].apply(pd.to_numeric, errors='raise')
    if np.isinf(numeric.to_numpy(float)).any():
        raise ValueError('Feature values contain infinity.')
    # Individual missing cells are retained for training-fold median imputation.
    if numeric.isna().all().any():
        raise ValueError('An entire required feature is missing.')
    return df


def read_features(value):
    df = pd.read_csv(input_path(value), dtype=IDENTIFIERS)
    df['FieldID'] = df['FieldID'].str.strip().str.replace(r'\.0$', '', regex=True)
    return canonical_order(validate_features(df))


def require_fields(df, fields, purpose):
    observed = set(df['FieldID'].astype(str))
    if observed != set(fields):
        raise ValueError(f'{purpose} requires fields {fields}; found {sorted(observed)}.')
    if set(df['Class']) != set(EXPECTED_CLASSES):
        raise ValueError(f'{purpose} requires all five finalized classes.')


def reference_cohort(df):
    expected = pd.read_csv(ROOT / 'data' / 'observation_order.csv', dtype=IDENTIFIERS)
    expected = expected[(expected['Source'] == 'L1') & (expected['Altitude'] == 60)]
    got = df[(df['Source'] == 'L1') & (df['Altitude'] == 60)]
    cols = ['TreeKey', 'Class']
    return expected[cols].sort_values('TreeKey').reset_index(drop=True).equals(
        got[cols].sort_values('TreeKey').reset_index(drop=True))


def profile_fingerprint(df):
    """Bind archived controls to the same identities, labels and profile values."""
    cols = ['TreeKey', 'Class'] + PROFILE_COLS
    if not set(cols).issubset(df.columns):
        raise ValueError('Profile comparison requires the extracted p1...p20 columns.')
    dat = df[(df['Source'] == 'L1') & (df['Altitude'] == 60)]
    text = dat[cols].sort_values('TreeKey').to_csv(index=False, float_format='%.12g', lineterminator='\n')
    return hashlib.sha256(text.encode()).hexdigest()
