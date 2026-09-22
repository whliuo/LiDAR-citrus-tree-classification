#!/usr/bin/env python3
"""Single public entry point for the citrus LiDAR research workflow."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return value


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    descriptions = {
        'extract': 'Extract locked midpoint features and Profile20 bins from LAS/LAZ.',
        'classify': 'Run all nine locked models, repeated CV, LOFO, merge01 and bootstrap.',
        'altitude': 'Evaluate compact and extended models over six flight heights.',
        'sensor': 'Run paired L1/H300 agreement and five transfer strategies.',
        'profiles': 'Recompute full-profile LR/RF controls, optionally the fixed CNN.',
        'tables': 'Combine saved results and explicitly labelled profile controls.',
        'figures': 'Plot saved tables without refitting classifiers.',
        'height': 'Calculate 20-m tree heights and reproduce distribution panels.',
        'density': 'Calculate raw LAS density from horizontal convex-hull footprints.',
        'context': 'Summarize optional, unlinked visual-repeat and qPCR tables.',
        'all': 'Run extraction, classification, altitude, available sensors, tables and figures.',
    }
    for name, help_text in descriptions.items():
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument('--output-dir', default='results', help='Directory inside this release (default: results).')
        if name in {'extract','all'}:
            p.add_argument('--data-dir', default='data/segtree', help='Parent of 020...120 and optional A120 folders.')
        if name in {'classify','altitude','sensor','profiles','tables'}:
            p.add_argument('--features', default='results/features/01_features_l1.csv')
        if name == 'sensor':
            p.add_argument('--h300-features', default='results/features/02_features_h300.csv')
        if name in {'extract','classify','altitude','sensor','profiles','height','all'}:
            p.add_argument('--workers', type=positive, default=min(8, os.cpu_count() or 1))
        if name == 'profiles':
            p.add_argument('--cnn', action='store_true', help='Also retrain the fixed CPU CNN; requires torch.')
        if name in {'tables','all'}:
            p.add_argument('--profile-table', default='data/reference/profile_benchmarks.csv',
                           help='Explicit archived or recomputed profile-control table; use none to omit.')
        if name == 'height':
            p.add_argument('--input-dir', default='data/segtree/020')
        if name == 'density':
            p.add_argument('--input-dir', default='data/rawdata')
            p.add_argument('--chunk-size', type=positive, default=500000)
            p.add_argument('--xy-unit', choices=['auto','meters','international_feet','us_survey_feet'], default='auto')
        if name == 'context':
            p.add_argument('--visual-repeats')
            p.add_argument('--visual-sheet', default='Tree_Level')
            p.add_argument('--visual-cols', default=None)
            p.add_argument('--qpcr')
            p.add_argument('--qpcr-dataset-col', default='field')
            p.add_argument('--qpcr-hlb-col', default='hlb')
            p.add_argument('--qpcr-ct-col', default='qpcr')
    return ap


def main():
    ap = parser()
    args = ap.parse_args()
    # Keep library caches and temporary files in the standalone release.
    cache = ROOT / '.cache'
    cache.mkdir(exist_ok=True)
    for variable, folder in [('MPLCONFIGDIR','matplotlib'), ('TMP','tmp'), ('TEMP','tmp'), ('TORCH_HOME','torch')]:
        path = cache / folder
        path.mkdir(exist_ok=True)
        os.environ[variable] = str(path)
    os.environ['MPLBACKEND'] = 'Agg'
    for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[variable] = '1'
    from src.workflow import run
    try:
        run(args)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        ap.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
