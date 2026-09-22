"""CLI orchestration; analyses write standardized tables for later plotting."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import numpy as np
import pandas as pd

from . import analysis
from .io_utils import (ROOT, canonical_order, discover, input_path, output_dir,
                       read_features, reference_cohort, require_fields,
                       validate_features, write_run_record, lidar_files, profile_fingerprint)
from .settings import METHODS, COMMON_FIELDS, ALTITUDES, BOOTSTRAP_REPS, RANDOM_SEED, CV_SPLITS, CV_REPEATS


def save(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def extract(data_root, out, workers):
    from .features import extract_one
    tasks, inventory = discover(data_root)
    dest = out / 'features'
    dest.mkdir(parents=True, exist_ok=True)
    save(inventory, dest / '00_extraction_audit.csv')
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_one, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                row = future.result()
                row['File'] = Path(task[0]).relative_to(data_root).as_posix()
                rows.append(row)
            except Exception as exc:
                errors.append({'File': Path(task[0]).relative_to(data_root).as_posix(), 'Error': str(exc)})
            if i % 500 == 0 or i == len(tasks):
                print(f'Extracted {i}/{len(tasks)} files', flush=True)
    save(pd.DataFrame(errors, columns=['File','Error']), dest / 'extraction_errors.csv')
    if errors:
        raise RuntimeError(f'{len(errors)} extraction failures; see {dest / "extraction_errors.csv"}. No partial feature table published.')
    frame = canonical_order(validate_features(pd.DataFrame(rows)))
    for source, filename in [('L1','01_features_l1.csv'), ('H300','02_features_h300.csv')]:
        part = frame[frame['Source'] == source].copy()
        if not part.empty:
            save(part, dest / filename)
    write_run_record(dest, 'extract', details={'midpoint': 0.5, 'files': len(tasks), 'workers': workers})
    return dest / '01_features_l1.csv', (dest / '02_features_h300.csv' if inventory.iloc[-1]['NFiles'] else None)


def classify(features, out, workers):
    l1 = read_features(features)
    dat = l1[(l1['Source'] == 'L1') & (l1['Altitude'] == 60)].copy()
    require_fields(dat, ['1','2','3','4'], '60-m classification')
    dest = out / 'classification'
    predictions, field_metrics, summaries = [], [], []
    for scheme in ['five_class', 'merge01']:
        print(f'Running leave-one-field-out: {scheme}', flush=True)
        pred, fields, summary = analysis.run_lofo(dat, scheme, METHODS, workers)
        predictions.append(pred); field_metrics.append(fields); summaries.append(summary)
    pred = pd.concat(predictions, ignore_index=True)
    fields = pd.concat(field_metrics, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    save(pred, dest / '01_midpoint_lofo_predictions.csv')
    save(fields, dest / '02_midpoint_lofo_field_metrics.csv')
    print('Running repeated group CV (5 folds x 10 repeats)', flush=True)
    cv, cv_summary = analysis.run_repeated_cv(dat, 'five_class', METHODS, workers)
    save(cv, dest / '03_midpoint_repeated_cv.csv')
    main = summary[summary['Scheme'] == 'five_class'].merge(cv_summary, on=['Scheme','Method'])
    main['CV_minus_LOFO_Accuracy'] = main['CV_Accuracy'] - main['Accuracy']
    main = main.set_index('Method').loc[list(METHODS)].reset_index()
    merge = summary[(summary['Scheme'] == 'merge01') & summary['Method'].isin([
        'VRI_Threshold','GlobalPF_Threshold','MidpointCompact_LR','MidpointExtended_LR','MidpointCompact_RF'])]
    save(main, dest / '04_table4A_main_model_midpoint.csv')
    save(merge, dest / '05_table4B_merge01_midpoint.csv')
    print(f'Running paired bootstrap ({BOOTSTRAP_REPS} resamples per comparison)', flush=True)
    save(analysis.paired_bootstrap(pred, workers), dest / '06_pairwise_bootstrap_midpoint.csv')
    write_run_record(dest, 'classify', [input_path(features)], {'seed': RANDOM_SEED,'cv_splits':CV_SPLITS,'cv_repeats':CV_REPEATS,'bootstrap_reps':BOOTSTRAP_REPS})


def altitude(features, out, workers):
    l1 = read_features(features)
    l1 = l1[l1['Source'] == 'L1']
    for alt in ALTITUDES:
        d = l1[(l1['Altitude'] == alt) & l1['FieldID'].isin(COMMON_FIELDS)]
        require_fields(d, COMMON_FIELDS, f'{alt}-m altitude audit')
    summary, fields = analysis.altitude_audit(l1, workers)
    save(summary, out / 'altitude' / '07_table5_altitude_midpoint.csv')
    save(fields, out / 'altitude' / '08_altitude_field_metrics_midpoint.csv')
    write_run_record(out / 'altitude', 'altitude', [input_path(features)])


def sensor(features, h300_features, out, workers):
    l1, h300 = read_features(features), read_features(h300_features)
    if set(l1['Source']) != {'L1'} or set(h300['Source']) != {'H300'}:
        raise ValueError('Expected separate L1 and H300 feature tables.')
    for frame, name in [(l1,'L1'), (h300,'H300')]:
        require_fields(frame[(frame['Altitude'] == 120) & frame['FieldID'].isin(COMMON_FIELDS)], COMMON_FIELDS, name)
    matched = analysis.match_sensor(l1, h300)
    expected = set(l1[(l1['Altitude'] == 120) & l1['FieldID'].isin(COMMON_FIELDS)]['TreeKey'])
    if set(matched['TreeKey']) != expected:
        raise ValueError('The three-field paired cohort is incomplete or contains inconsistent labels.')
    print(f'Paired L1/H300 trees: {len(matched)}', flush=True)
    dest = out / 'sensor'
    save(matched, dest / 'matched_observations.csv')
    save(analysis.sensor_feature_agreement(matched), dest / '09_sensor_feature_agreement_midpoint.csv')
    save(analysis.sensor_transfer(matched, workers), dest / '10_table6_sensor_transfer_midpoint.csv')
    write_run_record(dest, 'sensor', [input_path(features), input_path(h300_features)])
    (out / 'sensor_status.json').write_text(json.dumps({'completed': True}), encoding='utf-8')


def tables(features, out, profile_table):
    path = out / 'classification' / '04_table4A_main_model_midpoint.csv'
    main = pd.read_csv(path)
    main['ResultSource'] = 'Recomputed midpoint workflow'
    if profile_table and profile_table.lower() != 'none':
        controls = pd.read_csv(input_path(profile_table))
        feature_data = read_features(features)
        if not reference_cohort(feature_data):
            raise ValueError('Profile controls refer to the archived 882-tree cohort; use --profile-table none for another cohort.')
        if 'ResultSource' not in controls:
            raise ValueError('Profile control table must identify ResultSource.')
        if controls['ResultSource'].str.startswith('Archived').any():
            provenance = json.loads((ROOT / 'data/reference/provenance.json').read_text(encoding='utf-8'))
            if profile_fingerprint(feature_data) != provenance['profile_input_fingerprint']:
                raise ValueError('Archived controls do not match these profile values; recompute profiles or use --profile-table none.')
        main = pd.concat([main, controls], ignore_index=True)
    save(main, out / 'tables' / 'main_models.csv')
    pred = pd.read_csv(out / 'classification' / '01_midpoint_lofo_predictions.csv')
    # Useful field-inspection export derived from final, honest LOFO predictions.
    errors = pred[(pred['Scheme'] == 'five_class') & (pred['Method'] == 'MidpointCompact_LR')].copy()
    errors['GradeDifference'] = errors['Pred'] - errors['True']
    save(errors, out / 'tables' / 'per_tree_grade_comparison.csv')
    save(errors[errors['GradeDifference'] != 0], out / 'tables' / 'field_errors.csv')
    write_run_record(out / 'tables', 'tables', [path])


def height(data_root, out, workers):
    from . import heights
    files = lidar_files(data_root)
    if not files:
        raise FileNotFoundError(f'No segmented LAS/LAZ files in {data_root}')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda p: heights.extract_one(p, heights.MIN_TREE_HEIGHT_M), files))
    frame = pd.DataFrame(rows).sort_values(['FieldID','TreeID','FileName']).reset_index(drop=True)
    frame['FilePath'] = [Path(p).relative_to(data_root).as_posix() for p in frame['FilePath']]
    if frame.duplicated(['FieldID','TreeID','Sensor']).any():
        raise ValueError('Duplicate tree-height observations.')
    dest = out / 'height'
    valid = frame[frame['Status'] == 'OK'].copy()
    save(frame, dest / 'tree_height_per_tree_020.csv')
    save(frame[frame['Status'] == 'BELOW_MIN_HEIGHT'], dest / 'tree_height_removed_below_0p5m.csv')
    summary = heights.build_summary(frame, valid)
    save(summary, dest / 'tree_height_summary_020.csv')
    heights.set_nature_style()
    heights.make_figures(valid, dest, heights.DEFAULT_BIN_WIDTH_M)
    write_run_record(dest, 'height', details={'input_files':len(files),'height_percentiles':[1,99],'minimum_height_m':0.5})


def density(data_root, out, chunk_size, xy_unit):
    from . import density as module
    files = lidar_files(data_root)
    if not files:
        raise FileNotFoundError(f'No raw LAS/LAZ files in {data_root}')
    rows = []
    for i, path in enumerate(files, 1):
        print(f'Density {i}/{len(files)}: {path.name}', flush=True)
        rows.append(module.analyze_las(path, data_root, chunk_size, xy_unit))
    frame = pd.DataFrame(rows)
    save(frame, out / 'density' / 'raw_las_point_density_by_file.csv')
    save(module.build_summary(frame), out / 'density' / 'raw_las_point_density_summary.csv')
    write_run_record(out / 'density', 'density', details={'input_files':len(files),'chunk_size':chunk_size,'xy_unit':xy_unit})


def context(args, out):
    from .context import summarize_visual_repeats, summarize_qpcr
    if not (args.visual_repeats or args.qpcr):
        raise ValueError('Supply --visual-repeats and/or --qpcr.')
    dest = out / 'context'
    if args.visual_repeats:
        summary, distribution, detail = summarize_visual_repeats(input_path(args.visual_repeats), args.visual_cols, args.visual_sheet)
        save(summary, dest / 'visual_repeatability_summary.csv')
        save(distribution, dest / 'visual_repeatability_distribution.csv')
        save(detail, dest / 'visual_repeatability_detail.csv')
    if args.qpcr:
        results = summarize_qpcr(input_path(args.qpcr), args.qpcr_dataset_col, args.qpcr_hlb_col, args.qpcr_ct_col)
        for name, frame in zip(['dataset_associations','pooled_association','shared_domain_sensitivity','hlb_level_distribution'], results):
            save(frame, dest / f'qpcr_{name}.csv')


def run(args):
    out = output_dir(args.output_dir)
    name = args.command
    if name == 'all':
        # A new L1-only archive must not plot an earlier sensor result.
        (out / 'sensor_status.json').write_text(json.dumps({'completed': False}), encoding='utf-8')
        features, h300 = extract(input_path(args.data_dir), out, args.workers)
        classify(features, out, args.workers)
        altitude(features, out, args.workers)
        if h300:
            sensor(features, h300, out, args.workers)
        else:
            print('A120 was not supplied; sensor evaluation skipped.', flush=True)
        tables(features, out, args.profile_table)
        from .plotting import figures
        figures(out)
    elif name == 'extract': extract(input_path(args.data_dir), out, args.workers)
    elif name == 'classify': classify(args.features, out, args.workers)
    elif name == 'altitude': altitude(args.features, out, args.workers)
    elif name == 'sensor': sensor(args.features, args.h300_features, out, args.workers)
    elif name == 'tables': tables(args.features, out, args.profile_table)
    elif name == 'figures':
        from .plotting import figures
        figures(out)
    elif name == 'profiles':
        from .profiles import run_controls
        run_controls(read_features(args.features), out, args.workers, args.cnn)
    elif name == 'height': height(input_path(args.input_dir), out, args.workers)
    elif name == 'density': density(input_path(args.input_dir), out, args.chunk_size, args.xy_unit)
    elif name == 'context': context(args, out)
    print(f'Finished {name}: {out}', flush=True)
