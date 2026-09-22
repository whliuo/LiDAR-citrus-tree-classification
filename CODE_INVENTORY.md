# Development archive review and consolidation

The archive contains **38 Python files: 36 in `script/` and two in `script0/`**.
`fb_density.py` is empty; the other 37 programs were inspected. No research
notebooks, project README, license, pyproject or general environment specification
were found outside the installed `.venv`. Its Python configuration and the two
advanced-model requirements lists were reviewed. Installed dependency source
files are not counted as research scripts.

The recursive audit indexed 81,466 source lines, 532 CSV tables and 36 workbooks,
including schemas, row counts, workbook sheets, imports and data-flow literals.

All status labels below describe this release only. Every original script,
dataset, result, historical version and directory remains in the development
archive unchanged.

## Final version decision

The source and saved outputs agree on the final sequence:

1. V9D corrects and locks multi-altitude geometry and 20-bin profiles.
2. V9G/V9H evaluate the earlier 0.60-depth models and sensor variants.
3. Manuscript V2 adds fixed Profile20 LR/CNN controls and unlinked visual/qPCR
   summaries.
4. t02 extracts the midpoint candidates; t03 explicitly locks **0.50** and
   reruns the affected models, true merge01 retraining, altitude and sensor
   evaluations. Its saved `11_final_consistency_summary.txt` confirms that lock.

Accordingly, midpoint extraction comes from t02 and the main analysis from t03.
The public extraction computes only the finalized midpoint descriptors plus
unchanged Profile20 bins. Other t02 sensitivity thresholds are not reintroduced.
V9D supplies the profile-bin calculation; V9G supplies the RF profile-control
settings/seeds; manuscript V2 supplies the LR/CNN controls and ancillary stats.

## Per-script mapping

Names below are relative to the original archive. Most scripts are standalone
and pass CSVs or workbooks between stages rather than importing each other.
Earlier output locations named simply `processed/` or `segtree/` in source
were often renamed in the development archive; the public version does not
depend on recursive guessing among those historical locations.

| Original file | Function | Input | Output | Final status / public mapping |
| --- | --- | --- | --- | --- |
| `script/0_test_extract_density_penetration_features_fixed.py` | Initial six-class height/penetration/peak extraction and plots | Historical `segtree` LAS/LAZ | v1 feature CSV/XLSX, class/profile summaries, QC | Superseded by finalized extraction |
| `script/0_extract_density_penetration_features_v2.py` | Expanded whole-crown penetration, morphology and sector features | Historical segmented LAS/LAZ | v2 features, correlations, definitions, profiles | Superseded; six-class exploratory inventory |
| `script/04_extract_density_transmission_features_v4.py` | Appends transmission, grid and screening analyses to v2 code | Historical segmented LAS/LAZ | v3 features and threshold/model/LOFO tables | Superseded; source filename says v4 but final outputs are v3 |
| `script/04a_extract_citrus_surface_control_v4a.py` | Crown ROI and fitted upper-surface comparisons | Segmented LAS/LAZ | v4 surface-control features, QC and comparisons | Experimental; not part of locked midpoint workflow |
| `script/05_extract_citrus_radial_depth_v5.py` | Adds radial depth, residualization and ordinal/hierarchical variants | Segmented LAS/LAZ | v5 feature and evaluation tables | Experimental; superseded |
| `script/06_extract_citrus_all_designs_v6.py` | Large inherited feature-design benchmark | Segmented LAS/LAZ | v6 all-design features, rankings, label-scheme comparisons | Experimental; superseded |
| `script/07_citrus_locked_final_validation_v7.py` | Fixed candidate model validation and paired comparisons | v6 feature CSV | CV, LOFO, confusion, bootstrap and decision tables | Superseded by later regraded-data audits |
| `script/08_citrus_mechanism_locked_validation_v8.py` | Mechanism ablation and class/field diagnostics | v6 features | Validation and mechanism-diagnostic workbook/CSVs | Superseded by V9/t03 |
| `script/09b_citrus_multialtitude_mechanism_v9b.py` | Multi-altitude extraction, validation and transfer | `segtree_height` | V9B features, QC, model and altitude CSVs | Superseded by corrected V9D |
| `script/09c_citrus_multialtitude_mechanism_v9c.py` | V9 multi-altitude processing plus consolidated workbook | `segtree_height` | V9C features/results/workbook | Superseded by V9D |
| `script/09d_citrus_multialtitude_mechanism_v9d.py` | Corrected normalized geometry, fixed descriptors and altitude evaluations | Six L1 altitude folders | V9D features, QC, CV/LOFO, altitude tables/workbook | Integrated: unchanged Profile20 extraction and geometry lineage; other 0.60 models superseded |
| `script/09e_citrus_60m_final_locked_audit_v9e.py` | Focused 60-m audit using corrected features | V9D feature table | Decision, field, paired, transfer and merge01 outputs | Superseded by V9G then t03 |
| `script/09g_gcitrus_60m_final_locked_audit_v9g.py` | Locked 60-m and cross-sensor audit, profile RF control | V9D features and A120 LAS | V9G decision table, predictions, workbook | Integrated: Profile20 RF settings/seeds; midpoint analysis replaces its 0.60 branch |
| `script/09h_citrus_cross_sensor_robustness_v9h.py` | Sensor matching, harmonization and pooled models | V9D 120-m features and A120 LAS | Agreement, strategy, predictions and bootstrap tables | Retained conceptually; finalized subset comes from t03 sensor functions |
| `script/0_citrus_lidar_1dcnn_benchmark_3.py` | Broad full-profile/ordinal/hybrid CNN experiments | Historical segmented data and optional feature CSV | NPZ profiles, CV/LOFO, confusion figures | Experimental; superseded by fixed manuscript Profile20 control |
| `script/0_citrus_lidar_position_aware_1dcnn_v3a.py` | Position-aware CNN, binary tests and augmentation ablation | Historical segmented data and v2 features | Profile cache, model comparisons and plots | Experimental; not a dependency of final CNN |
| `script/0_d1_citrus_radially_constrained_penetration.py` | Core center/radius and radial-layer exploration | Historical segmented LAS | Core/layer features, CV/LOFO and contrasts | Experimental; geometry concept retained, final constants from t02 |
| `script/10_field4_grade_audit_60m.py` | Controlled effect of Field-4 grades on historical training | V9B/V9C features at 60 m | Field-4 review candidates, training-effect tables | Historical grading audit; not silently rerun with a different model |
| `script/11_field_inspection_predictions_60m.py` | Trusted-field-only inspection predictions and Field-4 predictions | V9B/V9C 60-m features | Per-tree grades, probabilities and inspection priority | Historical grading audit; its training design differs from final four-field LOFO |
| `script/12_export_field_errors_60m.py` | Reformats saved inspection predictions, no retraining | Script 11 honest-prediction CSV | Field errors and minimal inspection lists | Retained conceptually in `tables`; new exports explicitly use t03 LOFO predictions |
| `script/13_citrus_simple_mechanism_validation_regraded.py` | Regraded-data complexity ladder and matched-height cohorts | Regraded `segtree_height` | Feature cache, model/label/height comparisons | Superseded; distinct earlier geometry/CV settings are not mixed into t03 |
| `script/20_citrus_manuscript_final_statistics_v2.py` | Fixed profile LR/CNN controls, figure-data/table consolidation, optional context stats | V9D features, V9G/V9H results, visual/qPCR workbooks | Manuscript tables, profile predictions, context summaries | Integrated in `profiles`, `context`, `tables` and `figures`; old 0.60 model tables replaced by midpoint results |
| `script/d1a_compact_vertical_radial_benchmark.py` | Compresses earlier radial features and validates ablations | Radial core/layer CSVs | Compact feature table, CV/LOFO, binary and ablation tables | Experimental; not part of final feature representation |
| `script/d2_citrus_flight_height_radial_sensitivity.py` | Native versus 60-m-locked radial geometry and height transfer | Multi-height segmented LAS | Matched cohorts, feature stability, transfer and height tables | Experimental; different historical label/geometry design |
| `script/d2c_citrus_oof_ensemble_benchmark.py` | Fixed equal-weight ensembles of aligned OOF predictions | Advanced tabular and map OOF CSVs | Ensemble and component metrics | Experimental; final release does not ensemble selected candidates |
| `script/d2c_citrus_oof_ensemble_benchmark_LOCAL_v2.py` | Local variant of same OOF ensemble | Same OOF tables | Same family of ensemble tables | Near duplicate; removes foundation-model combination |
| `script/d3a_citrus_advanced_tabular_benchmark.py` | Broad tabular learner-family screening | Earlier radial core/layer CSVs | Model availability, CV, OOF and binary results | Experimental; optional CatBoost/XGBoost/LightGBM/EBM/TabPFN dependencies omitted |
| `script/d3a_citrus_advanced_tabular_benchmark_LOCAL.py` | Local-only tabular screening variant | Same radial tables | Same family of results without TabPFN | Near duplicate; not included in locked model set |
| `script/d3b_citrus_vertical_radial_map_benchmark.py` | 40×8 maps, PCA models and 2-D CNN | Segmented LAS and radial CSVs | NPZ maps, CV/OOF, binary metrics | Experimental; distinct representation from final Profile20 CNN |
| `script/fa_citrus_tree_height_distribution_020.py` | 20-m tree-height extraction and combined figure | 020 LAS/LAZ | Per-tree heights, summaries, figures/workbook | Superseded plotting layout; calculations retained via v2 |
| `script/fa_citrus_tree_height_distribution_020_v2.py` | Same height analysis with separate five-panel figures | 020 LAS/LAZ | Height CSVs and PNG/PDF/SVG panels | Integrated in `height`; final panel style retained |
| `script/fb_calculate_raw_las_point_density.py` | Chunked exact XY convex hull and footprint density | Raw flight LAS (optional LAZ) | Per-file/pooled densities and audit | Integrated in `density`; same calculations and chunk size |
| `script/fb_density.py` | Empty placeholder | None | None | Not used; original preserved |
| `script/t01_cpp_depth_sensitivity_v10.py` | Fixed depth-grid sensitivity and cross-height transfer | V9 feature table plus original LAS | Core-depth features, LOFO/paired/stability tables | Historical parameter audit; no new optimization in release |
| `script/t02_citrus_midpoint_penetration_audit_v1.py` | Extracts threshold candidates and audits midpoint versus 0.60 | L1 altitude folders and A120 | Midpoint feature tables, candidate/altitude/sensor metrics | Integrated: finalized 0.50 extraction and identity metadata; selection audit not rerun |
| `script/t03_citrus_midpoint_final_consistency_v1.py` | Final midpoint model hierarchy, merged-label retraining, altitude/sensor tests | t02 midpoint tables plus optional locked profile results | Final prediction/metric/manuscript tables | Integrated: authoritative `classify`, `altitude`, `sensor`, `tables` |
| `script0/00_clip_las_height.py` | Clips raw flights by tree polygons and writes named segments | Raw LAS and polygon shapefiles; machine-specific directories | Per-tree segmented LAS and clipping log | Upstream preprocessing; not needed once `segtree.7z` is supplied |
| `script0/00a_clip_las_height_coord_check.py` | LAS/polygon coordinate and coverage diagnostic | Raw LAS and polygon shapefiles | Printed coordinate/CRS/coverage checks | Upstream diagnostic; not in final analysis |

## Duplication removed from the release

An AST comparison found **465 repeated function/class definition occurrences**
across the 38 files. The v2→v4/v4a→v5→v6 scripts repeatedly embed earlier
definitions and append replacement implementations. Shared parsing,
normalization, geometry, penetration, peak, CV and metric routines account for
much of this duplication. V9B/C/D and V9E/G/H repeat related evaluation blocks;
the LOCAL model/ensemble variants are near duplicates. The two height scripts
share calculations but differ in figure layout.

The public release has one primary executable, `main.py`. Helpers are:

| Module | Responsibility |
| --- | --- |
| `src/settings.py` | Final model inventory, geometry settings, seeds and fold counts |
| `src/features.py` | Shared L1/H300 midpoint extraction and Profile20 bins |
| `src/io_utils.py` | Paths, data discovery, identifier validation, stable row order and run records |
| `src/models.py` | Class transforms, threshold/LR/RF models and metrics |
| `src/analysis.py` | Group CV, LOFO, bootstrap, altitude and sensor evaluation |
| `src/workflow.py` | Stage orchestration and standardized output tables |
| `src/profiles.py` | Fixed LR/RF/CNN profile controls |
| `src/context.py` | Independent visual-repeatability and qPCR summaries |
| `src/heights.py` | Height calculations and original v2 panels |
| `src/density.py` | Raw flight footprint density |
| `src/plotting.py` | Comparison figures from saved results |

Source functions were selected from reviewed authoritative versions; original
modules are never imported at runtime. Model mathematics, settings and seed
schedules are preserved. Changes address IO, validation, row ordering,
relative filenames, dead code, repeated implementations and separating plots
from fitting. CSVs are the standardized outputs; historical multi-sheet Excel
bundles and console upload instructions are not duplicated.

## Other archive material reviewed

- `segtree_height/`: 5,758 LAS observations in seven folders plus eight LiData
  sidecars. All LAS files were read in regression extraction. No original
  directory literally named `segtree/` is currently present.
- `rawdata/`: 24 raw LAS flights, two LiData sidecars and four shapefile groups.
  All LAS files were read in density regression. `rawpolytree 4/` and
  `rawpolytree 5/` hold alternate upstream polygon versions; neither is moved.
- `processed v1/v2/v4*/v5/v6/`: historical six-class feature searches and
  validation results. V6 includes locked V7/V8 result subdirectories. Two
  V7/V8 workbook ZIPs lack `xl/workbook.xml`; they are not relied on.
- `processed v9/`, `results_multialtitude_v9c/`, the two V9D output directories,
  V9E/V9G/V9H directories: successive regraded/multi-altitude audits. Both V9D
  feature tables contain 4,980 observations; finalized profile vectors were
  compared against the corrected V9D table.
- Radial, compact, flight-height, advanced-tabular/ensemble, map and both
  deep-learning result folders: experimental CSV/NPZ/figure outputs matched
  to their script families, not treated as required final-stage inputs.
- `results_cpp_depth_sensitivity_v10/`, `results_midpoint_penetration_audit_v1/`,
  `results_midpoint_final_consistency_v1/`, `results_manuscript_final_v2/`:
  parameter-decision history and authoritative finalized output lineage.
- `precision/`, `qpcr/`, `qpcr processed/`: spreadsheets inspected by sheet
  inventory; only finalized visual ratings and qPCR inputs feed the optional
  public context stage. Earlier raw collaborator tables remain archived.
- `results_tree_height_020/`, `results_raw_point_density/`: independent
  characterization utilities and their reference results.
- `.vscode/`, requirements lists, run JSON/text files: development settings,
  personal paths and environment assumptions. Original figure files and NPZ
  caches remain unchanged. The Word documents in `writing/` and `Submission/`
  include empty placeholders; they provide no final citation/license decision.

The local audit under `_validation/` records recursive file types, CSV schemas,
workbook sheets, script imports/functions/IO literals and source hashes. It is
excluded from the public ZIP and Git because it describes the development
archive. No installed-library code is bundled.

## Personal paths and reproducibility repairs

Most historical programs derive paths relative to a parent project and then
search recursively for old outputs. Run-config files and feature-table `File`
columns also contain absolute development paths. The clipping program has
explicit drive-based paths. Several later scripts have path examples/comments
or fallback searches. New code takes explicit input arguments, keeps all
outputs below the release, and records portable relative source filenames.

Parallel t02 extraction stored rows in worker-completion order. t03 consumed
that order directly, affecting RF resampling. The small observation-order
manifest preserves it deterministically. Profile controls use their original
V9D sorted field/tree/sensor order instead. Exact source lineage and tested
numeric equivalence are recorded in `VALIDATION.md` and
`data/reference/provenance.json`.
