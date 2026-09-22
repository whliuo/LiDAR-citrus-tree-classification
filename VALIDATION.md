# Regression and release validation

Validation was performed against the local development archive on Windows
with CPython 3.14.3 and the versions recorded in `ENVIRONMENT.md`. Tests write
only below this release directory. Original scripts were read, not executed
with their historical output destinations.

## Workflows executed

| Workflow | Actual data/check | Result |
| --- | --- | --- |
| Midpoint extraction | All 5,758 segmented LAS files: 4,980 L1 + 778 H300 | All retained original columns, identifiers and row order match t02 feature CSVs; portable `File` paths intentionally differ |
| Profile extraction | All 4,980 L1 observations × 20 depth bins | All vectors match corrected V9D features |
| Final classification | Nine methods; 882 trees; five-class and genuinely retrained merge01 LOFO | All 15,876 prediction rows match t03 exactly |
| Field/CV summaries | 72 field-metric rows; 90 repeated-CV rows; nine main and five merged-class table rows | Match t03 |
| Paired uncertainty | All 30 metric comparisons with 3,000 resamples each | Point differences and confidence intervals match t03 |
| Altitude | Six flight heights on fields 1–3; 12 summary and 36 field rows | Match t03 |
| Sensor | 778 matched L1/H300 trees; four feature-agreement rows and 20 method/strategy rows | Match t03 |
| Profile20 LR | Full 5×10 group CV and four-field LOFO, 882 predictions | Metrics and predictions match manuscript V2 |
| Profile20 RF | Full 5×10 group CV and four-field LOFO | Compared aggregate metrics and CV accuracy match V9G/t03 |
| Profile20 CNN | Fixed CPU CNN, original nested epoch selection, full 5×10 CV and four-field LOFO | Audit metrics, per-field metrics/epochs and all 882 predictions match manuscript V2 |
| Tree heights | All 882 trees at 20 m; nine below the original 0.5-m cutoff; 873 retained | Per-tree heights, excluded trees and all five summary rows match original results |
| Raw density | All 24 raw flight LAS files | Per-file and pooled footprint-density tables match original results |
| Visual repeatability | 882 rows × four ratings | Summary and rating-range distribution match manuscript V2 |
| qPCR context | 1,073 rows; no LiDAR linkage | Dataset, pooled, level and shared-domain tables match manuscript V2 |
| Tables/figures | Saved analysis outputs; profile controls explicitly labelled by provenance | Generated model, altitude and sensor comparison PNG/PDFs; five original-style height panels |
| Public regression tests | Six tests: archived primary LOFO, stable ordering, duplicate/label checks, parser and output boundary | Passed |
| End-to-end CLI | `all` against the entire segmented dataset in a separate `test_outputs/end_to_end/` directory | Passed through extraction, classification, altitude, sensor, tables and figures |
| CLI validation | Main help and all 11 subcommand help pages; invalid output path, missing input and nonpositive chunk size | Help succeeds; invalid requests fail with clear nonzero-exit errors |

Numeric comparisons used `atol=1e-12`, `rtol=1e-12`, with equal missing values.
Discrete predictions and identifiers agree exactly. Rows are compared using
explicit scheme/method/field/tree/altitude/strategy keys where appropriate.
The package does not claim byte-identical PDF/SVG output: timestamps, fonts
and backend details can differ.

The three comparison figures and overall height panel were visually inspected
for readable labels, axes, legends and clipping. The comparison figures are
new views of finalized result tables; no historical equivalent image is
claimed. Height calculations and the v2 panel drawing code are preserved.

## Reference files

Primary comparisons use originals in:

- `results_midpoint_penetration_audit_v1/01_features_l1.csv` and
  `02_features_h300.csv`;
- `processed results_multialtitude_v9d/features_multialtitude_v9d.csv`;
- all ten CSVs in `results_midpoint_final_consistency_v1/`;
- Profile20 LR/CNN audits, fields and predictions, plus visual/qPCR outputs in
  `results_manuscript_final_v2/`;
- `results_tree_height_020/` and `results_raw_point_density/`.

Selected small reference tables, an 882-tree midpoint regression fixture,
primary-model prediction fixture and source hashes are included under
`data/reference/`. The archived profile benchmark table is explicitly labelled
and bound to its original 60-m tree identities, labels and profile values.
The 12-significant-digit profile fingerprint is a compatibility check, not a
cryptographic claim about raw-data identity.

Re-run the download-free tests:

```bash
python -m unittest discover -s tests -v
```

For full regeneration, obtain the segmented archive and follow README. To
recompute all profile controls, also run `profiles --cnn`; `classify` and `all`
preserve the original final audit's use of explicitly archived profile rows.

## Limits

- A clean network dependency installation and other OS/Python combinations
  were not tested. The recorded environment reproduces the available results.
- Original clipping/segmentation, broad model searches, earlier label schemes
  and threshold-selection experiments were not rerun or optimized.
- LAZ reading is supported through laspy/lazrs, but the available regression
  point clouds were LAS.
- No data download URL, archive checksum, citation, DOI or license was invented.
- Saved classification tables contain metadata and numeric provenance; model
  serialization and deployment on an independently labelled cohort are outside
  the finalized research workflow.

## Upload-package checks

The public ZIP contains 42 files, including the SHA-256 manifest, and excludes
point clouds, original Word files, local audit tools, generated results and
caches. Its code and tests total 2,487 lines, compared with 81,466 lines in the
38 original Python files. ZIP integrity and the explicit public-file list
were checked. After extraction into a separate directory, CLI help and all
six fixture regression tests passed without importing original scripts or
reading original result directories.

## Original-file safety

Before generating release files, a baseline recorded names, sizes and
modification timestamps for every original file; research/data/result files
also received SHA-256 content checks. Installed `.venv` files were tracked by
names, size and modification timestamp. Final verification results are recorded
in `FILE_SAFETY.md` and the local `_validation/file_safety.json`.

All release code, metadata, generated outputs, caches, tests, audit artifacts
and the upload ZIP are below `Submission/` (the Windows-equivalent
`submission/`). The existing Word documents there are preserved and excluded
from the public package.
