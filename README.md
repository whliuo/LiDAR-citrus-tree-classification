# LiDAR citrus tree classification

Research code for tree-level citrus canopy-condition classification from UAV
LiDAR. The release consolidates the finalized midpoint workflow, including
feature extraction, field validation, flight-altitude robustness, and L1/H300
sensor comparisons. The associated paper will provide the scientific definitions
and parameter rationale.

Target repository: https://github.com/whliuo/LiDAR-citrus-tree-classification

## Quick start

Use Python **3.14** (tested with 3.14.3). From this repository's root:

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install and inspect the CLI:

```bash
python -m pip install -r requirements.txt
python main.py --help
```

Obtain `segtree.7z` separately (distribution link pending), then extract its
segmented-tree folders into `data/segtree/`:

```text
data/segtree/
  020/   040/   060/   080/   100/   120/   A120/
```

If the archive contains a top-level `segtree_height` directory, put its contents
at that location, or pass its path with `--data-dir`. Do not leave an extra
directory level between the data root and `020/`. Both `020` and `20` naming
are supported, but only one may contain observations at each altitude.
See [data/README.md](data/README.md) for filenames, identifiers and counts.

Run the complete midpoint pipeline:

```bash
python main.py all --workers 8
```

This extracts all features, runs the nine fixed midpoint models, repeated
5-fold group CV × 10, leave-one-field-out validation, retrained merged-class
sensitivity, 3,000-resample paired comparisons, altitude and available sensor
analyses, then generates tables and figures. It does not optimize models.
`A120/` supplies H300 observations; if absent, `all` explicitly skips the sensor
stage. The `sensor` command itself requires that data.

Three unchanged Profile20 benchmark rows are appended from clearly labelled
archived controls, as in the original final audit. They are **not** reported as
retrained by `all` or `classify`. To recompute the LR/RF controls:

```bash
python main.py profiles --workers 8
python main.py tables --profile-table results/profiles/profile_benchmarks.csv
python main.py figures
```

To also recompute the fixed CNN, install its optional dependency first:

```bash
python -m pip install -r requirements-cnn.txt
python main.py profiles --cnn --workers 8
python main.py tables --profile-table results/profiles/profile_benchmarks.csv
python main.py figures
```

The CNN runs on CPU with the original architecture, training settings and
nested early stopping. Retain `data/observation_order.csv`: it preserves the
final audit's training-row order across parallel extraction runs. Numerical
and package-version reproducibility is documented in [VALIDATION.md](VALIDATION.md).

## Stages and outputs

Run stages separately to reuse extracted features:

```bash
python main.py extract --data-dir data/segtree --workers 8
python main.py classify --workers 8
python main.py altitude --workers 8
python main.py sensor --workers 8
python main.py tables
python main.py figures
```

| Stage | Default output |
| --- | --- |
| `extract` | `results/features/01_features_l1.csv`, `02_features_h300.csv`, extraction audit |
| `classify` | `results/classification/`: per-tree LOFO predictions, field metrics, CV repeats, five-class and merged-class summaries, bootstrap intervals |
| `altitude` | `results/altitude/`: six-altitude common-field metrics |
| `sensor` | `results/sensor/`: matched observations, agreement and transfer metrics |
| `tables` | `results/tables/`: model comparison, tree-grade comparison, field errors |
| `figures` | `results/figures/`: PNG/PDF plots from saved tables |
| `profiles` | `results/profiles/`: recomputed profile controls, predictions and CV metrics |

All paths are relative to the repository root, regardless of the working
directory. Inputs may be absolute paths. `--output-dir` must name a directory
inside this release; it cannot point into the source or reference-data folders.
Subcommands with a `--features` option accept another extracted table. If using
a custom output directory, explicitly point later standalone stages to its
feature files, for example:

```bash
python main.py extract --output-dir results_rerun
python main.py classify --features results_rerun/features/01_features_l1.csv --output-dir results_rerun
```

Analyses overwrite their own named outputs when rerun. Extraction always reads
the supplied point clouds anew; it never silently trusts an old feature cache.
Malformed filenames, duplicate observations, inconsistent labels and missing
required cohorts produce errors. Missing individual feature values retain the
original training-fold median imputation.

## Additional manuscript utilities

```bash
python main.py height --input-dir data/segtree/020
python main.py density --input-dir data/rawdata
python main.py context --visual-repeats data/context/visual_ratings.csv --qpcr data/context/qpcr.csv
```

`height` reproduces the 20-m height statistics and five distribution panels.
`density` needs **separate raw flight LAS files**, which are not segmented-tree
inputs; it writes footprint-density CSVs. Raw flights are not bundled with
this repository or implicitly included in `segtree.7z`.
`context` summarizes the supplied small visual-rating and qPCR tables as
independent evidence streams. It never matches qPCR records to LiDAR trees.
These commands write `results/height/`, `results/density/` and `results/context/`.

## Workflow and repository structure

```text
segmented LAS/LAZ
  -> midpoint features + 20 depth-profile bins
  -> fixed classification / field validation / altitude / sensor analyses
  -> saved prediction and metric tables
  -> figures and manuscript tables

main.py                     single executable CLI
src/                        shared extraction, models, analysis and plotting
data/observation_order.csv   stable identity/order metadata
data/reference/              labelled archived regression/benchmark tables
data/context/                optional visual and qPCR inputs
results/                    generated outputs (ignored by Git)
tests/                      regression checks requiring no point-cloud download
CODE_INVENTORY.md            all original scripts and consolidation decisions
ENVIRONMENT.md              environment setup and tested package versions
VALIDATION.md               checks actually performed and their results
GITHUB_SETUP.md             release contents, About text, topics and manual steps
```

The older six-class feature searches, radial/map model searches, and early
grading audits are documented in the inventory. The finalized filenames use
classes 0–4 directly; the release does not silently relabel earlier datasets.
Altitude comparisons use common **fields** 1–3 at each altitude, matching the
final implementation; they do not impose a newly defined tree intersection.

## Associated publication and citation

Publication details, author list, year, journal and DOI: **to be added**.
Please cite the associated paper once its final citation is available.
No publication identifier is invented by this release.

## License

**A license must be selected by the repository owner before public release.**
No license decision was present in the development archive. No new license or
data-redistribution terms are assigned here.
