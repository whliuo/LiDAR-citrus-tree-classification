# Environment

Recommended and tested interpreter: **CPython 3.14.3, Windows x64**.
Python 3.14 is recommended for the pinned numerical stack. Other operating
systems have not been regression-tested for this release.

From the repository root:

```bash
python -m venv .venv
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

Windows Command Prompt: `.venv\Scripts\activate.bat`.
Linux/macOS: `source .venv/bin/activate`.

```bash
python -m pip install -r requirements.txt
python main.py --help
python -m unittest discover -s tests -v
```

The numerical/model packages are pinned because the goal is reproducing
archived predictions, rather than changing solver or random-forest behavior.
The requirements contain only imports used by the consolidated workflow.
They do not export the unrelated development environment.

| Dependency | Tested version | Use |
| --- | --- | --- |
| NumPy | 2.4.6 | feature calculations and arrays |
| pandas | 3.0.5 | feature/result tables |
| SciPy | 1.18.0 | statistics, KDE, convex hull |
| scikit-learn | 1.9.0 | locked classifiers and evaluation |
| laspy | 2.7.0 | LAS/LAZ reading |
| lazrs | 0.8.2 | compressed LAZ backend |
| Matplotlib | 3.11.1 | figures |
| openpyxl | 3.1.5 | optional XLSX input |
| pyproj | 3.7.2 | raw-LAS CRS/unit inspection |
| PyTorch (optional) | 2.13.0 | fixed Profile20 CNN retraining |

Use `python -m pip install -r requirements-cnn.txt` for CNN retraining. No GPU,
downloaded model checkpoint, account, or API is used by the final classifiers.
The standard pipeline can append the documented archived Profile20 results
without importing PyTorch.

`main.py` keeps temporary files, Matplotlib state and Torch cache under `.cache/`
inside the release, disables bytecode creation, and limits numerical-library
threads to one per worker. `--workers` controls extraction/evaluation workers
and RF CPU allocation without changing seeds, folds or model settings.
The tested default worker count is at most eight.

Validation used the existing local interpreter and inspected installed package
metadata. A fresh network dependency installation was not performed. Installer
availability on other platforms and future package indexes is not asserted.
