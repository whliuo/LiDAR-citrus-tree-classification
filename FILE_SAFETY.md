# File-safety verification

**The original project outside `submission/` was not modified.** On this
Windows filesystem, the existing directory name `Submission/` is equivalent
to the requested `submission/`; it was not renamed.

Final baseline comparison passed for **57,361 original files**:

- **6,597 research/source/data/result and existing document files**: identical
  file paths, sizes, modification timestamps and SHA-256 content digests.
- **50,764 installed `.venv` files**: identical file paths, sizes and modification
  timestamps.
- No missing files, renamed/moved files, overwritten files, changed original
  source/result files or unexpected new entries at the project root.
- The two pre-existing files `Submission/chatgpt.docx` and
  `Submission/~$hatgpt.docx` retain their original content and timestamps.

New files are confined to `Submission/`, including generated results,
`test_outputs/`, caches, local audit artifacts and the public ZIP. Existing
Word files, local caches, test outputs and development-audit artifacts are
excluded from the public ZIP. The ZIP contains no LAS/LAZ files or personal
absolute filesystem paths in executable code.

Local evidence: `_validation/original_baseline.json` and
`_validation/file_safety.json`. These describe the development archive and
are intentionally excluded from the public upload.
