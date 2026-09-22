# GitHub release setup

Owner: `whliuo`  
Repository: `LiDAR-citrus-tree-classification`

**About:** Research code for LiDAR-based citrus canopy-condition classification,
including tree-level feature extraction, field validation, flight-altitude
analysis and cross-sensor evaluation.

**Topics:** `lidar`, `uav`, `citrus`, `precision-agriculture`, `plant-phenotyping`,
`point-cloud`, `remote-sensing`, `machine-learning`, `reproducible-research`.

The prepared `LiDAR-citrus-tree-classification.zip` contains only public release
files at its root. Extract it to a fresh checkout and upload those contents to
the target repository. No remote repository changes have been made.

The local directory is named `Submission` because that directory already
existed on a case-insensitive Windows filesystem. It is the requested
`submission/` location. Its two original Word files remain untouched; they
are excluded from both the ZIP and `.gitignore`-respecting Git additions.
Generated `results/`, `.cache/`, `_validation/`, `test_outputs/` and the ZIP
itself are local artifacts, not public payload. Keep `results/README.md`.

Before publishing:

1. Select a code license and confirm distribution terms for the research data.
2. Prepare `segtree.7z` from the finalized regraded observations in
   `segtree_height/`, preserving `020`, `040`, `060`, `080`, `100`, `120`, `A120`.
   There are 5,758 LAS files; individual point clouds are not committed to Git.
3. Publish the data separately and add its real download URL and SHA-256 to
   `data/README.md`. Raw flights for the optional density utility require
   their own distribution arrangement.
4. Add the final paper citation, authors, journal and DOI to README.
5. Upload the prepared public files and set the About text and topics above.

Do not upload the entire development archive. The public release can run from
its own checkout without any original `script/` file or original output folder.
