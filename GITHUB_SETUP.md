# Data distribution and verification

This public repository provides the classification workflow, reference tables and the segmented point-cloud archive. Large point-cloud data are stored with Git LFS at `data/segtree_height.7z`.

After cloning, retrieve the archive with:

```bash
git lfs pull --include="data/segtree_height.7z"
```

See [README.md](README.md) for installation and analysis commands, and [data/README.md](data/README.md) for extraction instructions, dataset counts and the archive checksum.

`RELEASE_SHA256.txt` lists every published file except itself. Its archive entry describes the downloaded archive bytes, rather than the Git LFS pointer. `data/segtree_height.7z.sha256` provides the same archive checksum separately.

The `.gitattributes` file records Git LFS tracking. Generated results, extracted point clouds and caches are excluded from Git; `results/README.md` describes the output directory.
