# MultiScan Dataset Audit

Status: `METADATA_COMPLETE_RELEASE_BLOCKED`

- Official code commit: `697bc9ec86fb7d34d47cb4cdbddcfc3c7f18c605`
- Official data revision: `c62c9aad850a8638ac3e42605926a65707600125`
- Official inventory: 273 scans / 117 physical scenes
- Frozen collection: 23 scenes / 101 scans, all scenes with T>=3
- Release access: HTTP 401 `GatedRepo`; no authenticated Hugging Face session is present
- Raw geometry, annotations, alignment, scan metadata, and benchmark PTH were not downloaded

The official CSV inventory and taxonomy are auditable. Release-level identity,
gap, alignment, and model-coverage evidence are not auditable without licensed
file access. No GPU inference was run.
