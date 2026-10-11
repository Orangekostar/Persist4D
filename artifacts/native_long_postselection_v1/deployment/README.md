# Trained native-long task bundles

These are complete nonencoder task parameters, additional feedback parameters
where applicable, and all required mutable buffers. The frozen Concerto encoder
is supplied separately. They are inference assets, not optimizer-resume checkpoints.

| Asset | Arm / update | Role |
|---|---|---|
| `E0-05940.pt` | E0 / 5,940 | CAL-selected pair-trained baseline |
| `E1-23760.pt` | E1 / 23,760 | CAL-selected long-trained baseline |
| `E1-02970.pt` | E1 / 2,970 | Same-update control for E3 |
| `E2-29700.pt` | E2 / 29,700 | CAL-selected stage-sampling arm, including its negative short-horizon result |
| `E3-02970.pt` | E3 / 2,970 | CAL-selected feedback arm |

All four training trajectories ran to 29,700 updates with seed 45. Earlier
selected updates are evaluation choices fixed by CAL, not early training stops.
The five binaries use Git LFS; a small pointer file is not a downloaded model.
`MODEL_ASSET_MANIFEST.json` binds bytes and SHA-256. The external publication
receipt records final branch/tag and independent remote-download checks.

Use the repository's existing ReScene/Concerto/PyTorch environment, then download
the LFS assets with `git lfs pull`. Obtain the separately licensed Concerto encoder
through its original distribution. Its required SHA-256 is
`845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`.
The small `rio.yaml` file contains the evaluator taxonomy, not scene annotations.
`../evidence/ENVIRONMENT_SOURCES.json` records the four clean third-party Git
commits and the controller package versions used for the actual portable reload.
The original source/initialization receipt separately binds installed Python
source hashes used by the experiment.

From the repository root, load a model with relocated encoder/taxonomy paths:

```python
import json
from pathlib import Path

from scripts.native_long_delivery import load_portable_bundle
from scripts.native_long_postselection import sha256

assets = Path("artifacts/native_long_postselection_v1/deployment")
manifest = json.loads((assets / "MODEL_ASSET_MANIFEST.json").read_text())
asset = next(r for r in manifest["assets"] if r["name"] == "E1-23760.pt")
bundle = assets / asset["name"]
assert bundle.stat().st_size == asset["bytes"]
assert sha256(bundle) == asset["sha256"]
system = load_portable_bundle(
    bundle,
    encoder=Path("/your/path/concerto_base.pth"),
    dataset_spec=assets / "rio.yaml",
    dataset_spec_sha256=manifest["taxonomy_sha256"],
)
system.eval()
```

The loader verifies the original numerical source identity and encoder hash,
relocates both encoder config references and the metric taxonomy, requires all
task parameters and mutable buffers, and loads strictly. Each accompanying JSON
receipt binds its full checkpoint, configuration, training identity, and an actual
exact trained-tensor export/reload. `PORTABLE_ASSET_AUDIT.json` additionally records
an actual CPU reload of all five bundles from relocated paths, checked against
every original task/buffer tensor and frozen encoder parameter.

Exact parameter reload does not establish repeatable inference. The bound runtime
has order-dependent outputs, and the larger mechanism panel also found differences
between fresh processes using the same input/protocol. The two affected inputs'
scalar AP/REC repeated, while their tensors and sufficient states could change.
See `../evidence/FRESH_PROCESS_RUNTIME_LIMITATION.json`; retain both the protocol
and the `FRESH_PROCESS_NONREPEATABILITY_OBSERVED` limitation.
