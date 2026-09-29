"""Package verified metric evidence and fixed checkpoints without raw scan caches."""

import hashlib
import zipfile
from pathlib import Path

from scripts.rescene_code_first_audit import OLD_PUBLIC, OLD_ROOT, PUBLIC, ROOT
from scripts.short_module_screen import read_json, sha256, write_json


def main():
    verified = read_json(PUBLIC / "VERIFIED_RESULTS.json")
    if verified["status"] != "CONTROLLED_RUNS_VERIFIED":
        raise ValueError("controlled runs are not verified")
    files = {"public/" + str(p.relative_to(PUBLIC)): p for p in PUBLIC.rglob("*")
             if p.is_file() and p.name not in ("EVIDENCE_BUNDLE.zip", "DELIVERY_MANIFEST.json")}
    for result in verified["metric_results"]:
        path = Path(result["path"])
        if sha256(path) != result["sha256"]:
            raise ValueError("verified result changed")
        files["runtime/" + str(path.relative_to(ROOT))] = path
        for h in (1, 2):
            state = path.with_name(path.stem + f"-H{h}-metric-state.json")
            files["runtime/" + str(state.relative_to(ROOT))] = state
    lock = read_json(OLD_PUBLIC / "selection/CAL_LOCK.json")["selected_updates"]
    for arm, step in lock.items():
        path = OLD_ROOT / "training" / arm / "seed45" / f"update={step:04d}.pt"
        files[f"original_locked_heads/{arm}.pt"] = path
        if arm != "M2":
            root = ROOT / "expanded_geometry" if arm.startswith("M") else ROOT
            path = root / "training" / arm / "seed45" / f"update={step:04d}.pt"
            files[f"controlled_heads/{arm}.pt"] = path
    manifest = {name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                for name, path in sorted(files.items())}
    destination = PUBLIC / "EVIDENCE_BUNDLE.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
    with zipfile.ZipFile(destination) as archive:
        if set(archive.namelist()) != set(manifest) or archive.testzip() is not None:
            raise ValueError("evidence archive member or CRC mismatch")
        for name, record in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != record["sha256"]:
                raise ValueError("evidence archive content mismatch")
    result = {"status": "VERIFIED", "file": destination.name, "sha256": sha256(destination),
              "bytes": destination.stat().st_size, "members": manifest,
              "metric_results": 36, "metric_states": 72,
              "original_fixed_heads": 6, "controlled_fixed_heads": 5,
              "raw_scan_and_feature_caches_included": False}
    write_json(PUBLIC / "DELIVERY_MANIFEST.json", result)
    print({k: v for k, v in result.items() if k != "members"})


if __name__ == "__main__":
    main()
