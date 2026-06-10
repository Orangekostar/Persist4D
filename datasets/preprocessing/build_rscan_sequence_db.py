"""
Build a 3RScan (RScan) sequence database YAML from *already preprocessed* outputs.

This script only regenerates the sequence index/metadata (and optionally the per-sequence
change label files) and does NOT run per-scan preprocessing.

Example:
  python scripts/build_rscan_sequence_db.py \
    --data_dir /path/to/3RScan \
    --processed_dir /path/to/mask3d-3RScan-processed \
    --sequence_type sliding \
    --sequence_length 3
"""

from pathlib import Path
import sys

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.preprocessing.RScan_preprocessing import RScanPreprocessing


def main(
    data_dir: str,
    processed_dir: str,
    metadata_file: str = None,
    sequence_type: str = "sliding",
    sequence_length: int = 2,
    scannet200: bool = True,
):
    """
    Args:
        data_dir: Path to raw 3RScan root (must exist; used for reading metadata json).
        processed_dir: Path to the existing processed output dir (contains instance_gt/, etc).
        metadata_file: Optional path to 3RScan.json (defaults to <data_dir>/3RScan.json).
        sequence_type: One of {'sliding','exhaustive','single'}.
        sequence_length: Number of scans per sequence (ignored by 'single').
        scannet200: Whether preprocessing used scannet200 mapping (must match your processed_dir).
    """

    data_dir_p = Path(data_dir)
    processed_dir_p = Path(processed_dir)
    if not data_dir_p.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir_p}")
    if not processed_dir_p.exists():
        raise FileNotFoundError(f"processed_dir does not exist: {processed_dir_p}")

    # Quick sanity check: change generation reads instance_gt for train/validation.
    ig_train = processed_dir_p / "instance_gt" / "train"
    ig_val = processed_dir_p / "instance_gt" / "validation"
    if not ig_train.exists() and not ig_val.exists():
        logger.warning(
            "instance_gt/{train,validation} not found under processed_dir. "
            "Sequence YAML creation may fail if change_gt generation is attempted."
        )

    builder = RScanPreprocessing(
        data_dir=str(data_dir_p),
        save_dir=str(processed_dir_p),
        metadata_file=metadata_file,
        scannet200=scannet200,
    )

    sequence_db = builder.process_sequences(sequence_type=sequence_type, sequence_length=sequence_length)

    out = str(processed_dir_p / f"sequence_database_{sequence_type}_{sequence_length}.yaml")
    logger.info(f"Wrote {len(sequence_db)} sequences to {out}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build an RScan/3RScan sequence_database_*.yaml from existing processed outputs."
    )
    parser.add_argument("--data_dir", required=True, help="Path to raw 3RScan root (must exist).")
    parser.add_argument("--processed_dir", required=True, help="Path to existing processed output dir.")
    parser.add_argument("--metadata_file", default=None, help="Optional path to 3RScan.json.")
    parser.add_argument(
        "--sequence_type",
        default="sliding",
        choices=["sliding", "exhaustive", "single"],
        help="Sequence sampling strategy.",
    )
    parser.add_argument("--sequence_length", type=int, default=2, help="Number of scans per sequence.")
    parser.add_argument(
        "--scannet200",
        action="store_true",
        help="Use ScanNet200 mapping (must match your processed_dir).",
    )
    parser.add_argument(
        "--no_scannet200",
        action="store_true",
        help="Disable ScanNet200 mapping (must match your processed_dir).",
    )

    args = parser.parse_args()

    scannet200 = True
    if args.no_scannet200:
        scannet200 = False
    elif args.scannet200:
        scannet200 = True

    main(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        metadata_file=args.metadata_file,
        sequence_type=args.sequence_type,
        sequence_length=args.sequence_length,
        scannet200=scannet200,
    )


