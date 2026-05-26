#!/usr/bin/env python3
# ==========================================
# File: backend/scripts/promote_model.py
# Role: Safe promotion of model bundles to production.
# Logic: Validates bundle integrity and dataset hash before updating production symlink.
# ==========================================

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Tuple

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.utils.ops_reporting import file_sha256

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("promote_model")

STRICT_ARTIFACTS = (
    "metadata.json",
    "home_pipe.joblib",
    "away_pipe.joblib",
    "win_pipe.joblib",
    "score_preprocessor.joblib",
    "win_preprocessor.joblib",
)
OPTIONAL_REPORTS = ("training_report.json", "run_summary.json")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def validate_bundle(bundle_path: Path) -> Tuple[bool, str]:
    """Check if the bundle contains all required artifacts."""
    if not bundle_path.is_dir():
        return False, f"Bundle path is not a directory: {bundle_path}"

    missing = []
    for req in STRICT_ARTIFACTS:
        if not (bundle_path / req).exists():
            missing.append(req)
    if missing:
        return False, f"Missing required artifacts: {', '.join(missing)}"

    try:
        metadata = _read_json(bundle_path / "metadata.json")
        report_path = bundle_path / "training_report.json"
        report = _read_json(report_path) if report_path.exists() else {}
        for optional_name in OPTIONAL_REPORTS:
            optional_path = bundle_path / optional_name
            if optional_path.exists():
                _read_json(optional_path)
    except Exception as exc:
        return False, f"Invalid JSON metadata/report: {exc}"

    metadata_hash = metadata.get("dataset_hash")
    report_hash = report.get("dataset_hash")
    if metadata_hash and report_hash and metadata_hash != report_hash:
        return False, "metadata.json dataset_hash does not match training_report.json dataset_hash"

    serving_mode = metadata.get("serving_mode")
    if serving_mode and serving_mode != "pipeline_primary":
        return False, f"Unsupported serving_mode: {serving_mode}"

    return True, "Bundle is valid"


def promote(source_bundle: Path, target_dir: Path):
    """Promote source bundle to target directory."""
    source_bundle = source_bundle.resolve()
    target_dir = target_dir.resolve()
    if source_bundle == target_dir:
        raise ValueError("Source bundle and target directory must be different paths")

    # 1. Validate
    is_valid, msg = validate_bundle(source_bundle)
    if not is_valid:
        raise ValueError(f"Bundle validation failed: {msg}")

    # 2. Copy into a temporary sibling first so the current target stays intact
    # until the replacement bundle has been fully materialized and validated.
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = target_dir.parent / f".{target_dir.name}.promote_tmp"
    backup_dir = target_dir.parent / f".{target_dir.name}.promote_backup"
    _remove_path(tmp_dir)
    shutil.copytree(source_bundle, tmp_dir)
    is_valid, msg = validate_bundle(tmp_dir)
    if not is_valid:
        _remove_path(tmp_dir)
        raise ValueError(f"Copied bundle validation failed: {msg}")

    # 3. Swap target with rollback.
    _remove_path(backup_dir)
    if target_dir.exists() or target_dir.is_symlink():
        logger.info(f"Backing up existing production bundle at {target_dir}")
        shutil.move(str(target_dir), str(backup_dir))

    logger.info(f"Promoting {source_bundle} -> {target_dir}")
    try:
        shutil.move(str(tmp_dir), str(target_dir))
    except Exception:
        if not target_dir.exists() and backup_dir.exists():
            shutil.move(str(backup_dir), str(target_dir))
        raise
    else:
        _remove_path(backup_dir)

    # 4. Verify Hash of metadata
    meta_hash = file_sha256(target_dir / "metadata.json")
    logger.info(f"Promotion complete. Production Metadata Hash: {meta_hash}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote a trained model bundle to production.")
    parser.add_argument("--source", type=Path, required=True, help="Path to the trained bundle directory")
    parser.add_argument("--target", type=Path, default=Path("backend/data/models/current"), help="Production target directory")

    args = parser.parse_args()

    try:
        promote(args.source, args.target)
        print("SUCCESS: Model promoted to production.")
    except Exception as e:
        print(f"ERROR: Promotion failed: {e}")
        sys.exit(1)
