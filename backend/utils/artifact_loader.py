# ==========================================
# File: backend/utils/artifact_loader.py
# Role: Download/extract model + dataset artifacts from URLs at runtime.
# Input Data: Environment variables pointing to artifact archives.
# Output Data: Files written under MODELS_DIR / DATA_DIR.
# Dependencies: __future__, os, pathlib, tempfile, urllib, zipfile, tarfile
# Notes: Use for production deployments where artifacts live in object storage.
# ==========================================

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from ..config import DATA_DIR, MODELS_DIR, TRUTHY

log = logging.getLogger(__name__)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading artifacts from %s", url)
    with urllib.request.urlopen(url) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f)


def _extract(archive_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    entries = list(dest_dir.iterdir())
    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]
    if len(dirs) == 1 and not files:
        return dirs[0]
    return dest_dir


def _sync_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_root = dst / rel
        target_root.mkdir(parents=True, exist_ok=True)
        for d in dirs:
            (target_root / d).mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(Path(root) / f, target_root / f)


def _has_any_files(path: Path) -> bool:
    if not path.exists():
        return False
    for p in path.rglob("*"):
        if p.is_file():
            return True
    return False


def _fetch_bundle(url: str, target_dir: Path, *, force: bool, skip_if_present: bool) -> None:
    if skip_if_present and _has_any_files(target_dir) and not force:
        log.info("Skipping artifact download; %s already populated", target_dir)
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        archive_path = tmp_dir / "artifact.bundle"
        _download(url, archive_path)
        extract_root = _extract(archive_path, tmp_dir / "extract")
        _sync_tree(extract_root, target_dir)
        log.info("Artifacts synced to %s", target_dir)


def ensure_artifacts() -> None:
    """Download model/data artifacts if URLs are provided via env vars."""
    model_url = os.getenv("MODEL_BUNDLE_URL")
    data_url = os.getenv("DATA_BUNDLE_URL")
    bundle_url = os.getenv("ARTIFACT_BUNDLE_URL")

    if not any([model_url, data_url, bundle_url]):
        return

    force = os.getenv("ARTIFACT_FORCE", "false").strip().lower() in TRUTHY
    skip_if_present = os.getenv("ARTIFACT_SKIP_IF_PRESENT", "true").strip().lower() in TRUTHY

    if bundle_url:
        backend_root = Path(__file__).resolve().parents[1]
        target_root = Path(os.getenv("ARTIFACT_DIR", str(backend_root))).resolve()
        _fetch_bundle(bundle_url, target_root, force=force, skip_if_present=skip_if_present)

    if model_url:
        _fetch_bundle(model_url, MODELS_DIR, force=force, skip_if_present=skip_if_present)

    if data_url:
        _fetch_bundle(data_url, DATA_DIR, force=force, skip_if_present=skip_if_present)
