#!/usr/bin/env python3
# ==========================================
# File: backend/builddataset.py
# Role: Canonical dataset build entrypoint.
# Input Data: Raw NFL data, Schedule assets.
# Output Data: Cleaned feature datasets (CSV), Dataset manifests.
# Dependencies: pandas, pathlib, backend.pipeline_models
# Notes: Wrapper around build_csv_datasets_v3.py to provide stable run directories.
# ==========================================
"""
Canonical dataset build entrypoint.

This wrapper keeps the existing feature engineering logic in
`build_csv_datasets_v3.py`, but adds cleaner run directories, dataset cleaning,
typed manifests, and operator-friendly logging/output paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.pipeline_models import DatasetArtifactManifest, DatasetBuildConfig
from backend.score_sync import extract_score_entries_from_dataframe, write_score_snapshot


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _make_run_id() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _pick_latest_dataset_file(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("game_features_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No dataset artifacts found in {run_dir}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _ensure_game_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = {"season", "week", "home_team", "away_team"}
    if "game_id" in out.columns:
        out["game_id"] = out["game_id"].fillna("").astype(str).str.strip()
        return out
    if not required.issubset(out.columns):
        return out

    season = pd.to_numeric(out["season"], errors="coerce")
    week = pd.to_numeric(out["week"], errors="coerce")
    home = out["home_team"].astype(str).str.strip().str.upper()
    away = out["away_team"].astype(str).str.strip().str.upper()
    out["game_id"] = [
        f"{int(season_val)}-{int(week_val)}-{home_val}-{away_val}"
        if pd.notna(season_val) and pd.notna(week_val) and home_val and away_val
        else ""
        for season_val, week_val, home_val, away_val in zip(season, week, home, away)
    ]
    return out


def _clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply predictable cleanup so training reads a stable dataset."""

    out = df.copy()
    out.columns = out.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    before_rows = len(out)

    blank_mask = out.isna().all(axis=1)
    if blank_mask.any():
        out = out.loc[~blank_mask].copy()

    out = _ensure_game_id(out)

    duplicate_game_ids_removed = 0
    if "game_id" in out.columns:
        label_cols = [c for c in ("home_points_for", "away_points_for", "home_win", "winner") if c in out.columns]
        if label_cols:
            out["_label_priority"] = out[label_cols].notna().sum(axis=1)
        else:
            out["_label_priority"] = 0
        out["_completeness"] = out.notna().sum(axis=1)
        out = out.sort_values(
            ["_label_priority", "_completeness", "game_id"],
            ascending=[False, False, True],
            kind="stable",
        )
        valid_game_ids = out["game_id"].fillna("").astype(str).str.strip().ne("")
        if valid_game_ids.any():
            before_dedupe = int(valid_game_ids.sum())
            deduped = out.loc[valid_game_ids].drop_duplicates(subset=["game_id"], keep="first")
            duplicate_game_ids_removed = before_dedupe - len(deduped)
            out = pd.concat([deduped, out.loc[~valid_game_ids]], axis=0, ignore_index=True)
        out = out.drop(columns=["_label_priority", "_completeness"], errors="ignore")

    # Drop optional feature columns that are entirely empty or constant. Keep
    # identity and target columns protected so the trainer contract stays intact.
    protected_columns = {
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
        "home_points_for",
        "away_points_for",
        "home_win",
        "winner",
    }
    cols_to_drop = []
    for col in out.columns:
        if col in protected_columns:
            continue
        if out[col].isna().all():
            cols_to_drop.append(col)
            continue
        try:
            if out[col].dropna().nunique() <= 1:
                cols_to_drop.append(col)
        except TypeError:
            continue

    if cols_to_drop:
        out = out.drop(columns=cols_to_drop)

    sort_columns = [column for column in ("season", "week", "game_id") if column in out.columns]
    if sort_columns:
        out = out.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    completed_rows = 0
    future_rows = 0
    if "home_win" in out.columns:
        completed_rows = int(out["home_win"].notna().sum())
        future_rows = int(out["home_win"].isna().sum())

    stats = {
        "rows_before_cleaning": before_rows,
        "rows_after_cleaning": len(out),
        "blank_rows_removed": int(blank_mask.sum()),
        "duplicate_game_ids_removed": duplicate_game_ids_removed,
        "dropped_empty_or_constant_columns": len(cols_to_drop),
        "dropped_empty_or_constant_column_names": cols_to_drop,
        "completed_rows": completed_rows,
        "future_rows": future_rows,
    }
    return out, stats


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _manifest_path(path: Path, base_dir: Path) -> str:
    resolved_path = path.resolve()
    resolved_base = base_dir.resolve()
    try:
        return resolved_path.relative_to(resolved_base).as_posix()
    except ValueError:
        return str(resolved_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and clean the canonical NFL dataset.")
    parser.add_argument("--start", type=int, default=2018, help="Start season inclusive.")
    parser.add_argument("--end", type=int, default=2025, help="End season inclusive.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="backend/data/datasets",
        help="Root directory for dataset runs and promoted clean artifacts.",
    )
    parser.add_argument("--encode", choices=["onehot", "none"], default="onehot")
    parser.add_argument("--save-dominance-matrix", action="store_true")
    parser.add_argument(
        "--no-calibration-rows",
        dest="no_calibration_rows",
        action="store_true",
        help="Keep calibration-only rows out of the exported training dataset.",
    )
    parser.add_argument(
        "--with-calibration-rows",
        dest="no_calibration_rows",
        action="store_false",
        help="Append the legacy blank calibration rows for compatibility workflows.",
    )
    parser.add_argument("--legacy-root-copy", action="store_true")
    parser.add_argument("--dominance-log", type=str, default=None)
    parser.set_defaults(no_calibration_rows=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from backend import build_csv_datasets_v3 as dataset_builder
    config = DatasetBuildConfig(
        start_season=args.start,
        end_season=args.end,
        out_dir=args.out_dir,
        encode=args.encode,
        save_dominance_matrix=args.save_dominance_matrix,
        no_calibration_rows=args.no_calibration_rows,
        legacy_root_copy=args.legacy_root_copy,
        dominance_log=args.dominance_log,
    )

    out_root = Path(config.out_dir).resolve()
    run_id = _make_run_id()
    run_dir = out_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_builder.setup_logger(run_dir)
    dataset_builder.logging.info("Canonical builddataset run started | run_id=%s", run_id)
    dataset_builder.logging.info(
        "Dataset build config | seasons=%s-%s | out_dir=%s | encode=%s | include_future=%s | no_calibration_rows=%s",
        config.start_season,
        config.end_season,
        out_root,
        config.encode,
        config.include_future,
        config.no_calibration_rows,
    )

    dataset_builder.build_dataset(
        start_season=config.start_season,
        end_season=config.end_season,
        out_dir=run_dir,
        legacy_root_copy=config.legacy_root_copy,
        production_mode=True,
        include_future=config.include_future,
        encode=config.encode,
        save_dominance_matrix=config.save_dominance_matrix,
        no_calibration_rows=config.no_calibration_rows,
        dominance_log=config.dominance_log,
    )

    raw_dataset_path = _pick_latest_dataset_file(run_dir)
    raw_df = pd.read_csv(raw_dataset_path)
    clean_df, clean_stats = _clean_dataset(raw_df)

    promoted_clean_path = out_root / f"{raw_dataset_path.stem}_clean.csv"
    promoted_clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(promoted_clean_path, index=False)

    # Keep a copy inside the run directory for full run provenance.
    run_clean_path = run_dir / f"{raw_dataset_path.stem}_clean.csv"
    if run_clean_path != promoted_clean_path:
        shutil.copy2(promoted_clean_path, run_clean_path)

    metadata_path = run_dir / "game_features_metadata.json"
    quality_report_path = run_dir / "game_features_quality_report.json"
    score_snapshot_path = run_dir / "game_scores.json"
    log_path = run_dir / "build_csv_datasets.log"
    manifest_base_dir = out_root.parent
    score_entries = extract_score_entries_from_dataframe(clean_df, updated_at=_utc_now().isoformat())
    write_score_snapshot(score_snapshot_path, score_entries)
    write_score_snapshot(out_root / "latest_scores.json", score_entries)

    manifest = DatasetArtifactManifest(
        run_id=run_id,
        generated_at_utc=_utc_now().isoformat(),
        start_season=config.start_season,
        end_season=config.end_season,
        rows=int(len(clean_df)),
        columns=int(len(clean_df.columns)),
        completed_rows=clean_stats["completed_rows"],
        future_rows=clean_stats["future_rows"],
        blank_rows_removed=clean_stats["blank_rows_removed"],
        duplicate_game_ids_removed=clean_stats["duplicate_game_ids_removed"],
        include_future=config.include_future,
        encode=config.encode,
        no_calibration_rows=config.no_calibration_rows,
        legacy_root_copy=config.legacy_root_copy,
        raw_dataset_path=_manifest_path(raw_dataset_path, manifest_base_dir),
        clean_dataset_path=_manifest_path(promoted_clean_path, manifest_base_dir),
        run_dir=_manifest_path(run_dir, manifest_base_dir),
        metadata_path=_manifest_path(metadata_path, manifest_base_dir) if metadata_path.exists() else None,
        quality_report_path=(
            _manifest_path(quality_report_path, manifest_base_dir)
            if quality_report_path.exists()
            else None
        ),
        score_snapshot_path=_manifest_path(score_snapshot_path, manifest_base_dir),
        log_path=_manifest_path(log_path, manifest_base_dir) if log_path.exists() else None,
        dataset_hash=_sha256_file(promoted_clean_path),
        cleaning_stats={key: int(value) for key, value in clean_stats.items()},
    )

    _write_json(run_dir / "dataset_manifest.json", manifest.model_dump(mode="json"))
    _write_json(out_root / "latest_dataset.json", manifest.model_dump(mode="json"))

    dataset_builder.logging.info(
        "Clean dataset stats | rows_before=%d | rows_after=%d | blanks_removed=%d | duplicate_game_ids_removed=%d | completed_rows=%d | future_rows=%d",
        clean_stats["rows_before_cleaning"],
        clean_stats["rows_after_cleaning"],
        clean_stats["blank_rows_removed"],
        clean_stats["duplicate_game_ids_removed"],
        clean_stats["completed_rows"],
        clean_stats["future_rows"],
    )
    dataset_builder.logging.info("Promoted clean dataset to %s", promoted_clean_path)
    dataset_builder.logging.info("Wrote dataset manifest to %s", run_dir / "dataset_manifest.json")
    dataset_builder.logging.info("Wrote %d completed game scores to %s", len(score_entries), score_snapshot_path)


if __name__ == "__main__":
    main()
