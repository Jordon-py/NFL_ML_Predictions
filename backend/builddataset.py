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


TEAM_ABBR_ALIASES: dict[str, str] = {
    "LA": "LAR",
    "STL": "LAR",
    "SD": "LAC",
    "OAK": "LV",
    "WSH": "WAS",
}

REQUIRED_SCHEMA_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "home_points_for",
    "away_points_for",
    "home_win",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _make_run_id() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _pick_latest_dataset_file(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("game_features_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No dataset artifacts found in {run_dir}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _canonical_team_code(value: Any) -> str:
    if pd.isna(value):
        return ""
    code = str(value).strip().upper()
    if not code or code in {"NAN", "NONE", "NULL"}:
        return ""
    return TEAM_ABBR_ALIASES.get(code, code)


def _canonical_game_id(season: Any, week: Any, away_team: Any, home_team: Any) -> str:
    season_value = pd.to_numeric(pd.Series([season]), errors="coerce").iloc[0]
    week_value = pd.to_numeric(pd.Series([week]), errors="coerce").iloc[0]
    away = _canonical_team_code(away_team)
    home = _canonical_team_code(home_team)
    if pd.isna(season_value) or pd.isna(week_value) or not away or not home:
        return ""
    return f"{int(season_value)}_{int(week_value):02d}_{away}_{home}"


def _ensure_game_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = {"season", "week", "home_team", "away_team"}
    if not required.issubset(out.columns):
        if "game_id" in out.columns:
            out["game_id"] = out["game_id"].fillna("").astype(str).str.strip()
        return out

    out["home_team"] = out["home_team"].apply(_canonical_team_code)
    out["away_team"] = out["away_team"].apply(_canonical_team_code)
    out["game_id"] = [
        _canonical_game_id(season, week, away, home)
        for season, week, away, home in zip(
            out["season"],
            out["week"],
            out["away_team"],
            out["home_team"],
        )
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
    duplicate_game_ids: list[str] = []
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
            duplicate_game_ids = (
                out.loc[valid_game_ids, "game_id"]
                .value_counts()
                .loc[lambda counts: counts > 1]
                .head(100)
                .index.astype(str)
                .tolist()
            )
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
        "duplicate_game_ids_sample": duplicate_game_ids,
        "dropped_empty_or_constant_columns": len(cols_to_drop),
        "dropped_empty_or_constant_column_names": cols_to_drop,
        "completed_rows": completed_rows,
        "future_rows": future_rows,
    }
    return out, stats


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _completed_mask(df: pd.DataFrame) -> pd.Series:
    if "home_win" in df.columns:
        return df["home_win"].notna()
    score_cols = [col for col in ("home_points_for", "away_points_for") if col in df.columns]
    if score_cols:
        return df[score_cols].notna().all(axis=1)
    return pd.Series([True] * len(df), index=df.index)


def _schema_report(df: pd.DataFrame) -> dict[str, Any]:
    missing_required = [col for col in REQUIRED_SCHEMA_COLUMNS if col not in df.columns]
    columns: list[dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        columns.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_ratio": float(series.isna().mean()) if len(series) else 0.0,
                "unique_count": int(series.nunique(dropna=True)),
            }
        )
    return {
        "required_columns": list(REQUIRED_SCHEMA_COLUMNS),
        "missing_required_columns": missing_required,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_report": columns,
    }


def _missingness_report(df: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        rows.append(
            {
                "column": str(col),
                "missing_count": missing_count,
                "missing_ratio": float(missing_count / max(1, len(df))),
            }
        )
    rows = sorted(rows, key=lambda item: (-item["missing_ratio"], item["column"]))
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "columns_with_missing_values": int(sum(1 for row in rows if row["missing_count"] > 0)),
        "missingness": rows,
    }


def _duplicate_report(df: pd.DataFrame, clean_stats: dict[str, Any]) -> dict[str, Any]:
    duplicate_rows: list[dict[str, Any]] = []
    if "game_id" in df.columns:
        counts = df["game_id"].fillna("").astype(str).str.strip().value_counts()
        duplicate_rows = [
            {"game_id": str(game_id), "count": int(count)}
            for game_id, count in counts[counts > 1].head(100).items()
        ]
    return {
        "duplicate_game_ids_removed": int(clean_stats.get("duplicate_game_ids_removed", 0) or 0),
        "duplicate_game_ids_sample_before_cleaning": clean_stats.get("duplicate_game_ids_sample", []),
        "remaining_duplicate_game_ids": duplicate_rows,
    }


def _write_dataset_partitions(
    *,
    clean_df: pd.DataFrame,
    out_root: Path,
    run_dir: Path,
    raw_stem: str,
) -> tuple[Path, Path]:
    completed = clean_df.loc[_completed_mask(clean_df)].copy()
    future = clean_df.loc[~_completed_mask(clean_df)].copy()

    completed_path = out_root / f"{raw_stem}_completed.csv"
    future_path = out_root / f"{raw_stem}_future.csv"
    completed.to_csv(completed_path, index=False)
    future.to_csv(future_path, index=False)

    shutil.copy2(completed_path, run_dir / completed_path.name)
    shutil.copy2(future_path, run_dir / future_path.name)
    return completed_path, future_path


def _sha256_file(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        data = path.read_bytes().replace(b"\r\n", b"\n")
        return hashlib.sha256(data).hexdigest()
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
    completed_dataset_path, future_dataset_path = _write_dataset_partitions(
        clean_df=clean_df,
        out_root=out_root,
        run_dir=run_dir,
        raw_stem=raw_dataset_path.stem,
    )

    # Keep a copy inside the run directory for full run provenance.
    run_clean_path = run_dir / f"{raw_dataset_path.stem}_clean.csv"
    if run_clean_path != promoted_clean_path:
        shutil.copy2(promoted_clean_path, run_clean_path)

    metadata_path = run_dir / "game_features_metadata.json"
    quality_report_path = run_dir / "game_features_quality_report.json"
    schema_report_path = run_dir / "schema_report.json"
    missingness_report_path = run_dir / "missingness_report.json"
    duplicate_report_path = run_dir / "duplicate_report.json"
    score_snapshot_path = run_dir / "game_scores.json"
    log_path = run_dir / "build_csv_datasets.log"
    manifest_base_dir = out_root.parent
    _write_json(schema_report_path, _schema_report(clean_df))
    _write_json(missingness_report_path, _missingness_report(clean_df))
    _write_json(duplicate_report_path, _duplicate_report(clean_df, clean_stats))

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
        completed_dataset_path=_manifest_path(completed_dataset_path, manifest_base_dir),
        future_dataset_path=_manifest_path(future_dataset_path, manifest_base_dir),
        run_dir=_manifest_path(run_dir, manifest_base_dir),
        metadata_path=_manifest_path(metadata_path, manifest_base_dir) if metadata_path.exists() else None,
        quality_report_path=(
            _manifest_path(quality_report_path, manifest_base_dir)
            if quality_report_path.exists()
            else None
        ),
        schema_report_path=_manifest_path(schema_report_path, manifest_base_dir),
        missingness_report_path=_manifest_path(missingness_report_path, manifest_base_dir),
        duplicate_report_path=_manifest_path(duplicate_report_path, manifest_base_dir),
        score_snapshot_path=_manifest_path(score_snapshot_path, manifest_base_dir),
        log_path=_manifest_path(log_path, manifest_base_dir) if log_path.exists() else None,
        dataset_hash=_sha256_file(promoted_clean_path),
        cleaning_stats=clean_stats,
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
