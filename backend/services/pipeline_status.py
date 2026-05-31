from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from backend.schemas_pipeline_status import (
    ContractStatus,
    DataSourceStatus,
    DatasetQualityStatus,
    FeatureGroupStatus,
    FileStatus,
    ModelBundleStatus,
    PipelineStatusResponse,
)


CURRENT_NFL_SEASON = 2026


def _file_status(path: Optional[Path]) -> FileStatus:
    if path is None:
        return FileStatus()
    try:
        exists = path.exists()
        stat = path.stat() if exists else None
        modified = (
            datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            if stat is not None
            else None
        )
        return FileStatus(
            path=str(path),
            exists=exists,
            size_bytes=int(stat.st_size) if stat is not None else None,
            modified_at_utc=modified,
        )
    except Exception:
        return FileStatus(path=str(path), exists=False)


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _numeric_range(df: pd.DataFrame, column: str) -> tuple[Optional[int], Optional[int]]:
    if df is None or df.empty or column not in df.columns:
        return None, None
    vals = pd.to_numeric(df[column], errors="coerce").dropna()
    if vals.empty:
        return None, None
    return int(vals.min()), int(vals.max())


def _future_rows(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    score_pairs = [
        ("home_points_for", "away_points_for"),
        ("home_score", "away_score"),
        ("home_points", "away_points"),
    ]
    for home_col, away_col in score_pairs:
        if {home_col, away_col}.issubset(df.columns):
            mask = df[home_col].isna() | df[away_col].isna()
            return int(mask.sum())
    return 0


def _feature_groups(df: pd.DataFrame) -> List[FeatureGroupStatus]:
    groups = {
        "market": ["moneyline", "spread_line", "total_line"],
        "prior_team_form": ["prior_", "rolling_"],
        "pbp_efficiency": ["epa", "success_rate", "explosive_rate", "turnover_rate", "takeaway_rate"],
        "weather_rest": ["temp", "wind", "rest", "roof", "surface"],
        "elo_qb": ["elo", "qb_"],
        "team_encoding": ["home_team_", "away_team_"],
    }
    out: List[FeatureGroupStatus] = []
    if df is None or df.empty:
        return out

    for group, needles in groups.items():
        cols = [
            str(col)
            for col in df.columns
            if any(str(col).startswith(needle) or needle in str(col) for needle in needles)
        ]
        if not cols:
            out.append(FeatureGroupStatus(group=group))
            continue
        null_rate = float(df[cols].isna().mean().mean())
        out.append(
            FeatureGroupStatus(
                group=group,
                columns=len(cols),
                coverage=float(1.0 - null_rate),
                null_rate=null_rate,
                sample_columns=cols[:8],
            )
        )
    return out


def _critical_null_rates(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {}
    critical = [
        "home_team",
        "away_team",
        "season",
        "week",
        "spread_line",
        "total_line",
        "home_rest",
        "away_rest",
        "home_moneyline_prob",
        "away_moneyline_prob",
    ]
    out: Dict[str, float] = {}
    for col in critical:
        if col in df.columns:
            out[col] = float(df[col].isna().mean())
    return out


def build_dataset_quality_status(
    *,
    dataset: Optional[pd.DataFrame],
    dataset_path: Optional[Path],
    dataset_hash: Optional[str],
    manifest: Dict[str, Any],
    metadata: Dict[str, Any],
) -> DatasetQualityStatus:
    df = dataset if dataset is not None else pd.DataFrame()
    seasons_min, seasons_max = _numeric_range(df, "season")
    weeks_min, weeks_max = _numeric_range(df, "week")
    future_count = _future_rows(df)
    completed_rows = max(0, int(len(df)) - future_count)

    stale = False
    stale_reason: Optional[str] = None
    if not df.empty and seasons_max is not None and seasons_max < CURRENT_NFL_SEASON:
        stale = True
        stale_reason = f"dataset max season {seasons_max} is older than {CURRENT_NFL_SEASON}"
    if df.empty:
        stale = True
        stale_reason = "dataset not loaded"

    quality_report_path = manifest.get("quality_report_path") if isinstance(manifest, dict) else None
    return DatasetQualityStatus(
        dataset_path=str(dataset_path) if dataset_path else None,
        dataset_hash=dataset_hash,
        rows=int(len(df)),
        columns=int(df.shape[1]) if not df.empty else 0,
        seasons_min=seasons_min,
        seasons_max=seasons_max,
        weeks_min=weeks_min,
        weeks_max=weeks_max,
        completed_rows=completed_rows,
        future_rows=future_count,
        stale=stale,
        stale_reason=stale_reason,
        manifest=manifest if isinstance(manifest, dict) else {},
        metadata=metadata if isinstance(metadata, dict) else {},
        quality_report_path=str(quality_report_path) if quality_report_path else None,
        feature_groups=_feature_groups(df),
        critical_null_rates=_critical_null_rates(df),
    )


def build_data_source_status(
    *,
    backend_dir: Path,
    dataset: Optional[pd.DataFrame],
    schedule_df: Optional[pd.DataFrame],
    schedule_error: Optional[str] = None,
) -> DataSourceStatus:
    pbp_path = _first_existing(
        [
            backend_dir / "pbp_cache.csv",
            backend_dir / "data" / "pbp_cache.csv",
            backend_dir.parent / "pbp_cache.csv",
            backend_dir.parent / "data" / "pbp_cache.csv",
        ]
    )
    df = dataset if dataset is not None else pd.DataFrame()
    player_cols = [
        col for col in df.columns
        if any(token in str(col).lower() for token in ("qb_", "receiver", "rusher", "player"))
    ]
    degraded: List[str] = []
    if not _file_status(pbp_path).exists:
        degraded.append("PBP_CACHE_MISSING")
    if not player_cols:
        degraded.append("PLAYER_STATS_UNAVAILABLE")
    if schedule_error:
        degraded.append("SCHEDULE_LOAD_FAILED")
    if schedule_df is None or schedule_df.empty:
        degraded.append("SCHEDULE_EMPTY")
    if _future_rows(df) <= 0:
        degraded.append("NO_FUTURE_ROWS_IN_ACTIVE_DATASET")

    return DataSourceStatus(
        selected_nfl_backend="nflreadpy" if importlib.util.find_spec("nflreadpy") else "unavailable",
        nflreadpy_available=bool(importlib.util.find_spec("nflreadpy")),
        fallback_reason=schedule_error,
        pbp_cache=_file_status(pbp_path),
        player_stats_available=bool(player_cols),
        schedule_row_count=int(len(schedule_df)) if schedule_df is not None else 0,
        future_game_support=bool((_future_rows(df) > 0) or (schedule_df is not None and not schedule_df.empty)),
        degraded_reason_codes=sorted(set(degraded)),
    )


def build_model_bundle_status(
    *,
    models_dir: Path,
    metadata: Dict[str, Any],
    loaded_models: List[str],
    missing_required: List[str],
    load_errors: Dict[str, str],
    runtime_contract_validation: Dict[str, Any],
) -> ModelBundleStatus:
    metadata_path = models_dir / "metadata.json"
    artifacts_raw = metadata.get("artifacts") if isinstance(metadata, dict) else {}
    artifacts_raw = artifacts_raw if isinstance(artifacts_raw, dict) else {}
    artifact_names = {
        "metadata": "metadata.json",
        "home_pipe": "home_pipe.joblib",
        "away_pipe": "away_pipe.joblib",
        "win_pipe": "win_pipe.joblib",
        **{str(key): str(value) for key, value in artifacts_raw.items()},
    }
    artifacts = {
        key: _file_status(models_dir / rel_path)
        for key, rel_path in artifact_names.items()
    }
    contract = runtime_contract_validation or {}
    return ModelBundleStatus(
        models_dir=str(models_dir),
        metadata_path=str(metadata_path) if metadata_path.exists() else None,
        loaded_models=sorted(loaded_models),
        missing_required=sorted(missing_required),
        load_errors=load_errors,
        artifacts=artifacts,
        contract=ContractStatus(
            ok=bool(contract.get("ok", False)),
            blockers=list(contract.get("blockers") or []),
            warnings=list(contract.get("warnings") or []),
            summary={
                "bundle": contract.get("bundle", {}),
                "dataset_features": contract.get("dataset_features", {}),
            },
        ),
        provenance={
            "trained_at": metadata.get("timestamp") or metadata.get("training_timestamp_utc"),
            "bundle_version": metadata.get("bundle_version"),
            "dataset_hash": metadata.get("dataset_hash"),
            "sklearn_version": metadata.get("sklearn_version"),
            "training_script": metadata.get("training_script"),
        },
    )


def build_pipeline_status(
    *,
    backend_dir: Path,
    models_dir: Path,
    dataset: Optional[pd.DataFrame],
    dataset_path: Optional[Path],
    dataset_hash: Optional[str],
    dataset_manifest: Dict[str, Any],
    dataset_metadata: Dict[str, Any],
    models_metadata: Dict[str, Any],
    loaded_models: List[str],
    missing_required_models: List[str],
    model_load_errors: Dict[str, str],
    production_blockers: List[str],
    production_warnings: List[str],
    runtime_contract_validation: Dict[str, Any],
    schedule_df: Optional[pd.DataFrame] = None,
    schedule_error: Optional[str] = None,
) -> PipelineStatusResponse:
    generated_at = datetime.now(timezone.utc).isoformat()
    dataset_status = build_dataset_quality_status(
        dataset=dataset,
        dataset_path=dataset_path,
        dataset_hash=dataset_hash,
        manifest=dataset_manifest,
        metadata=dataset_metadata,
    )
    source_status = build_data_source_status(
        backend_dir=backend_dir,
        dataset=dataset,
        schedule_df=schedule_df,
        schedule_error=schedule_error,
    )
    model_status = build_model_bundle_status(
        models_dir=models_dir,
        metadata=models_metadata,
        loaded_models=loaded_models,
        missing_required=missing_required_models,
        load_errors=model_load_errors,
        runtime_contract_validation=runtime_contract_validation,
    )

    blockers = list(production_blockers)
    warnings = list(production_warnings)
    if dataset_status.stale and dataset_status.stale_reason:
        warnings.append(dataset_status.stale_reason)
    for code in source_status.degraded_reason_codes:
        warnings.append(f"pipeline degraded: {code}")

    blockers = sorted(set(blockers))
    warnings = sorted(set(warnings))
    production_ready = not blockers and not dataset_status.stale and not missing_required_models
    degraded = bool(source_status.degraded_reason_codes or dataset_status.stale or blockers)
    return PipelineStatusResponse(
        generated_at=generated_at,
        production_ready=production_ready,
        warning_only=bool(warnings and not blockers),
        degraded=degraded,
        blockers=blockers,
        warnings=warnings,
        data_sources=source_status,
        dataset=dataset_status,
        model_bundle=model_status,
    )
