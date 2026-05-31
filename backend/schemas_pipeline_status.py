from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FileStatus(BaseModel):
    path: Optional[str] = None
    exists: bool = False
    size_bytes: Optional[int] = None
    modified_at_utc: Optional[str] = None


class FeatureGroupStatus(BaseModel):
    group: str
    columns: int = 0
    coverage: Optional[float] = None
    null_rate: Optional[float] = None
    sample_columns: List[str] = Field(default_factory=list)


class DataSourceStatus(BaseModel):
    selected_nfl_backend: str = "unknown"
    nflreadpy_available: bool = False
    fallback_reason: Optional[str] = None
    pbp_cache: FileStatus = Field(default_factory=FileStatus)
    player_stats_available: bool = False
    schedule_row_count: int = 0
    future_game_support: bool = False
    degraded_reason_codes: List[str] = Field(default_factory=list)


class DatasetQualityStatus(BaseModel):
    dataset_path: Optional[str] = None
    dataset_hash: Optional[str] = None
    rows: int = 0
    columns: int = 0
    seasons_min: Optional[int] = None
    seasons_max: Optional[int] = None
    weeks_min: Optional[int] = None
    weeks_max: Optional[int] = None
    completed_rows: int = 0
    future_rows: int = 0
    stale: bool = False
    stale_reason: Optional[str] = None
    manifest: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    quality_report_path: Optional[str] = None
    feature_groups: List[FeatureGroupStatus] = Field(default_factory=list)
    critical_null_rates: Dict[str, float] = Field(default_factory=dict)


class ContractStatus(BaseModel):
    ok: bool = False
    blockers: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)


class ModelBundleStatus(BaseModel):
    models_dir: Optional[str] = None
    metadata_path: Optional[str] = None
    loaded_models: List[str] = Field(default_factory=list)
    missing_required: List[str] = Field(default_factory=list)
    load_errors: Dict[str, str] = Field(default_factory=dict)
    artifacts: Dict[str, FileStatus] = Field(default_factory=dict)
    contract: ContractStatus = Field(default_factory=ContractStatus)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class PipelineStatusResponse(BaseModel):
    generated_at: str
    production_ready: bool = False
    warning_only: bool = False
    degraded: bool = False
    blockers: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    data_sources: DataSourceStatus
    dataset: DatasetQualityStatus
    model_bundle: ModelBundleStatus
