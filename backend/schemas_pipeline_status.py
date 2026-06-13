from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FileStatus(BaseModel):
    """Filesystem status for one expected file.

    Data shape: path, existence flag, size, and modified timestamp.
    Methods: Pydantic validation only.
    """

    path: Optional[str] = None
    exists: bool = False
    size_bytes: Optional[int] = None
    modified_at_utc: Optional[str] = None


class FeatureGroupStatus(BaseModel):
    """Coverage summary for one feature group.

    Data shape: group name, column count, coverage/null rates, and sample
    columns.
    Methods: Pydantic validation only.
    """

    group: str
    columns: int = 0
    coverage: Optional[float] = None
    null_rate: Optional[float] = None
    sample_columns: List[str] = Field(default_factory=list)


class DataSourceStatus(BaseModel):
    """Runtime data-source availability summary.

    Data shape: selected NFL data backend, cache status, schedule coverage,
    and degradation reasons.
    Methods: Pydantic validation only.
    """

    selected_nfl_backend: str = "unknown"
    nflreadpy_available: bool = False
    fallback_reason: Optional[str] = None
    pbp_cache: FileStatus = Field(default_factory=FileStatus)
    player_stats_available: bool = False
    schedule_row_count: int = 0
    future_game_support: bool = False
    degraded_reason_codes: List[str] = Field(default_factory=list)


class DatasetQualityStatus(BaseModel):
    """Dataset quality and freshness summary.

    Data shape: dataset identity, size, season/week ranges, future/completed
    row counts, manifest/metadata, feature groups, and null rates.
    Methods: Pydantic validation only.
    """

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
    """Generic contract validation status.

    Data shape: readiness flag, blockers, warnings, and summary details.
    Methods: Pydantic validation only.
    """

    ok: bool = False
    blockers: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)


class ModelBundleStatus(BaseModel):
    """Model-bundle readiness summary.

    Data shape: model directory, metadata path, loaded/missing models, load
    errors, artifact statuses, contract status, and provenance.
    Methods: Pydantic validation only.
    """

    models_dir: Optional[str] = None
    metadata_path: Optional[str] = None
    loaded_models: List[str] = Field(default_factory=list)
    missing_required: List[str] = Field(default_factory=list)
    load_errors: Dict[str, str] = Field(default_factory=dict)
    artifacts: Dict[str, FileStatus] = Field(default_factory=dict)
    contract: ContractStatus = Field(default_factory=ContractStatus)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class PipelineStatusResponse(BaseModel):
    """Full pipeline status response.

    Data shape: generated timestamp, readiness/degradation flags, blockers,
    warnings, data-source status, dataset quality, and model-bundle status.
    Methods: Pydantic validation only.
    """

    generated_at: str
    production_ready: bool = False
    warning_only: bool = False
    degraded: bool = False
    blockers: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    data_sources: DataSourceStatus
    dataset: DatasetQualityStatus
    model_bundle: ModelBundleStatus
