from __future__ import annotations

"""
    Typed contracts for pipeline runs and prediction persistence.

    These models sit outside the HTTP schemas so the data pipeline, training
    artifacts, and local prediction ledger can share one validated vocabulary.
"""


from typing import Any, Optional
from pydantic import BaseModel, Field



class PredictionUserContext(BaseModel):

    """Normalized user identity used for local prediction storage."""


    user_id: str = Field(..., min_length=1, description="Logical user identifier.")

    storage_key: str = Field(..., min_length=1, description="Filesystem-safe storage key.")



class DatasetBuildConfig(BaseModel):

    """Validated settings for the canonical dataset builder entrypoint."""


    start_season: int = Field(..., ge=1999)

    end_season: int = Field(..., ge=1999)

    out_dir: str = Field(..., min_length=1)

    encode: str = Field(default="onehot", pattern="^(onehot|none)$")

    include_future: bool = True

    save_dominance_matrix: bool = False

    no_calibration_rows: bool = True

    legacy_root_copy: bool = False

    dominance_log: Optional[str] = None



class DatasetArtifactManifest(BaseModel):

    """Summary of a dataset build run and where the artifacts were written."""

    run_id: str
    generated_at_utc: str
    start_season: int
    end_season: int

    rows: int
    columns: int

    completed_rows: int

    future_rows: int

    blank_rows_removed: int = 0

    duplicate_game_ids_removed: int = 0

    include_future: bool = True
    encode: str = "onehot"

    no_calibration_rows: bool = True

    legacy_root_copy: bool = False

    raw_dataset_path: str
    clean_dataset_path: str
    completed_dataset_path: Optional[str] = None
    future_dataset_path: Optional[str] = None
    run_dir: str

    metadata_path: Optional[str] = None

    quality_report_path: Optional[str] = None
    schema_report_path: Optional[str] = None
    missingness_report_path: Optional[str] = None
    duplicate_report_path: Optional[str] = None
    training_readiness_report_path: Optional[str] = None

    score_snapshot_path: Optional[str] = None

    log_path: Optional[str] = None

    dataset_hash: Optional[str] = None
    training_ready: Optional[bool] = None
    training_blockers: list[str] = Field(default_factory=list)

    cleaning_stats: dict[str, Any] = Field(

        default_factory=dict,

        description="Deterministic cleanup counts for blank rows, duplicates, and label availability.",

    )



class TrainingRunConfig(BaseModel):

    """Validated configuration for model training runs."""


    data_path: str = Field(..., min_length=1)

    out_dir: str = Field(..., min_length=1)

    near_empty_threshold: float = Field(default=0.95, ge=0.0, le=1.0)

    complete_missing_max: float = Field(default=0.20, ge=0.0, le=1.0)

    future_missing_min: float = Field(default=0.95, ge=0.0, le=1.0)

    numeric_object_parse_rate: float = Field(default=0.98, ge=0.0, le=1.0)

    walk_start_calib: int = Field(default=2019, ge=1999)

    walk_end_calib: int = Field(default=2024, ge=1999)

    bootstrap_samples: int = Field(default=1500, ge=100)

    threshold: float = Field(default=0.54, ge=0.0, le=1.0)

    train_end_season: int = Field(default=2025, ge=1999)

    train_end_week: int = Field(default=17, ge=0)

    test_season: int = Field(default=2025, ge=1999)

    force_retrain: bool = Field(

        default=False,

        description="When true, ignore the monthly in-season freshness check and retrain immediately.",

    )



class TrainingScheduleManifest(BaseModel):

    """Human-readable retraining cadence for operators."""

    cadence: str

    in_season: bool
    last_trained_at_utc: str

    next_recommended_training_at_utc: Optional[str] = None



class TrainingArtifactManifest(BaseModel):

    """Summary of a training run and the active/archive artifact layout."""

    run_id: str
    trained_at_utc: str
    dataset_path: str

    deployed_models_dir: str

    archived_run_dir: Optional[str] = None
    training_report_path: str
    metadata_path: str
    schedule_path: str

    log_path: Optional[str] = None

    predictions_future_path: Optional[str] = None

    walkforward_predictions_path: Optional[str] = None

    schedule: TrainingScheduleManifest



class TrainingExecutionResult(BaseModel):

    """Structured outcome for one attempted training command."""


    trained: bool

    skipped: bool = False
    reason: str

    run_id: Optional[str] = None

    dataset_path: Optional[str] = None

    out_dir: Optional[str] = None

    latest_manifest_path: Optional[str] = None

    archived_run_dir: Optional[str] = None

    next_recommended_training_at_utc: Optional[str] = None



class PredictionStorageProfile(BaseModel):

    """User-scoped prediction ledger metadata stored alongside prediction history."""


    user_id: str = Field(..., min_length=1)

    storage_key: str = Field(..., min_length=1)
    updated_at_utc: str

    retained_predictions: int = Field(default=0, ge=0)

    total_predictions_all_time: int = Field(default=0, ge=0)

