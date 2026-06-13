from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .feature_contract import build_feature_contract


class ArtifactContractStatus(BaseModel):
    """Filesystem status for one model-bundle artifact.

    Data shape:
        ``key`` is the semantic artifact name, ``path`` is the resolved path,
        and ``exists`` reflects current filesystem availability.
    Methods:
        Pydantic validation only.
    """

    key: str
    path: str
    exists: bool


class ModelBundleContractResult(BaseModel):
    """Model-bundle validation summary.

    Data shape:
        Bundle readiness, strict/legacy mode, blockers, warnings, artifact
        statuses, dataset-hash match, calibration presence, and feature
        contract summaries.
    Methods:
        Pydantic validation only.
    """

    ok: bool
    strict: bool = False
    blockers: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    artifacts: Dict[str, ArtifactContractStatus] = Field(default_factory=dict)
    dataset_hash_match: Optional[bool] = None
    calibration_metadata_present: bool = False
    feature_contracts: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


def _bundle_timestamp(metadata: Dict[str, Any]) -> Optional[str]:
    for key in ("bundle_timestamp_utc", "timestamp", "training_timestamp_utc"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _strict_bundle(metadata: Dict[str, Any]) -> bool:
    return bool(
        metadata.get("serving_mode") == "pipeline_primary"
        or metadata.get("bundle_contract_version")
    )


def _artifact_entries(models_dir: Path, metadata: Dict[str, Any]) -> Dict[str, ArtifactContractStatus]:
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    default_artifacts = {
        "reg_home": "home_pipe.joblib",
        "reg_away": "away_pipe.joblib",
        "clf_home_win": "win_pipe.joblib",
        "score_preprocessor": "score_preprocessor.joblib",
        "win_preprocessor": "win_preprocessor.joblib",
        "metadata": "metadata.json",
    }
    merged = {**default_artifacts, **{str(k): str(v) for k, v in artifacts.items()}}
    out: Dict[str, ArtifactContractStatus] = {}
    for key, rel_path in merged.items():
        path = (models_dir / rel_path).resolve()
        out[key] = ArtifactContractStatus(key=key, path=str(path), exists=path.exists())
    return out


def validate_model_bundle_contract(
    *,
    models_dir: Path,
    metadata: Dict[str, Any],
    dataset_hash: Optional[str] = None,
    sklearn_runtime_version: Optional[str] = None,
) -> ModelBundleContractResult:
    metadata = metadata if isinstance(metadata, dict) else {}
    blockers: List[str] = []
    warnings: List[str] = []
    strict = _strict_bundle(metadata)

    if not metadata:
        blockers.append("model metadata missing or invalid")
    elif strict:
        required = [
            "serving_mode",
            "feature_manifests",
            "generated_features",
            "dataset_hash",
            "sklearn_version",
        ]
        missing = [key for key in required if not metadata.get(key)]
        if not _bundle_timestamp(metadata):
            missing.append("bundle_timestamp_utc")
        if missing:
            blockers.append(
                "model bundle metadata missing required contract field(s): "
                + ", ".join(sorted(set(missing)))
            )
    else:
        warnings.append("legacy model bundle contract")

    declared_sklearn = str(metadata.get("sklearn_version") or "").strip()
    if sklearn_runtime_version and declared_sklearn and sklearn_runtime_version != declared_sklearn:
        blockers.append(
            f"model bundle requires scikit-learn {declared_sklearn}; runtime has {sklearn_runtime_version}"
        )

    dataset_hash_match: Optional[bool] = None
    declared_dataset_hash = str(metadata.get("dataset_hash") or "").strip()
    if declared_dataset_hash and dataset_hash:
        dataset_hash_match = declared_dataset_hash == str(dataset_hash)
        if not dataset_hash_match:
            blockers.append("active dataset hash does not match model bundle training dataset hash")

    artifacts = _artifact_entries(models_dir, metadata)
    required_artifact_keys = ("reg_home", "reg_away", "clf_home_win")
    for key in required_artifact_keys:
        artifact = artifacts.get(key)
        if artifact is None or not artifact.exists:
            blockers.append(f"model bundle missing required artifact: {key}")
    for key in ("score_preprocessor", "win_preprocessor"):
        artifact = artifacts.get(key)
        if artifact is None or not artifact.exists:
            warnings.append(f"model bundle missing optional preprocessor artifact: {key}")

    feature_contracts: Dict[str, Dict[str, Any]] = {}
    for model_key in ("win", "score"):
        contract = build_feature_contract(metadata, model_key)
        feature_contracts[model_key] = {
            "expected_count": len(contract.expected_features),
            "numeric_count": len(contract.numeric_features),
            "categorical_count": len(contract.categorical_features),
            "generated_features": contract.generated_feature_names,
            "training_dataset_hash": contract.training_dataset_hash,
            "imputation_strategy": contract.imputation_strategy,
        }
        if strict and not contract.expected_features:
            blockers.append(f"{model_key} feature contract is empty")

    metrics = metadata.get("metrics")
    calibration = None
    if isinstance(metrics, dict):
        calibration = metrics.get("calibration")
        if calibration is None:
            win_metrics = metrics.get("win")
            calibration = win_metrics.get("calibration") if isinstance(win_metrics, dict) else None
        if calibration is None:
            classification_metrics = metrics.get("classification")
            calibration = (
                classification_metrics.get("calibration")
                if isinstance(classification_metrics, dict)
                else None
            )
    calibration_present = bool(calibration)
    if strict and not calibration_present:
        warnings.append("win calibration metadata missing from model bundle")

    return ModelBundleContractResult(
        ok=not blockers,
        strict=strict,
        blockers=sorted(set(blockers)),
        warnings=sorted(set(warnings)),
        artifacts=artifacts,
        dataset_hash_match=dataset_hash_match,
        calibration_metadata_present=calibration_present,
        feature_contracts=feature_contracts,
    )
