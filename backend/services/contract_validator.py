from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from backend.contracts.feature_contract import (
    FeatureValidationResult,
    align_feature_frame,
    build_feature_contract,
    validate_feature_frame,
)
from backend.contracts.model_bundle_contract import (
    ModelBundleContractResult,
    validate_model_bundle_contract,
)


class RuntimeContractValidation(BaseModel):
    """Combined runtime contract validation for dataset and model bundle.

    Data shape:
        Top-level readiness plus bundle validation and per-model feature-frame
        validation for the ``win`` and ``score`` model families.
    Methods:
        Pydantic validation only.
    """

    ok: bool
    blockers: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    bundle: ModelBundleContractResult
    dataset_features: Dict[str, FeatureValidationResult] = Field(default_factory=dict)


def validate_runtime_contracts(
    *,
    models_dir: Path,
    metadata: Dict[str, Any],
    dataset: Optional[pd.DataFrame],
    dataset_hash: Optional[str],
    sklearn_runtime_version: Optional[str],
) -> RuntimeContractValidation:
    bundle = validate_model_bundle_contract(
        models_dir=models_dir,
        metadata=metadata,
        dataset_hash=dataset_hash,
        sklearn_runtime_version=sklearn_runtime_version,
    )
    blockers = list(bundle.blockers)
    warnings = list(bundle.warnings)
    dataset_features: Dict[str, FeatureValidationResult] = {}

    if dataset is None or dataset.empty:
        blockers.append("cannot validate feature contract because dataset is empty")
    else:
        for model_key in ("win", "score"):
            contract = build_feature_contract(metadata, model_key)
            projected_cols = [
                col for col in contract.expected_features
                if col in dataset.columns
            ]
            validation_frame = dataset.loc[:, projected_cols].copy()
            result = validate_feature_frame(
                validation_frame,
                contract,
                allow_unexpected_columns=True,
                allow_generated_missing=True,
            )
            dataset_features[model_key] = result
            blockers.extend(
                f"dataset {message}" for message in result.blockers
            )
            warnings.extend(
                f"dataset {message}" for message in result.warnings
            )

    return RuntimeContractValidation(
        ok=not blockers,
        blockers=sorted(set(blockers)),
        warnings=sorted(set(warnings)),
        bundle=bundle,
        dataset_features=dataset_features,
    )


def validate_prediction_feature_frame(
    *,
    frame: pd.DataFrame,
    metadata: Dict[str, Any],
    model_key: str,
) -> FeatureValidationResult:
    contract = build_feature_contract(metadata, model_key)
    return validate_feature_frame(
        frame,
        contract,
        allow_unexpected_columns=True,
        allow_generated_missing=False,
    )


def align_prediction_feature_frame(
    *,
    frame: pd.DataFrame,
    metadata: Dict[str, Any],
    model_key: str,
) -> pd.DataFrame:
    contract = build_feature_contract(metadata, model_key)
    return align_feature_frame(frame, contract)
