from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field


class FeatureContract(BaseModel):
    model_key: str
    expected_features: List[str] = Field(default_factory=list)
    numeric_features: List[str] = Field(default_factory=list)
    categorical_features: List[str] = Field(default_factory=list)
    generated_features: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    training_dataset_hash: Optional[str] = None
    imputation_strategy: str = "dataset_numeric_median"

    @property
    def generated_feature_names(self) -> List[str]:
        return [str(name) for name in self.generated_features.keys()]


class FeatureValidationResult(BaseModel):
    ok: bool
    model_key: str
    expected_count: int = 0
    observed_count: int = 0
    missing_columns: List[str] = Field(default_factory=list)
    unexpected_columns: List[str] = Field(default_factory=list)
    order_mismatch: bool = False
    non_numeric_columns: List[str] = Field(default_factory=list)
    coerced_numeric_columns: List[str] = Field(default_factory=list)
    null_counts: Dict[str, int] = Field(default_factory=dict)
    blockers: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


def _feature_parts(metadata: Dict[str, Any], model_key: str) -> tuple[List[str], List[str]]:
    aliases = [model_key]
    if model_key == "score":
        aliases.append("scores")
    if model_key == "scores":
        aliases.extend(["score"])

    for container_name in ("feature_manifests", "raw_feature_columns"):
        container = metadata.get(container_name)
        if not isinstance(container, dict):
            continue
        for alias in aliases:
            selected = container.get(alias)
            if isinstance(selected, dict):
                numeric = [str(x) for x in selected.get("numeric", [])]
                categorical = [str(x) for x in selected.get("categorical", [])]
                return numeric, categorical

    if model_key in {"score", "scores"} and isinstance(metadata.get("feature_names"), list):
        return [str(x) for x in metadata["feature_names"]], []
    if model_key == "win" and isinstance(metadata.get("feature_names_win"), list):
        return [str(x) for x in metadata["feature_names_win"]], []
    return [], []


def build_feature_contract(metadata: Dict[str, Any], model_key: str) -> FeatureContract:
    metadata = metadata if isinstance(metadata, dict) else {}
    normalized = "score" if model_key == "scores" else str(model_key)
    numeric, categorical = _feature_parts(metadata, normalized)
    generated = metadata.get("generated_features")
    if not isinstance(generated, dict):
        generated = {}

    return FeatureContract(
        model_key=normalized,
        expected_features=[*numeric, *categorical],
        numeric_features=numeric,
        categorical_features=categorical,
        generated_features={
            str(key): value if isinstance(value, dict) else {"source": str(value)}
            for key, value in generated.items()
        },
        training_dataset_hash=metadata.get("dataset_hash"),
        imputation_strategy=str(metadata.get("imputation_strategy") or "dataset_numeric_median"),
    )


def align_feature_frame(frame: pd.DataFrame, contract: FeatureContract) -> pd.DataFrame:
    if frame is None:
        frame = pd.DataFrame()
    if not contract.expected_features:
        return frame.copy()
    return frame.reindex(columns=contract.expected_features).copy()


def _coercible_numeric(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return True
    coerced = pd.to_numeric(non_null, errors="coerce")
    return bool(coerced.notna().all())


def validate_feature_frame(
    frame: pd.DataFrame,
    contract: FeatureContract,
    *,
    allow_unexpected_columns: bool = True,
    allow_generated_missing: bool = False,
) -> FeatureValidationResult:
    if frame is None:
        frame = pd.DataFrame()

    observed = [str(col) for col in frame.columns]
    expected = list(contract.expected_features)
    expected_set = set(expected)
    generated = set(contract.generated_feature_names)

    missing = [col for col in expected if col not in observed]
    if allow_generated_missing:
        missing = [col for col in missing if col not in generated]

    unexpected = [col for col in observed if col not in expected_set]
    projected_order = [col for col in observed if col in expected_set]
    expected_observed_order = [col for col in expected if col in observed]
    order_mismatch = bool(projected_order and projected_order != expected_observed_order)

    blockers: List[str] = []
    warnings: List[str] = []
    if missing:
        blockers.append(
            f"{contract.model_key} feature contract missing {len(missing)} expected column(s)"
        )
    if unexpected and not allow_unexpected_columns:
        blockers.append(
            f"{contract.model_key} feature contract has {len(unexpected)} unexpected column(s)"
        )
    elif unexpected:
        warnings.append(
            f"{contract.model_key} feature frame has {len(unexpected)} unexpected column(s)"
        )
    if order_mismatch:
        warnings.append(f"{contract.model_key} feature order differs from training contract")

    non_numeric: List[str] = []
    coerced_numeric: List[str] = []
    for col in contract.numeric_features:
        if col not in frame.columns:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            continue
        if _coercible_numeric(frame[col]):
            coerced_numeric.append(col)
        else:
            non_numeric.append(col)
    if non_numeric:
        blockers.append(
            f"{contract.model_key} feature contract has non-numeric values in numeric column(s)"
        )
    if coerced_numeric:
        warnings.append(
            f"{contract.model_key} feature frame has {len(coerced_numeric)} numeric column(s) requiring coercion"
        )

    null_counts: Dict[str, int] = {}
    for col in expected:
        if col in frame.columns:
            missing_count = int(frame[col].isna().sum())
            if missing_count:
                null_counts[col] = missing_count
    if null_counts:
        warnings.append(
            f"{contract.model_key} feature frame has nulls in {len(null_counts)} expected column(s); "
            f"strategy={contract.imputation_strategy}"
        )

    return FeatureValidationResult(
        ok=not blockers,
        model_key=contract.model_key,
        expected_count=len(expected),
        observed_count=len(observed),
        missing_columns=missing,
        unexpected_columns=unexpected,
        order_mismatch=order_mismatch,
        non_numeric_columns=non_numeric,
        coerced_numeric_columns=coerced_numeric,
        null_counts=null_counts,
        blockers=blockers,
        warnings=warnings,
    )
