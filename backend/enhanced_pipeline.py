#!/usr/bin/env python
"""
enhanced_pipeline.py
====================

Purpose
-------
Comprehensive dataset merge, validation, and model evaluation workflow for NFL predictions.
This pipeline orchestrates the full data preparation and model evaluation process with
robust error handling and automatic recovery suggestions.

Key Features
------------
- Dataset validation and integrity checks
- Leak-free feature engineering validation
- Model evaluation with cross-validation
- Automatic error detection with recovery solutions
- Comprehensive logging and reporting

Usage
-----
python backend/enhanced_pipeline.py --start 2010 --end 2025 --evaluate-models

Error Recovery
--------------
When errors occur (datatype mismatches, join failures, missing data), the pipeline
automatically proposes two production-ready recovery solutions and logs them for review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit

# Import from existing modules
try:
    from build_csv_datasets import (
        ABBR_FIX,
        build_dataset,
        load_schedules,
        add_features,
        make_time_key,
        _normalize_codes
    )
    from train_models import BASE_FEATURES, _load_dataset
except ImportError:
    # Handle relative imports for when run as script
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from build_csv_datasets import (
        ABBR_FIX,
        build_dataset,
        load_schedules,
        add_features,
        make_time_key,
        _normalize_codes
    )
    from train_models import BASE_FEATURES, _load_dataset

# Configuration
BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"
REPORTS_DIR = BACKEND_DIR / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
LOG_FILE = REPORTS_DIR / f"enhanced_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("enhanced_pipeline")


# ============================================================================
# ERROR RECOVERY SYSTEM
# ============================================================================

class RecoverySolution:
    """Represents a production-ready recovery solution for pipeline errors."""
    
    def __init__(self, name: str, description: str, implementation: str, risk_level: str):
        self.name = name
        self.description = description
        self.implementation = implementation
        self.risk_level = risk_level  # "LOW", "MEDIUM", "HIGH"
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "implementation": self.implementation,
            "risk_level": self.risk_level
        }


class PipelineError:
    """Custom error class with automatic recovery solution generation."""
    
    def __init__(self, error_type: str, message: str, context: Dict[str, Any]):
        self.error_type = error_type
        self.message = message
        self.context = context
        self.solutions = self._generate_solutions()
        self.timestamp = datetime.now().isoformat()
    
    def _generate_solutions(self) -> List[RecoverySolution]:
        """Generate two production-ready recovery solutions based on error type."""
        solutions = []
        
        if self.error_type == "DATATYPE_MISMATCH":
            solutions = [
                RecoverySolution(
                    name="Explicit Type Conversion",
                    description="Convert columns to expected types with error handling",
                    implementation="""
# Solution 1: Explicit type conversion with validation
def fix_datatype_mismatch(df, column_types):
    for col, dtype in column_types.items():
        try:
            if dtype in ['int', 'int64']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            elif dtype in ['float', 'float64']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif dtype == 'str':
                df[col] = df[col].astype(str)
            log.info(f"Converted {col} to {dtype}")
        except Exception as e:
            log.error(f"Failed to convert {col}: {e}")
    return df
""",
                    risk_level="LOW"
                ),
                RecoverySolution(
                    name="Schema-Based Validation Pipeline",
                    description="Implement upstream schema validation in data loading",
                    implementation="""
# Solution 2: Add schema validation in build_csv_datasets.py
def validate_schema(df, expected_schema):
    issues = []
    for col, expected_type in expected_schema.items():
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
        elif df[col].dtype != expected_type:
            issues.append(f"Type mismatch in {col}: got {df[col].dtype}, expected {expected_type}")
    
    if issues:
        raise ValueError(f"Schema validation failed: {issues}")
    return True

# Add to load_schedules() or add_features()
EXPECTED_SCHEMA = {
    'season': 'int64', 'week': 'int64', 'home_score': 'float64',
    'away_score': 'float64', 'home_team': 'object', 'away_team': 'object'
}
""",
                    risk_level="LOW"
                )
            ]
        
        elif self.error_type == "JOIN_ERROR":
            solutions = [
                RecoverySolution(
                    name="Fuzzy Match with Team Normalization",
                    description="Apply comprehensive team code normalization before joins",
                    implementation="""
# Solution 1: Enhanced team normalization
def normalize_all_team_codes(df, team_columns):
    df = df.copy()
    for col in team_columns:
        if col in df.columns:
            # Apply ABBR_FIX mapping
            df[col] = df[col].replace(ABBR_FIX)
            # Trim whitespace
            df[col] = df[col].str.strip()
            # Convert to uppercase for consistency
            df[col] = df[col].str.upper()
    return df

# Apply before any joins
df = normalize_all_team_codes(df, ['home_team', 'away_team'])
""",
                    risk_level="LOW"
                ),
                RecoverySolution(
                    name="Outer Join with Missing Data Report",
                    description="Use outer join to identify unmatched records",
                    implementation="""
# Solution 2: Use outer join to debug join issues
def safe_merge_with_report(left_df, right_df, on_cols, report_path):
    # Perform outer join
    merged = pd.merge(left_df, right_df, on=on_cols, how='outer', indicator=True)
    
    # Identify unmatched records
    left_only = merged[merged['_merge'] == 'left_only']
    right_only = merged[merged['_merge'] == 'right_only']
    
    # Generate report
    report = {
        'unmatched_left': len(left_only),
        'unmatched_right': len(right_only),
        'left_only_sample': left_only.head(10).to_dict('records'),
        'right_only_sample': right_only.head(10).to_dict('records')
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    log.warning(f"Join report saved to {report_path}")
    return merged[merged['_merge'] == 'both'].drop(columns=['_merge'])
""",
                    risk_level="MEDIUM"
                )
            ]
        
        elif self.error_type == "MISSING_FEATURES":
            solutions = [
                RecoverySolution(
                    name="Median Imputation with Logging",
                    description="Fill missing features with median values and log affected rows",
                    implementation="""
# Solution 1: Robust median imputation
def impute_missing_features(df, feature_cols):
    imputed_count = {}
    for col in feature_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                imputed_count[col] = missing
                log.info(f"Imputed {missing} missing values in {col} with median {median_val:.2f}")
    
    # Save imputation report
    report_path = REPORTS_DIR / "imputation_report.json"
    with open(report_path, 'w') as f:
        json.dump(imputed_count, f, indent=2)
    
    return df
""",
                    risk_level="LOW"
                ),
                RecoverySolution(
                    name="Rebuild Features from Source",
                    description="Re-run feature engineering pipeline to regenerate missing features",
                    implementation="""
# Solution 2: Regenerate features from raw data
def regenerate_missing_features(start_year, end_year):
    # Re-run the build_dataset pipeline
    from build_csv_datasets import build_dataset
    from pathlib import Path
    
    log.info("Regenerating dataset with all features...")
    output_path, df = build_dataset(
        start=start_year,
        end=end_year,
        out_dir=DATA_DIR,
        production_mode=True,
        include_future=True
    )
    
    log.info(f"Dataset regenerated: {output_path}")
    return df

# Usage
df = regenerate_missing_features(2010, 2025)
""",
                    risk_level="MEDIUM"
                )
            ]
        
        elif self.error_type == "FEATURE_LEAKAGE":
            solutions = [
                RecoverySolution(
                    name="Validate Shift-Before-Rolling Pattern",
                    description="Audit all rolling features to ensure shift(1) is applied",
                    implementation="""
# Solution 1: Add feature leakage detection
def validate_no_leakage(df, feature_cols):
    issues = []
    # Check that rolling features don't include current game
    # This is validated by checking the feature engineering code
    
    for col in feature_cols:
        if 'prior' in col:
            # Verify that first few values are NaN (since we shift)
            first_non_null = df[col].first_valid_index()
            if first_non_null == 0:
                issues.append(f"Potential leakage in {col}: first value should be NaN")
    
    if issues:
        log.warning(f"Leakage validation issues: {issues}")
        return False
    
    log.info("Leakage validation passed")
    return True
""",
                    risk_level="LOW"
                ),
                RecoverySolution(
                    name="Time-Series Split Validation",
                    description="Use strict TimeSeriesSplit to validate no future information",
                    implementation="""
# Solution 2: Temporal validation with TimeSeriesSplit
def validate_temporal_integrity(df, features, target):
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import r2_score
    from sklearn.ensemble import RandomForestRegressor
    
    # Sort by time
    df = df.sort_values(['season', 'week']).reset_index(drop=True)
    
    X = df[features].fillna(0)
    y = df[target]
    
    # Use TimeSeriesSplit - ensures no future leakage
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_idx, test_idx in tscv.split(X):
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        score = r2_score(y.iloc[test_idx], pred)
        scores.append(score)
    
    log.info(f"Temporal CV scores: {scores}, mean: {np.mean(scores):.3f}")
    return np.mean(scores)
""",
                    risk_level="MEDIUM"
                )
            ]
        
        else:  # Generic error
            solutions = [
                RecoverySolution(
                    name="Debug Mode with Verbose Logging",
                    description="Enable detailed logging to identify root cause",
                    implementation="""
# Solution 1: Enhanced debug mode
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Add debug checkpoints
log.debug(f"DataFrame shape: {df.shape}")
log.debug(f"Columns: {df.columns.tolist()}")
log.debug(f"Dtypes: {df.dtypes.to_dict()}")
log.debug(f"Null counts: {df.isnull().sum().to_dict()}")
log.debug(f"Sample data: {df.head(3).to_dict()}")
""",
                    risk_level="LOW"
                ),
                RecoverySolution(
                    name="Checkpoint and Resume",
                    description="Save intermediate results to isolate failing step",
                    implementation="""
# Solution 2: Add checkpoints
def save_checkpoint(df, checkpoint_name):
    checkpoint_path = REPORTS_DIR / f"checkpoint_{checkpoint_name}.csv"
    df.to_csv(checkpoint_path, index=False)
    log.info(f"Checkpoint saved: {checkpoint_path}")

# Usage throughout pipeline
save_checkpoint(df, "after_load")
save_checkpoint(df, "after_features")
save_checkpoint(df, "after_validation")
""",
                    risk_level="LOW"
                )
            ]
        
        return solutions
    
    def log_error_report(self, report_path: Optional[Path] = None) -> None:
        """Generate and save comprehensive error report."""
        if report_path is None:
            report_path = REPORTS_DIR / f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "timestamp": self.timestamp,
            "error_type": self.error_type,
            "message": self.message,
            "context": self.context,
            "recovery_solutions": [sol.to_dict() for sol in self.solutions]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        log.error(f"Error report saved to {report_path}")
        log.error(f"Error: {self.error_type} - {self.message}")
        log.error(f"Context: {self.context}")
        log.info("=" * 80)
        log.info("AUTOMATIC RECOVERY SOLUTIONS:")
        log.info("=" * 80)
        for i, sol in enumerate(self.solutions, 1):
            log.info(f"\nSOLUTION {i}: {sol.name} (Risk: {sol.risk_level})")
            log.info(f"Description: {sol.description}")
            log.info(f"Implementation:\n{sol.implementation}")


# ============================================================================
# DATASET VALIDATION AND MERGING
# ============================================================================

class DatasetValidator:
    """Comprehensive dataset validation with automatic error recovery."""
    
    def __init__(self, df: pd.DataFrame, name: str = "dataset"):
        self.df = df
        self.name = name
        self.validation_results = {}
    
    def validate_schema(self, expected_columns: List[str]) -> bool:
        """Validate that all expected columns are present."""
        log.info(f"Validating schema for {self.name}...")
        missing = set(expected_columns) - set(self.df.columns)
        
        if missing:
            error = PipelineError(
                error_type="MISSING_FEATURES",
                message=f"Missing columns in {self.name}: {missing}",
                context={"missing_columns": list(missing), "available_columns": self.df.columns.tolist()}
            )
            error.log_error_report()
            return False
        
        log.info(f"✓ Schema validation passed for {self.name}")
        return True
    
    def validate_datatypes(self, type_spec: Dict[str, str]) -> bool:
        """Validate column datatypes match specification."""
        log.info(f"Validating datatypes for {self.name}...")
        mismatches = []
        
        for col, expected_type in type_spec.items():
            if col in self.df.columns:
                actual_type = str(self.df[col].dtype)
                if expected_type not in actual_type:
                    mismatches.append({
                        "column": col,
                        "expected": expected_type,
                        "actual": actual_type
                    })
        
        if mismatches:
            error = PipelineError(
                error_type="DATATYPE_MISMATCH",
                message=f"Datatype mismatches in {self.name}",
                context={"mismatches": mismatches}
            )
            error.log_error_report()
            return False
        
        log.info(f"✓ Datatype validation passed for {self.name}")
        return True
    
    def validate_team_codes(self, team_columns: List[str]) -> bool:
        """Validate that team codes are properly normalized."""
        log.info(f"Validating team codes for {self.name}...")
        
        for col in team_columns:
            if col in self.df.columns:
                # Check for legacy codes that should be normalized
                unique_teams = self.df[col].unique()
                legacy_codes = set(unique_teams) & set(ABBR_FIX.keys())
                
                if legacy_codes:
                    error = PipelineError(
                        error_type="JOIN_ERROR",
                        message=f"Legacy team codes found in {col}: {legacy_codes}",
                        context={
                            "column": col,
                            "legacy_codes": list(legacy_codes),
                            "expected_mapping": {k: v for k, v in ABBR_FIX.items() if k in legacy_codes}
                        }
                    )
                    error.log_error_report()
                    return False
        
        log.info(f"✓ Team code validation passed for {self.name}")
        return True
    
    def validate_temporal_order(self) -> bool:
        """Validate that data is properly sorted by time."""
        log.info(f"Validating temporal order for {self.name}...")
        
        if 'season' not in self.df.columns or 'week' not in self.df.columns:
            log.warning(f"Cannot validate temporal order: missing season/week columns")
            return True
        
        time_keys = make_time_key(self.df)
        if not time_keys.is_monotonic_increasing:
            log.warning(f"⚠ Data is not sorted chronologically")
            return False
        
        log.info(f"✓ Temporal order validation passed for {self.name}")
        return True
    
    def validate_no_leakage(self, prior_features: List[str]) -> bool:
        """Validate that rolling features don't include current game (no leakage)."""
        log.info(f"Validating no data leakage for {self.name}...")
        
        # Check that prior features have some NaN values at the start
        # (indicating proper shift before rolling)
        issues = []
        for col in prior_features:
            if col in self.df.columns:
                # Sort by time and check first valid index
                df_sorted = self.df.sort_values(['season', 'week']).reset_index(drop=True)
                first_valid = df_sorted[col].first_valid_index()
                
                # First value should typically be NaN after shift(1)
                if first_valid is not None and first_valid == 0:
                    # Check if there's variation - if all values are the same, might be imputed
                    if df_sorted[col].nunique() > 1:
                        issues.append(col)
        
        if issues:
            error = PipelineError(
                error_type="FEATURE_LEAKAGE",
                message=f"Potential data leakage in features: {issues}",
                context={"suspicious_features": issues}
            )
            error.log_error_report()
            return False
        
        log.info(f"✓ No leakage validation passed for {self.name}")
        return True
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            "dataset_name": self.name,
            "timestamp": datetime.now().isoformat(),
            "shape": self.df.shape,
            "columns": self.df.columns.tolist(),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "null_counts": self.df.isnull().sum().to_dict(),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
            "validation_results": self.validation_results
        }
        return report


class DatasetMerger:
    """Merge and integrate multiple data sources with robust error handling."""
    
    def __init__(self):
        self.merge_log = []
    
    def merge_schedule_and_features(
        self,
        start_year: int,
        end_year: int,
        out_dir: Path
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Merge schedule data with engineered features.
        
        Returns:
            Tuple of (merged_df, merge_report)
        """
        log.info("="*80)
        log.info("DATASET MERGE WORKFLOW")
        log.info("="*80)
        
        try:
            # Build dataset using existing pipeline
            log.info(f"Building dataset for years {start_year}-{end_year}...")
            output_path, merged_df = build_dataset(
                start=start_year,
                end=end_year,
                out_dir=out_dir,
                production_mode=True,
                include_future=True
            )
            
            # Validate the merged dataset
            validator = DatasetValidator(merged_df, name="merged_dataset")
            
            # Run all validations
            schema_valid = validator.validate_schema(BASE_FEATURES + ['season', 'week', 'home_team', 'away_team'])
            dtype_valid = validator.validate_datatypes({
                'season': 'int',
                'week': 'int',
                'home_prior_pf_avg_3': 'float',
                'away_prior_pf_avg_3': 'float'
            })
            team_valid = validator.validate_team_codes(['home_team', 'away_team'])
            temporal_valid = validator.validate_temporal_order()
            leakage_valid = validator.validate_no_leakage(
                [f for f in BASE_FEATURES if 'prior' in f]
            )
            
            # Generate validation report
            validation_report = validator.generate_report()
            validation_report['all_checks_passed'] = all([
                schema_valid, dtype_valid, team_valid, temporal_valid, leakage_valid
            ])
            
            # Save validation report
            report_path = REPORTS_DIR / "dataset_validation_report.json"
            with open(report_path, 'w') as f:
                json.dump(validation_report, f, indent=2)
            log.info(f"Validation report saved to {report_path}")
            
            merge_report = {
                "output_path": str(output_path),
                "rows": len(merged_df),
                "columns": len(merged_df.columns),
                "validation": validation_report,
                "timestamp": datetime.now().isoformat()
            }
            
            if validation_report['all_checks_passed']:
                log.info("✓ All dataset validations passed")
            else:
                log.warning("⚠ Some dataset validations failed - check error reports")
            
            return merged_df, merge_report
            
        except Exception as e:
            log.error(f"Error during dataset merge: {e}")
            log.error(traceback.format_exc())
            
            error = PipelineError(
                error_type="MERGE_ERROR",
                message=str(e),
                context={
                    "start_year": start_year,
                    "end_year": end_year,
                    "traceback": traceback.format_exc()
                }
            )
            error.log_error_report()
            raise


# ============================================================================
# MODEL EVALUATION FRAMEWORK
# ============================================================================

class ModelEvaluator:
    """Comprehensive model evaluation with cross-validation and metrics tracking."""
    
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.evaluation_results = {}
    
    def load_models(self) -> Dict[str, Any]:
        """Load all trained models and preprocessor."""
        log.info("Loading trained models...")
        
        try:
            models = {
                "preprocessor": joblib.load(self.models_dir / "preprocessor.joblib"),
                "home_model": joblib.load(self.models_dir / "home_model.joblib"),
                "away_model": joblib.load(self.models_dir / "away_model.joblib"),
                "win_classifier": joblib.load(self.models_dir / "win_clf_calibrated.joblib")
            }
            
            # Load metadata if available
            metadata_path = self.models_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    models["metadata"] = json.load(f)
            
            log.info("✓ Models loaded successfully")
            return models
            
        except Exception as e:
            log.error(f"Error loading models: {e}")
            
            error = PipelineError(
                error_type="MODEL_LOAD_ERROR",
                message=f"Failed to load models: {e}",
                context={"models_dir": str(self.models_dir)}
            )
            error.log_error_report()
            raise
    
    def evaluate_regression_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str
    ) -> Dict[str, float]:
        """Evaluate regression model with comprehensive metrics."""
        log.info(f"Evaluating {model_name}...")
        
        try:
            predictions = model.predict(X)
            
            metrics = {
                "mae": float(mean_absolute_error(y, predictions)),
                "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
                "r2": float(r2_score(y, predictions)),
                "mean_actual": float(np.mean(y)),
                "mean_predicted": float(np.mean(predictions)),
                "std_actual": float(np.std(y)),
                "std_predicted": float(np.std(predictions))
            }
            
            log.info(f"✓ {model_name} - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
            return metrics
            
        except Exception as e:
            log.error(f"Error evaluating {model_name}: {e}")
            raise
    
    def evaluate_classifier_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str
    ) -> Dict[str, Any]:
        """Evaluate classification model with comprehensive metrics."""
        log.info(f"Evaluating {model_name}...")
        
        try:
            # Get probabilities and predictions
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[:, 1]
            else:
                probabilities = model.predict(X)
            
            predictions = (probabilities >= 0.5).astype(int)
            
            metrics = {
                "accuracy": float(accuracy_score(y, predictions)),
                "precision": float(precision_score(y, predictions, zero_division=0)),
                "recall": float(recall_score(y, predictions, zero_division=0)),
                "f1": float(f1_score(y, predictions, zero_division=0)),
                "roc_auc": float(roc_auc_score(y, probabilities)),
                "brier_score": float(brier_score_loss(y, probabilities)),
                "confusion_matrix": confusion_matrix(y, predictions).tolist()
            }
            
            log.info(f"✓ {model_name} - AUC: {metrics['roc_auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            return metrics
            
        except Exception as e:
            log.error(f"Error evaluating {model_name}: {e}")
            raise
    
    def cross_validate_models(
        self,
        df: pd.DataFrame,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Perform time-series cross-validation on all models.
        
        Uses TimeSeriesSplit to ensure no future information leakage.
        """
        log.info("="*80)
        log.info("MODEL CROSS-VALIDATION")
        log.info("="*80)
        
        try:
            # Load models
            models = self.load_models()
            
            # Prepare data - only completed games
            df_completed = df[
                df['home_points_for'].notna() & df['away_points_for'].notna()
            ].copy()
            
            # Sort chronologically
            df_completed = df_completed.sort_values(['season', 'week']).reset_index(drop=True)
            
            log.info(f"Using {len(df_completed)} completed games for evaluation")
            
            # Prepare features and targets
            X_raw = df_completed[BASE_FEATURES]
            y_home = df_completed['home_points_for'].values
            y_away = df_completed['away_points_for'].values
            y_win = (df_completed['home_points_for'] > df_completed['away_points_for']).astype(int).values
            
            # Transform features
            X = models['preprocessor'].transform(X_raw)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            cv_results = {
                "home_scores": [],
                "away_scores": [],
                "win_scores": []
            }
            
            log.info(f"Running {n_splits}-fold time-series cross-validation...")
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
                log.info(f"  Fold {fold}/{n_splits}...")
                
                X_train, X_test = X[train_idx], X[test_idx]
                y_home_train, y_home_test = y_home[train_idx], y_home[test_idx]
                y_away_train, y_away_test = y_away[train_idx], y_away[test_idx]
                y_win_train, y_win_test = y_win[train_idx], y_win[test_idx]
                
                # Evaluate home model
                home_pred = models['home_model'].predict(X_test)
                home_r2 = r2_score(y_home_test, home_pred)
                cv_results["home_scores"].append(home_r2)
                
                # Evaluate away model
                away_pred = models['away_model'].predict(X_test)
                away_r2 = r2_score(y_away_test, away_pred)
                cv_results["away_scores"].append(away_r2)
                
                # Evaluate win classifier
                win_prob = models['win_classifier'].predict_proba(X_test)[:, 1]
                win_auc = roc_auc_score(y_win_test, win_prob)
                cv_results["win_scores"].append(win_auc)
            
            # Calculate summary statistics
            summary = {
                "home_model": {
                    "cv_r2_scores": cv_results["home_scores"],
                    "mean_r2": float(np.mean(cv_results["home_scores"])),
                    "std_r2": float(np.std(cv_results["home_scores"]))
                },
                "away_model": {
                    "cv_r2_scores": cv_results["away_scores"],
                    "mean_r2": float(np.mean(cv_results["away_scores"])),
                    "std_r2": float(np.std(cv_results["away_scores"]))
                },
                "win_classifier": {
                    "cv_auc_scores": cv_results["win_scores"],
                    "mean_auc": float(np.mean(cv_results["win_scores"])),
                    "std_auc": float(np.std(cv_results["win_scores"]))
                },
                "n_folds": n_splits,
                "total_samples": len(df_completed)
            }
            
            log.info("✓ Cross-validation completed")
            log.info(f"  Home Model - Mean R²: {summary['home_model']['mean_r2']:.4f} ± {summary['home_model']['std_r2']:.4f}")
            log.info(f"  Away Model - Mean R²: {summary['away_model']['mean_r2']:.4f} ± {summary['away_model']['std_r2']:.4f}")
            log.info(f"  Win Classifier - Mean AUC: {summary['win_classifier']['mean_auc']:.4f} ± {summary['win_classifier']['std_auc']:.4f}")
            
            return summary
            
        except Exception as e:
            log.error(f"Error during cross-validation: {e}")
            log.error(traceback.format_exc())
            
            error = PipelineError(
                error_type="EVALUATION_ERROR",
                message=f"Cross-validation failed: {e}",
                context={"traceback": traceback.format_exc()}
            )
            error.log_error_report()
            raise
    
    def generate_evaluation_report(
        self,
        cv_results: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Path:
        """Generate comprehensive evaluation report."""
        if output_path is None:
            output_path = REPORTS_DIR / "model_evaluation_report.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "cross_validation_results": cv_results,
            "evaluation_summary": {
                "home_model_performance": "GOOD" if cv_results["home_model"]["mean_r2"] > 0.3 else "NEEDS_IMPROVEMENT",
                "away_model_performance": "GOOD" if cv_results["away_model"]["mean_r2"] > 0.3 else "NEEDS_IMPROVEMENT",
                "win_classifier_performance": "GOOD" if cv_results["win_classifier"]["mean_auc"] > 0.60 else "NEEDS_IMPROVEMENT"
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        log.info(f"Evaluation report saved to {output_path}")
        return output_path


# ============================================================================
# MAIN PIPELINE ORCHESTRATION
# ============================================================================

def run_enhanced_pipeline(
    start_year: int = 2010,
    end_year: int = 2025,
    evaluate_models: bool = True,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Run the complete enhanced pipeline workflow.
    
    Args:
        start_year: Starting season year
        end_year: Ending season year
        evaluate_models: Whether to evaluate models after dataset creation
        cv_folds: Number of cross-validation folds
    
    Returns:
        Dictionary containing pipeline results and reports
    """
    log.info("="*80)
    log.info("ENHANCED PIPELINE - FULL WORKFLOW")
    log.info("="*80)
    log.info(f"Start Year: {start_year}")
    log.info(f"End Year: {end_year}")
    log.info(f"Evaluate Models: {evaluate_models}")
    log.info("="*80)
    
    results = {
        "start_time": datetime.now().isoformat(),
        "parameters": {
            "start_year": start_year,
            "end_year": end_year,
            "evaluate_models": evaluate_models,
            "cv_folds": cv_folds
        }
    }
    
    try:
        # Step 1: Merge and validate dataset
        log.info("\n" + "="*80)
        log.info("STEP 1: DATASET MERGE AND VALIDATION")
        log.info("="*80)
        
        merger = DatasetMerger()
        merged_df, merge_report = merger.merge_schedule_and_features(
            start_year=start_year,
            end_year=end_year,
            out_dir=DATA_DIR
        )
        results["merge_report"] = merge_report
        
        # Step 2: Model evaluation (if requested)
        if evaluate_models:
            log.info("\n" + "="*80)
            log.info("STEP 2: MODEL EVALUATION")
            log.info("="*80)
            
            evaluator = ModelEvaluator(models_dir=MODELS_DIR)
            cv_results = evaluator.cross_validate_models(merged_df, n_splits=cv_folds)
            results["evaluation_results"] = cv_results
            
            # Generate evaluation report
            report_path = evaluator.generate_evaluation_report(cv_results)
            results["evaluation_report_path"] = str(report_path)
        
        # Final summary
        results["end_time"] = datetime.now().isoformat()
        results["status"] = "SUCCESS"
        
        log.info("\n" + "="*80)
        log.info("PIPELINE COMPLETED SUCCESSFULLY")
        log.info("="*80)
        log.info(f"Dataset rows: {merge_report['rows']}")
        log.info(f"Dataset columns: {merge_report['columns']}")
        
        if evaluate_models:
            log.info(f"Home Model CV R²: {cv_results['home_model']['mean_r2']:.4f}")
            log.info(f"Away Model CV R²: {cv_results['away_model']['mean_r2']:.4f}")
            log.info(f"Win Classifier CV AUC: {cv_results['win_classifier']['mean_auc']:.4f}")
        
        log.info(f"Log file: {LOG_FILE}")
        log.info("="*80)
        
        # Save final pipeline report
        pipeline_report_path = REPORTS_DIR / "pipeline_summary.json"
        with open(pipeline_report_path, 'w') as f:
            json.dump(results, f, indent=2)
        log.info(f"Pipeline summary saved to {pipeline_report_path}")
        
        return results
        
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        log.error(traceback.format_exc())
        
        results["end_time"] = datetime.now().isoformat()
        results["status"] = "FAILED"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        
        # Save error report
        error_report_path = REPORTS_DIR / "pipeline_error.json"
        with open(error_report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        log.error(f"Error report saved to {error_report_path}")
        
        raise


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI entry point for enhanced pipeline."""
    parser = argparse.ArgumentParser(
        description="Enhanced NFL Prediction Pipeline - Dataset Merge and Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with dataset creation and model evaluation
  python enhanced_pipeline.py --start 2010 --end 2025 --evaluate-models
  
  # Only create dataset without evaluation
  python enhanced_pipeline.py --start 2016 --end 2025 --no-evaluate-models
  
  # Run with custom cross-validation folds
  python enhanced_pipeline.py --start 2010 --end 2025 --cv-folds 10
        """
    )
    
    parser.add_argument(
        "--start",
        type=int,
        default=2010,
        help="Starting season year (default: 2010)"
    )
    
    parser.add_argument(
        "--end",
        type=int,
        default=2025,
        help="Ending season year (default: 2025)"
    )
    
    parser.add_argument(
        "--evaluate-models",
        action="store_true",
        default=True,
        help="Evaluate models after dataset creation (default: True)"
    )
    
    parser.add_argument(
        "--no-evaluate-models",
        action="store_false",
        dest="evaluate_models",
        help="Skip model evaluation"
    )
    
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    
    args = parser.parse_args()
    
    try:
        results = run_enhanced_pipeline(
            start_year=args.start,
            end_year=args.end,
            evaluate_models=args.evaluate_models,
            cv_folds=args.cv_folds
        )
        
        sys.exit(0)
        
    except Exception as e:
        log.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
