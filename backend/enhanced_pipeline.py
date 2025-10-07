#!/usr/bin/env python
"""
enhanced_pipeline.py - Full Dataset Merge and Model Evaluation Workflow

This script performs:
1. Dataset validation and corruption detection
2. Dataset rebuilding with proper column names
3. Model training with comprehensive error handling
4. Model evaluation with production-readiness checks
5. Automated recovery suggestions for any failures

Usage:
    python backend/enhanced_pipeline.py --rebuild-dataset --train-models --evaluate
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np

# Setup paths
BACKEND_DIR = Path(__file__).resolve().parent
REPO_DIR = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"
REPORTS_DIR = BACKEND_DIR / "reports"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(REPORTS_DIR / "enhanced_pipeline.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("enhanced_pipeline")


class DatasetValidationError(Exception):
    """Raised when dataset validation fails"""
    pass


class ModelTrainingError(Exception):
    """Raised when model training fails"""
    pass


class RecoverySolution:
    """Container for recovery solutions"""
    def __init__(self, issue: str, solutions: List[str]):
        self.issue = issue
        self.solutions = solutions
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue": self.issue,
            "solutions": self.solutions,
            "timestamp": self.timestamp
        }
    
    def print_solutions(self):
        log.error(f"\n{'='*80}")
        log.error(f"ISSUE DETECTED: {self.issue}")
        log.error(f"{'='*80}")
        log.error("\nProduction-Ready Recovery Solutions:")
        for i, solution in enumerate(self.solutions, 1):
            log.error(f"\nSolution {i}:")
            log.error(solution)
        log.error(f"\n{'='*80}\n")


def validate_dataset(csv_path: Path) -> Tuple[bool, Optional[RecoverySolution]]:
    """
    Validate the dataset structure and detect corruption.
    
    Returns:
        Tuple of (is_valid, recovery_solution)
    """
    log.info(f"Validating dataset: {csv_path}")
    
    if not csv_path.exists():
        issue = f"Dataset not found at {csv_path}"
        solutions = [
            "Run: python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data",
            "Ensure nfl-data-py is installed: pip install nfl-data-py",
            "Check that the backend/data directory has write permissions"
        ]
        return False, RecoverySolution(issue, solutions)
    
    try:
        # Read first few rows to check structure
        df = pd.read_csv(csv_path, nrows=5)
        
        # Check for corrupted headers
        corrupted_cols = [col for col in df.columns if 'no need to ask' in str(col).lower()]
        if corrupted_cols:
            issue = f"Dataset has corrupted column headers: {corrupted_cols}"
            solutions = [
                "Solution 1 - Rebuild dataset:\n"
                "  python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data",
                
                "Solution 2 - Fix headers manually:\n"
                "  1. Backup current file: cp backend/data/Nfl_data_sorted.csv backend/data/Nfl_data_sorted.csv.bak\n"
                "  2. Open file and replace corrupted header with proper column names\n"
                "  3. Verify columns match: home_prior_pf_avg_3, home_prior_pf_avg_5, etc."
            ]
            return False, RecoverySolution(issue, solutions)
        
        # Check for required base columns
        required_cols = ['season', 'week', 'game_id', 'home_team', 'away_team', 
                        'home_points_for', 'away_points_for']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            issue = f"Dataset missing required columns: {missing_cols}"
            solutions = [
                "Rebuild dataset with proper structure:\n"
                "  python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data",
                
                "Check if dataset is from old format:\n"
                "  1. Review dataset source\n"
                "  2. Update build_csv_datasets.py if schema changed\n"
                "  3. Re-run dataset generation"
            ]
            return False, RecoverySolution(issue, solutions)
        
        # Check for expected feature columns
        expected_features = [
            'home_prior_pf_avg_3', 'home_prior_pf_avg_5',
            'away_prior_pf_avg_3', 'away_prior_pf_avg_5',
            'home_minus_away_pf_avg_3', 'home_minus_away_pf_avg_5'
        ]
        missing_features = [col for col in expected_features if col not in df.columns]
        
        if missing_features:
            issue = f"Dataset missing expected feature columns: {missing_features}"
            solutions = [
                "Rebuild dataset with feature engineering:\n"
                "  python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data\n"
                "  This will regenerate rolling averages and differential features",
                
                "Check build_csv_datasets.py add_features() function:\n"
                "  Verify windows=(3, 5) is set correctly\n"
                "  Ensure differential features are computed"
            ]
            return False, RecoverySolution(issue, solutions)
        
        # Check data types
        dtype_issues = []
        if not pd.api.types.is_integer_dtype(df['season']):
            dtype_issues.append(f"season column has type {df['season'].dtype}, expected integer")
        if not pd.api.types.is_integer_dtype(df['week']):
            dtype_issues.append(f"week column has type {df['week'].dtype}, expected integer")
        
        if dtype_issues:
            issue = "Dataset has datatype mismatches:\n" + "\n".join(dtype_issues)
            solutions = [
                "Solution 1 - Rebuild with proper types:\n"
                "  python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data",
                
                "Solution 2 - Fix types in place:\n"
                "  import pandas as pd\n"
                "  df = pd.read_csv('backend/data/Nfl_data_sorted.csv')\n"
                "  df['season'] = df['season'].astype(int)\n"
                "  df['week'] = df['week'].astype(int)\n"
                "  df.to_csv('backend/data/Nfl_data_sorted.csv', index=False)"
            ]
            return False, RecoverySolution(issue, solutions)
        
        log.info(f"✓ Dataset validation passed: {len(df)} rows checked")
        return True, None
        
    except pd.errors.ParserError as e:
        issue = f"Dataset parsing error: {str(e)}"
        solutions = [
            "Rebuild dataset from scratch:\n"
            "  python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data",
            
            "Check for file corruption:\n"
            "  1. Delete corrupted file\n"
            "  2. Re-download or regenerate dataset\n"
            "  3. Verify file integrity"
        ]
        return False, RecoverySolution(issue, solutions)
    
    except Exception as e:
        issue = f"Unexpected validation error: {type(e).__name__}: {str(e)}"
        solutions = [
            "General recovery steps:\n"
            "  1. Check error log above for details\n"
            "  2. Rebuild dataset: python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data\n"
            "  3. Verify dependencies: pip install -r requirements.txt"
        ]
        return False, RecoverySolution(issue, solutions)


def fix_dataset_headers(csv_path: Path) -> bool:
    """
    Attempt to fix corrupted dataset headers.
    
    Returns:
        True if fix was successful, False otherwise
    """
    log.info(f"Attempting to fix dataset headers: {csv_path}")
    
    try:
        # Read the raw file to inspect
        with open(csv_path, 'r') as f:
            first_line = f.readline()
            
        # Expected column structure
        expected_cols = [
            "season", "week", "game_id", "home_game_date",
            "home_team", "away_team",
            "home_points_for", "away_points_for", "point_diff", "winner",
            # Home priors (windows 3 and 5)
            "home_prior_pf_avg_3", "home_prior_pf_avg_5",
            "home_prior_pa_avg_3", "home_prior_pa_avg_5",
            "home_prior_win_pct_3", "home_prior_win_pct_5",
            # Away priors (windows 3 and 5)
            "away_prior_pf_avg_3", "away_prior_pf_avg_5",
            "away_prior_pa_avg_3", "away_prior_pa_avg_5",
            "away_prior_win_pct_3", "away_prior_win_pct_5",
            # Differentials (windows 3 and 5)
            "home_minus_away_pf_avg_3", "home_minus_away_pf_avg_5",
            "home_minus_away_pa_avg_3", "home_minus_away_pa_avg_5",
            "home_minus_away_win_pct_3", "home_minus_away_win_pct_5",
        ]
        
        # Read a data row to count actual columns
        with open(csv_path, 'r') as f:
            f.readline()  # skip header
            second_line = f.readline()
        
        data_commas = second_line.count(',')
        data_cols = data_commas + 1
        
        header_commas = first_line.count(',')
        header_cols = header_commas + 1
        
        log.info(f"Header has {header_cols} columns")
        log.info(f"Data rows have {data_cols} columns")
        log.info(f"Expected {len(expected_cols)} columns")
        
        # If data has correct number of columns but header doesn't, fix it
        if data_cols == len(expected_cols):
            # Read data skipping corrupted header
            df = pd.read_csv(csv_path, skiprows=1, header=None, names=expected_cols)
            
            # Backup original
            backup_path = csv_path.parent / f"{csv_path.stem}.backup{csv_path.suffix}"
            import shutil
            shutil.copy2(csv_path, backup_path)
            log.info(f"Created backup at {backup_path}")
            
            # Save with correct headers
            df.to_csv(csv_path, index=False)
            log.info(f"✓ Fixed dataset headers successfully")
            return True
        elif header_cols == len(expected_cols):
            log.info("✓ Header already correct")
            return True
        else:
            log.warning(f"Column count mismatch - cannot fix automatically")
            log.warning(f"Header: {header_cols}, Data: {data_cols}, Expected: {len(expected_cols)}")
            return False
            
    except Exception as e:
        log.error(f"Failed to fix headers: {type(e).__name__}: {str(e)}")
        return False


def merge_datasets() -> Tuple[bool, Optional[RecoverySolution]]:
    """
    Merge multiple datasets if needed and ensure consistency.
    
    Returns:
        Tuple of (success, recovery_solution)
    """
    log.info("Checking for datasets to merge...")
    
    main_csv = DATA_DIR / "Nfl_data_sorted.csv"
    team_game_csv = DATA_DIR / "team_game_base.csv"
    schedule_csv = DATA_DIR / "Nfl_schedule_2025_2026.csv"
    
    # Check if main dataset needs merging
    if main_csv.exists():
        try:
            df_main = pd.read_csv(main_csv)
            log.info(f"Main dataset has {len(df_main)} rows")
            
            # If team_game_base exists, it's already been merged
            if team_game_csv.exists():
                log.info("✓ Datasets already merged")
                return True, None
            else:
                log.info("✓ Main dataset complete, no merge needed")
                return True, None
                
        except Exception as e:
            issue = f"Error reading main dataset: {str(e)}"
            solutions = [
                "Rebuild complete dataset:\n"
                "  python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data",
                
                "Check file integrity:\n"
                "  Verify backend/data/Nfl_data_sorted.csv is not corrupted"
            ]
            return False, RecoverySolution(issue, solutions)
    else:
        issue = "Main dataset not found"
        solutions = [
            "Generate dataset:\n"
            "  python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data"
        ]
        return False, RecoverySolution(issue, solutions)


def train_models() -> Tuple[bool, Optional[RecoverySolution]]:
    """
    Train models with comprehensive error handling.
    
    Returns:
        Tuple of (success, recovery_solution)
    """
    log.info("Training models...")
    
    try:
        # Import train_models module
        import sys
        sys.path.insert(0, str(BACKEND_DIR))
        import train_models
        
        # Run training
        train_models.main()
        log.info("✓ Model training completed successfully")
        return True, None
        
    except ImportError as e:
        issue = f"Failed to import train_models: {str(e)}"
        solutions = [
            "Install required dependencies:\n"
            "  pip install -r backend/requirements.txt",
            
            "Check Python environment:\n"
            "  python --version\n"
            "  which python"
        ]
        return False, RecoverySolution(issue, solutions)
    
    except FileNotFoundError as e:
        issue = f"Dataset not found during training: {str(e)}"
        solutions = [
            "Generate dataset first:\n"
            "  python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data",
            
            "Verify dataset location:\n"
            "  ls -la backend/data/Nfl_data_sorted.csv"
        ]
        return False, RecoverySolution(issue, solutions)
    
    except ValueError as e:
        error_msg = str(e)
        
        if "Missing required features" in error_msg:
            issue = f"Feature mismatch during training: {error_msg}"
            solutions = [
                "Rebuild dataset with all features:\n"
                "  python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data\n"
                "  This will regenerate all 18 required features",
                
                "Check feature generation in build_csv_datasets.py:\n"
                "  Verify add_features() creates all prior and differential columns"
            ]
        elif "Insufficient training data" in error_msg:
            issue = f"Insufficient data for training: {error_msg}"
            solutions = [
                "Expand date range for dataset:\n"
                "  python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data",
                
                "Check completed games:\n"
                "  Ensure dataset includes games with scores, not just scheduled games"
            ]
        else:
            issue = f"Training validation error: {error_msg}"
            solutions = [
                "Review error and rebuild dataset:\n"
                "  python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data",
                
                "Check train_models.py validation logic:\n"
                "  Review _load_dataset() function for validation rules"
            ]
        
        return False, RecoverySolution(issue, solutions)
    
    except Exception as e:
        issue = f"Unexpected training error: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        solutions = [
            "Full recovery procedure:\n"
            "  1. Clean existing models: rm -rf backend/models/*.joblib\n"
            "  2. Rebuild dataset: python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data\n"
            "  3. Retry training: python backend/train_models.py",
            
            "Debug mode:\n"
            "  Set logging to DEBUG in train_models.py\n"
            "  Review full error traceback above"
        ]
        return False, RecoverySolution(issue, solutions)


def evaluate_models() -> Tuple[bool, Optional[RecoverySolution]]:
    """
    Evaluate trained models and check production readiness.
    
    Returns:
        Tuple of (success, recovery_solution)
    """
    log.info("Evaluating models...")
    
    # Check if models exist
    required_models = [
        "home_model.joblib",
        "away_model.joblib",
        "win_clf_calibrated.joblib",
        "preprocessor.joblib"
    ]
    
    missing_models = [m for m in required_models if not (MODELS_DIR / m).exists()]
    
    if missing_models:
        issue = f"Missing trained models: {missing_models}"
        solutions = [
            "Train models:\n"
            "  python backend/train_models.py",
            
            "Full rebuild:\n"
            "  1. python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data\n"
            "  2. python backend/train_models.py"
        ]
        return False, RecoverySolution(issue, solutions)
    
    try:
        # Read metadata
        metadata_path = MODELS_DIR / "metadata.json"
        if not metadata_path.exists():
            issue = "metadata.json not found"
            solutions = [
                "Retrain models to generate metadata:\n"
                "  python backend/train_models.py"
            ]
            return False, RecoverySolution(issue, solutions)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Read training report
        report_path = MODELS_DIR / "training_report.json"
        training_report = {}
        if report_path.exists():
            with open(report_path, 'r') as f:
                training_report = json.load(f)
        
        # Generate evaluation report
        evaluation = {
            "timestamp": datetime.now().isoformat(),
            "models_evaluated": required_models,
            "metadata": metadata,
            "training_report": training_report,
            "production_readiness": {}
        }
        
        # Check production readiness criteria
        model_scores = metadata.get("model_scores", {})
        win_auc = model_scores.get("win_auc_cv", 0)
        
        # Production criteria
        criteria = {
            "win_auc_threshold": 0.60,
            "min_training_samples": 500,
            "all_models_present": len(missing_models) == 0
        }
        
        evaluation["production_readiness"] = {
            "win_auc_cv": win_auc,
            "meets_auc_threshold": win_auc >= criteria["win_auc_threshold"],
            "training_samples": metadata.get("training_samples", 0),
            "meets_sample_threshold": metadata.get("training_samples", 0) >= criteria["min_training_samples"],
            "all_models_present": criteria["all_models_present"],
            "overall_ready": (
                win_auc >= criteria["win_auc_threshold"] and
                metadata.get("training_samples", 0) >= criteria["min_training_samples"] and
                criteria["all_models_present"]
            )
        }
        
        # Save evaluation report
        eval_report_path = REPORTS_DIR / "model_evaluation.json"
        with open(eval_report_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
        
        log.info(f"✓ Model evaluation complete. Report saved to {eval_report_path}")
        
        # Log key metrics
        log.info("\n" + "="*80)
        log.info("MODEL EVALUATION SUMMARY")
        log.info("="*80)
        log.info(f"Win Classifier AUC: {win_auc:.4f}")
        log.info(f"Training Samples: {metadata.get('training_samples', 0)}")
        log.info(f"Production Ready: {evaluation['production_readiness']['overall_ready']}")
        log.info("="*80 + "\n")
        
        # If not production ready, provide solutions
        if not evaluation['production_readiness']['overall_ready']:
            issues = []
            if not evaluation['production_readiness']['meets_auc_threshold']:
                issues.append(f"Win AUC ({win_auc:.4f}) below threshold ({criteria['win_auc_threshold']})")
            if not evaluation['production_readiness']['meets_sample_threshold']:
                issues.append(f"Training samples ({metadata.get('training_samples', 0)}) below threshold ({criteria['min_training_samples']})")
            
            issue = "Models not production-ready:\n" + "\n".join(f"  - {i}" for i in issues)
            solutions = [
                "Solution 1 - Expand training data:\n"
                "  python backend/build_csv_datasets.py --start 2005 --end 2025 --out-dir backend/data\n"
                "  python backend/train_models.py\n"
                "  More historical data may improve model performance",
                
                "Solution 2 - Tune hyperparameters:\n"
                "  Edit train_models.py _grid_lgbm_clf() function\n"
                "  Expand hyperparameter search space\n"
                "  Consider different model architectures",
                
                "Solution 3 - Feature engineering:\n"
                "  Add more rolling windows (e.g., 7, 10 games)\n"
                "  Include additional features (strength of schedule, weather, injuries)\n"
                "  Edit build_csv_datasets.py add_features() function"
            ]
            
            # Still return success but log warning
            recovery = RecoverySolution(issue, solutions)
            recovery.print_solutions()
            log.warning("Models trained but not meeting production criteria - see solutions above")
        
        return True, None
        
    except json.JSONDecodeError as e:
        issue = f"Invalid JSON in metadata: {str(e)}"
        solutions = [
            "Retrain models to regenerate metadata:\n"
            "  python backend/train_models.py"
        ]
        return False, RecoverySolution(issue, solutions)
    
    except Exception as e:
        issue = f"Evaluation error: {type(e).__name__}: {str(e)}"
        solutions = [
            "Check model files:\n"
            "  ls -la backend/models/\n"
            "  Verify all .joblib files are present",
            
            "Retrain if needed:\n"
            "  python backend/train_models.py"
        ]
        return False, RecoverySolution(issue, solutions)


def main():
    parser = argparse.ArgumentParser(
        description="Full NFL Dataset Merge and Model Evaluation Workflow"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing dataset"
    )
    parser.add_argument(
        "--fix-headers",
        action="store_true",
        help="Attempt to fix corrupted dataset headers"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge multiple datasets if needed"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train models"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate models"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline: validate → merge → train → evaluate"
    )
    
    args = parser.parse_args()
    
    # If no args, run full pipeline
    if not any([args.validate, args.fix_headers, args.merge, args.train, args.evaluate, args.full]):
        args.full = True
    
    log.info("\n" + "="*80)
    log.info("NFL DATASET MERGE AND MODEL EVALUATION WORKFLOW")
    log.info("="*80 + "\n")
    
    success = True
    
    # Step 1: Validate dataset
    if args.validate or args.full:
        dataset_path = DATA_DIR / "Nfl_data_sorted.csv"
        valid, recovery = validate_dataset(dataset_path)
        
        if not valid:
            recovery.print_solutions()
            if args.full:
                # Try to fix headers if it's a header issue
                if "corrupted column headers" in recovery.issue.lower():
                    log.info("Attempting automatic header fix...")
                    if fix_dataset_headers(dataset_path):
                        log.info("Headers fixed, re-validating...")
                        valid, recovery = validate_dataset(dataset_path)
                        if not valid:
                            recovery.print_solutions()
                            success = False
                    else:
                        log.error("Automatic header fix failed")
                        success = False
                else:
                    success = False
        
        if not valid and not args.full:
            sys.exit(1)
    
    # Step 2: Fix headers if requested
    if args.fix_headers:
        dataset_path = DATA_DIR / "Nfl_data_sorted.csv"
        if fix_dataset_headers(dataset_path):
            log.info("✓ Headers fixed successfully")
        else:
            log.error("Failed to fix headers")
            sys.exit(1)
    
    # Step 3: Merge datasets
    if args.merge or args.full:
        success_merge, recovery = merge_datasets()
        if not success_merge:
            recovery.print_solutions()
            success = False
            if not args.full:
                sys.exit(1)
    
    # Step 4: Train models
    if args.train or args.full:
        success_train, recovery = train_models()
        if not success_train:
            recovery.print_solutions()
            success = False
            if not args.full:
                sys.exit(1)
    
    # Step 5: Evaluate models
    if args.evaluate or args.full:
        success_eval, recovery = evaluate_models()
        if not success_eval:
            recovery.print_solutions()
            success = False
            if not args.full:
                sys.exit(1)
    
    # Final summary
    log.info("\n" + "="*80)
    if success:
        log.info("✓ WORKFLOW COMPLETED SUCCESSFULLY")
    else:
        log.error("✗ WORKFLOW COMPLETED WITH ERRORS - See recovery solutions above")
    log.info("="*80 + "\n")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
