#!/usr/bin/env python
"""
Quick status check script for the enhanced pipeline
"""
import json
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"
REPORTS_DIR = BACKEND_DIR / "reports"

print("="*80)
print("NFL DATASET MERGE & MODEL EVALUATION - STATUS CHECK")
print("="*80)
print()

# Check dataset
dataset_path = BACKEND_DIR / "data" / "Nfl_data_sorted.csv"
if dataset_path.exists():
    import pandas as pd
    df = pd.read_csv(dataset_path)
    print(f"✓ Dataset: {len(df)} games, {len(df.columns)} columns")
    completed = df[df['home_points_for'].notna()]
    print(f"  - Completed games: {len(completed)}")
    print(f"  - Future games: {len(df) - len(completed)}")
else:
    print("✗ Dataset not found!")

print()

# Check models
required_models = [
    "home_model.joblib",
    "away_model.joblib",
    "win_clf_calibrated.joblib",
    "preprocessor.joblib"
]

all_present = True
for model in required_models:
    model_path = MODELS_DIR / model
    if model_path.exists():
        size_kb = model_path.stat().st_size / 1024
        print(f"✓ {model}: {size_kb:.1f} KB")
    else:
        print(f"✗ {model}: MISSING")
        all_present = False

print()

# Check metadata
metadata_path = MODELS_DIR / "metadata.json"
if metadata_path.exists():
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("Model Metadata:")
    print(f"  Training timestamp: {metadata.get('training_timestamp', 'N/A')}")
    print(f"  Training samples: {metadata.get('training_samples', 'N/A')}")
    print(f"  Dataset hash: {metadata.get('dataset_hash', 'N/A')}")
    
    scores = metadata.get('model_scores', {})
    print()
    print("Model Scores:")
    print(f"  Home RMSE (CV): {scores.get('home_r2_cv', 'N/A'):.3f}" if scores.get('home_r2_cv') else "  Home RMSE (CV): N/A")
    print(f"  Away RMSE (CV): {scores.get('away_r2_cv', 'N/A'):.3f}" if scores.get('away_r2_cv') else "  Away RMSE (CV): N/A")
    print(f"  Win AUC (CV): {scores.get('win_auc_cv', 'N/A'):.4f}" if scores.get('win_auc_cv') else "  Win AUC (CV): N/A")
    
    production_ready = metadata.get('production_ready_win_model', False)
    print()
    print(f"Production Ready: {'✓ YES' if production_ready else '⚠ NO (below threshold)'}")
else:
    print("✗ Metadata not found - models may not be trained yet")

print()

# Check reports
print("Reports:")
if (REPORTS_DIR / "enhanced_pipeline.log").exists():
    log_size = (REPORTS_DIR / "enhanced_pipeline.log").stat().st_size / 1024
    print(f"✓ enhanced_pipeline.log: {log_size:.1f} KB")

if (REPORTS_DIR / "model_evaluation.json").exists():
    eval_path = REPORTS_DIR / "model_evaluation.json"
    with open(eval_path, 'r') as f:
        evaluation = json.load(f)
    print(f"✓ model_evaluation.json: {evaluation['timestamp']}")
    prod_ready = evaluation['production_readiness']['overall_ready']
    print(f"  Overall Ready: {'✓ YES' if prod_ready else '⚠ NO'}")
else:
    print("⚠ model_evaluation.json: Not generated yet")

training_report = MODELS_DIR / "training_report.json"
if training_report.exists():
    with open(training_report, 'r') as f:
        report = json.load(f)
    print(f"✓ training_report.json")
    print(f"  Dataset rows: {report.get('dataset', {}).get('rows', 'N/A')}")
else:
    print("⚠ training_report.json: Not generated yet")

print()
print("="*80)

# Summary recommendation
if all_present and metadata_path.exists():
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    auc = metadata.get('model_scores', {}).get('win_auc_cv', 0)
    
    if auc >= 0.65:
        print("✓ READY FOR PRODUCTION")
    elif auc >= 0.60:
        print("⚠ READY FOR TESTING (Consider improving AUC)")
        print("  Recommendation: Expand training data or tune hyperparameters")
    else:
        print("✗ NOT READY - Models need improvement")
        print("  Run: python backend/enhanced_pipeline.py --full")
else:
    print("⚠ INCOMPLETE - Run full pipeline")
    print("  Run: python backend/enhanced_pipeline.py --full")

print("="*80)
