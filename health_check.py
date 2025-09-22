#!/usr/bin/env python3
"""
Production Health Check Script
==============================

Validates that the NFL Prediction API is ready for deployment by checking:
- Model files exist and are valid
- Dataset is available and properly formatted
- All required dependencies are installed
- API endpoints respond correctly

Usage:
    python health_check.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies() -> bool:
    """Check if all required packages are available."""
    required_packages = [
        'fastapi', 'uvicorn', 'pandas', 'numpy', 'scikit-learn', 
        'joblib', 'lightgbm', 'tensorflow', 'nfl_data_py'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✓ {package} is available")
        except ImportError:
            missing.append(package)
            logger.error(f"✗ {package} is missing")
    
    if missing:
        logger.error(f"Missing dependencies: {missing}")
        return False
    
    logger.info("✓ All dependencies are available")
    return True

def check_model_files() -> bool:
    """Check if model files exist and metadata is valid."""
    base_dir = Path(__file__).parent
    models_dir = base_dir / "backend" / "models"
    
    if not models_dir.exists():
        logger.error("✗ Models directory does not exist")
        return False
    
    # Check metadata
    metadata_path = models_dir / "metadata.json"
    if not metadata_path.exists():
        logger.error("✗ Model metadata.json not found")
        return False
    
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        logger.info("✓ Model metadata is valid JSON")
    except json.JSONDecodeError:
        logger.error("✗ Model metadata is invalid JSON")
        return False
    
    # Check required model files
    required_files = [
        metadata.get('preprocessor', 'preprocessor.joblib'),
        metadata.get('models', {}).get('home_model', 'home_model.joblib'),
        metadata.get('models', {}).get('away_model', 'away_model.joblib')
    ]
    
    for file_name in required_files:
        file_path = models_dir / file_name
        if not file_path.exists():
            logger.error(f"✗ Model file missing: {file_name}")
            return False
        logger.info(f"✓ Model file exists: {file_name}")
    
    logger.info("✓ All model files are present")
    return True

def check_dataset() -> bool:
    """Check if dataset exists and has required structure."""
    base_dir = Path(__file__).parent
    data_dir = base_dir / "backend" / "data"
    
    dataset_path = data_dir / "Nfl_data_sorted.csv"
    if not dataset_path.exists():
        logger.error("✗ Main dataset file not found")
        return False
    
    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"✓ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"✗ Failed to load dataset: {e}")
        return False
    
    # Check required columns
    required_columns = [
        'season', 'week', 'home_team', 'away_team',
        'home_prior_pf_avg_3', 'home_prior_pa_avg_3', 'home_prior_win_pct_3',
        'away_prior_pf_avg_3', 'away_prior_pa_avg_3', 'away_prior_win_pct_3'
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"✗ Missing required columns: {missing_cols}")
        return False
    
    logger.info("✓ Dataset has all required columns")
    return True

def check_heroku_config() -> bool:
    """Check Heroku deployment configuration."""
    base_dir = Path(__file__).parent
    
    # Check required files
    required_files = ['Procfile', 'requirements.txt', 'runtime.txt']
    for filename in required_files:
        file_path = base_dir / filename
        if not file_path.exists():
            logger.error(f"✗ Missing Heroku config file: {filename}")
            return False
        logger.info(f"✓ Heroku config file exists: {filename}")
    
    # Validate Procfile
    procfile_path = base_dir / "Procfile"
    with open(procfile_path) as f:
        procfile_content = f.read().strip()
    
    if not procfile_content.startswith('web:'):
        logger.error("✗ Procfile does not define a web process")
        return False
    
    if 'backend.main:app' not in procfile_content:
        logger.error("✗ Procfile does not reference backend.main:app")
        return False
    
    logger.info("✓ Procfile is correctly configured")
    return True

def main() -> None:
    """Run all health checks."""
    logger.info("🏈 NFL Prediction API - Production Health Check")
    logger.info("=" * 50)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Model Files", check_model_files),
        ("Dataset", check_dataset),
        ("Heroku Config", check_heroku_config)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        logger.info(f"\nRunning {check_name} check...")
        if check_func():
            passed += 1
        else:
            logger.error(f"❌ {check_name} check failed")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"Health Check Results: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info("🎉 All checks passed! Ready for production deployment.")
        sys.exit(0)
    else:
        logger.error(f"💥 {total - passed} checks failed. Fix issues before deployment.")
        sys.exit(1)

if __name__ == "__main__":
    main()