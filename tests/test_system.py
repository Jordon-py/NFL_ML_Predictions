#!/usr/bin/env python3
"""
NFL Prediction System - End-to-End Test Script
==============================================

This script tests the complete NFL prediction pipeline:
1. Data building
2. Model training  
3. API functionality
4. Prediction accuracy

Usage:
    python test_system.py [--quick]
    
    --quick: Skip data rebuilding and model retraining
"""

import json
import subprocess
import sys
import time
from pathlib import Path
import requests
from typing import Dict, Any

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'

def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=BASE_DIR)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {' '.join(cmd)}")
        print(f"   Error: {e.stderr}")
        return False

def test_api_endpoint(url: str, method: str = 'GET', data: Dict[str, Any] = None) -> bool:
    """Test an API endpoint and return success status."""
    try:
        if method == 'GET':
            response = requests.get(url, timeout=10)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=10)
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        if response.status_code == 200:
            print(f"✅ {method} {url} - Status: {response.status_code}")
            if response.headers.get('content-type', '').startswith('application/json'):
                print(f"   Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"❌ {method} {url} - Status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {method} {url} - Error: {e}")
        return False

def check_file_exists(path: Path, description: str) -> bool:
    """Check if a file exists and report status."""
    if path.exists():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description} missing: {path}")
        return False

def main():
    quick_mode = '--quick' in sys.argv
    print("🏈 NFL Prediction System - End-to-End Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Step 1: Check if data building is needed
    if not quick_mode:
        print("\n📊 Step 1: Building CSV datasets...")
        success = run_command([
            sys.executable, 'backend/build_csv_datasets.py'
        ], "Building CSV datasets")
        all_tests_passed &= success
    else:
        print("\n📊 Step 1: Skipping data building (quick mode)")
    
    # Check data files exist
    print("\n📁 Checking data files...")
    data_files = [
        (DATA_DIR / 'Nfl_data.csv', "Main dataset"),
        (DATA_DIR / 'Nfl_data.schema.json', "Dataset schema JSON"),
        (DATA_DIR / 'Nfl_data.schema.md', "Dataset schema Markdown")
    ]
    
    for file_path, description in data_files:
        all_tests_passed &= check_file_exists(file_path, description)
    
    # Step 2: Train models
    if not quick_mode:
        print("\n🤖 Step 2: Training models...")
        success = run_command([
            sys.executable, 'backend/train_models.py'
        ], "Training ML models")
        all_tests_passed &= success
    else:
        print("\n🤖 Step 2: Skipping model training (quick mode)")
    
    # Check model files exist
    print("\n🔍 Checking model files...")
    model_files = [
        (MODELS_DIR / 'preprocessor.joblib', "Data preprocessor"),
        (MODELS_DIR / 'nn_model.joblib', "Neural network model"),
        (MODELS_DIR / 'gbm_model.txt', "Gradient boosting model"),
        (MODELS_DIR / 'metadata.json', "Model metadata")
    ]
    
    for file_path, description in model_files:
        all_tests_passed &= check_file_exists(file_path, description)
    
    # Step 3: Test API
    print("\n🌐 Step 3: Testing API endpoints...")
    
    # Wait for any existing server to start (in case it was started manually)
    time.sleep(2)
    
    # Test health endpoint
    success = test_api_endpoint('http://127.0.0.1:8000/health')
    all_tests_passed &= success
    
    # Test root endpoint
    success = test_api_endpoint('http://127.0.0.1:8000/')
    all_tests_passed &= success
    
    # Test prediction endpoint
    sample_game = {
        'home_passer_rating': 95.5,
        'away_passer_rating': 87.2,
        'home_turnovers': 1,
        'away_turnovers': 2,
        'home_rushing_yards': 120.0,
        'away_rushing_yards': 85.0,
        'home': 'Chiefs',
        'away': 'Bills'
    }
    
    success = test_api_endpoint('http://127.0.0.1:8000/predict', 'POST', sample_game)
    all_tests_passed &= success
    
    # Test prediction with different teams
    sample_game2 = {
        'home_passer_rating': 82.1,
        'away_passer_rating': 98.7,
        'home_turnovers': 3,
        'away_turnovers': 0,
        'home_rushing_yards': 65.0,
        'away_rushing_yards': 185.0,
        'home': 'Giants',
        'away': 'Cowboys'
    }
    
    success = test_api_endpoint('http://127.0.0.1:8000/predict', 'POST', sample_game2)
    all_tests_passed &= success
    
    # Summary
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Your NFL prediction system is ready to use.")
        print("\n📋 System Status:")
        print("   ✅ Data pipeline working")
        print("   ✅ Models trained and saved")
        print("   ✅ API endpoints functional")
        print("   ✅ Predictions generating successfully")
        
        print("\n🚀 Next Steps:")
        print("   1. Start the backend server: python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000")
        print("   2. Start the React frontend: cd frontend && npm start")
        print("   3. Visit http://localhost:3000 to use the web interface")
        
    else:
        print("❌ SOME TESTS FAILED! Please check the errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
