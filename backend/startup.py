#!/usr/bin/env python3
"""
Production Startup Validation
=============================

Validates all dependencies and configurations before starting the API server.
This ensures that any issues are caught early in the deployment process.

Usage:
    python backend/startup.py && uvicorn backend.main:app --host 0.0.0.0 --port $PORT
"""

import logging
import os
import sys
from pathlib import Path

# Add backend to path for imports
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from main import load_objects, DATA_DIR, MODELS_DIR, DEFAULT_DATASET

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_environment():
    """Validate environment variables and paths."""
    logger.info("Validating environment configuration...")
    
    # Check PORT is set (required for Heroku)
    port = os.getenv("PORT")
    if not port:
        logger.warning("PORT environment variable not set, using default")
    else:
        try:
            port_int = int(port)
            if port_int < 1 or port_int > 65535:
                raise ValueError(f"Invalid port number: {port_int}")
            logger.info(f"✓ PORT={port_int}")
        except ValueError as e:
            logger.error(f"✗ Invalid PORT: {e}")
            return False
    
    # Check data directories exist
    for dir_path, name in [(DATA_DIR, "Data"), (MODELS_DIR, "Models")]:
        if not dir_path.exists():
            logger.error(f"✗ {name} directory missing: {dir_path}")
            return False
        logger.info(f"✓ {name} directory exists: {dir_path}")
    
    # Check dataset exists
    dataset_path = Path(os.getenv("DATASET_PATH", str(DEFAULT_DATASET)))
    if not dataset_path.exists():
        logger.error(f"✗ Dataset missing: {dataset_path}")
        return False
    logger.info(f"✓ Dataset exists: {dataset_path}")
    
    return True

def validate_models():
    """Validate that models can be loaded successfully."""
    logger.info("Validating model loading...")
    
    try:
        model_objects = load_objects()
        logger.info("✓ Models loaded successfully")
        
        # Validate model structure
        required_keys = ["mode", "preprocessor", "home_model", "away_model"]
        for key in required_keys:
            if key not in model_objects:
                logger.error(f"✗ Missing model component: {key}")
                return False
        
        logger.info(f"✓ Model mode: {model_objects['mode']}")
        logger.info(f"✓ Model types: {model_objects.get('model_types', {})}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        return False

def validate_dependencies():
    """Validate all required Python packages are available."""
    logger.info("Validating Python dependencies...")
    
    required_packages = [
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "ASGI server"),
        ("pandas", "Data manipulation"),
        ("numpy", "Numerical computing"),
        ("sklearn", "Machine learning"),
        ("joblib", "Model serialization"),
        ("lightgbm", "Gradient boosting"),
        ("tensorflow", "Deep learning"),
        ("nfl_data_py", "NFL data")
    ]
    
    for package, description in required_packages:
        try:
            __import__(package.replace("-", "_"))
            logger.info(f"✓ {package} ({description})")
        except ImportError as e:
            logger.error(f"✗ {package} missing: {e}")
            return False
    
    return True

def main():
    """Run all startup validations."""
    logger.info("🏈 NFL Prediction API - Production Startup Validation")
    logger.info("=" * 60)
    
    validations = [
        ("Environment", validate_environment),
        ("Dependencies", validate_dependencies), 
        ("Models", validate_models)
    ]
    
    for name, validation_func in validations:
        logger.info(f"\n--- {name} Validation ---")
        if not validation_func():
            logger.error(f"❌ {name} validation failed!")
            logger.error("🚨 Startup validation failed - fix issues before deployment")
            sys.exit(1)
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 All startup validations passed!")
    logger.info("🚀 Ready to start NFL Prediction API server")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()