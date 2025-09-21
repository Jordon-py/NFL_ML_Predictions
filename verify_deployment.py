#!/usr/bin/env python3
"""
Heroku Deployment Verification Script
====================================
Run this before deploying to verify all files are properly configured.
"""

import json
import sys
from pathlib import Path


def check_file_exists(filepath: str, required: bool = True) -> bool:
    """Check if a file exists and report status."""
    path = Path(filepath)
    exists = path.exists()
    status = "✅" if exists else ("❌" if required else "⚠️")
    requirement = "Required" if required else "Optional"
    print(f"{status} {filepath} - {requirement}")
    return exists


def check_procfile():
    """Verify Procfile configuration."""
    print("\n📋 Checking Procfile...")
    if not check_file_exists("Procfile"):
        return False
    
    with open("Procfile", "r") as f:
        content = f.read().strip()
    
    if "gunicorn" in content and "uvicorn.workers.UvicornWorker" in content:
        print("✅ Procfile correctly configured for FastAPI")
        return True
    else:
        print("❌ Procfile may not be properly configured")
        return False


def check_requirements():
    """Verify requirements.txt has necessary packages."""
    print("\n📦 Checking requirements.txt...")
    if not check_file_exists("requirements.txt"):
        return False
    
    with open("requirements.txt", "r") as f:
        content = f.read()
    
    required_packages = ["fastapi", "gunicorn", "uvicorn", "pandas", "numpy", "scikit-learn"]
    missing = []
    
    for package in required_packages:
        if package not in content:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        return False
    else:
        print("✅ All required packages present")
        return True


def check_app_json():
    """Verify app.json is valid JSON."""
    print("\n⚙️ Checking app.json...")
    if not check_file_exists("app.json", required=False):
        return True
    
    try:
        with open("app.json", "r") as f:
            json.load(f)
        print("✅ app.json is valid JSON")
        return True
    except json.JSONDecodeError as e:
        print(f"❌ app.json has JSON syntax error: {e}")
        return False


def check_runtime():
    """Verify Python runtime version."""
    print("\n🐍 Checking runtime.txt...")
    if not check_file_exists("runtime.txt", required=False):
        print("⚠️ No runtime.txt - Heroku will use default Python version")
        return True
    
    with open("runtime.txt", "r") as f:
        content = f.read().strip()
    
    if content.startswith("python-"):
        print(f"✅ Python runtime specified: {content}")
        return True
    else:
        print("❌ Invalid runtime.txt format")
        return False


def main():
    """Run all verification checks."""
    print("🚀 Heroku Deployment Verification")
    print("=" * 35)
    
    checks = [
        check_procfile(),
        check_requirements(),
        check_runtime(),
        check_app_json(),
    ]
    
    # Check optional files
    print("\n📁 Optional deployment files:")
    check_file_exists(".env.example", required=False)
    check_file_exists("Dockerfile", required=False)
    check_file_exists("heroku.yml", required=False)
    check_file_exists(".dockerignore", required=False)
    check_file_exists("DEPLOYMENT.md", required=False)
    
    # Check data files
    print("\n📊 Data files:")
    check_file_exists("backend/data/Nfl_data_sorted.csv", required=False)
    check_file_exists("backend/models/metadata.json", required=False)
    
    print("\n" + "=" * 35)
    if all(checks):
        print("🎉 All required files are properly configured!")
        print("Ready for Heroku deployment.")
        sys.exit(0)
    else:
        print("❌ Some issues need to be fixed before deployment.")
        sys.exit(1)


if __name__ == "__main__":
    main()