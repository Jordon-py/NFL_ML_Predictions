import joblib
import pandas as pd
from pathlib import Path
import os
import sys

def verify():
    model_dir = Path("backend/models")
    print(f"Checking models in {model_dir.absolute()}...")

    files = [
        "preprocessor.joblib",
        "home_model.joblib",
        "away_model.joblib",
        "win_clf_calibrated.joblib"
    ]

    all_ok = True
    for f in files:
        path = model_dir / f
        if not path.exists():
            print(f"[MISSING] {f}")
            all_ok = False
            continue

        try:
            obj = joblib.load(path)
            print(f"[LOADED]  {f} - Type: {type(obj)}")

            # Specifically check SimpleImputer in preprocessor
            if f == "preprocessor.joblib":
                print("Inspecting preprocessor for _fill_dtype attribute...")
                # Dive into the ColumnTransformer
                for name, transformer, columns in obj.transformers_:
                    if transformer != 'remainder' and transformer != 'drop':
                        # transformer is usually a Pipeline
                        if hasattr(transformer, 'steps'):
                            for s_name, step in transformer.steps:
                                if "imputer" in s_name.lower():
                                    print(f"  Found imputer in {name}: {type(step)}")
                                    # This is the test - if it fails here, the error is real
                                    try:
                                        val = step._fill_dtype
                                        print(f"  [SUCCESS] {type(step).__name__} has _fill_dtype: {val}")
                                    except AttributeError:
                                        print(f"  [FAIL] {type(step).__name__} MISSING _fill_dtype")
                                        all_ok = False
        except Exception as e:
            print(f"[ERROR]   {f}: {e}")
            all_ok = False

    if all_ok:
        print("\n[VERDICT] All models loaded successfully and are compatible with scikit-learn 1.7.2")
        sys.exit(0)
    else:
        print("\n[VERDICT] Some models are still incompatible or missing.")
        sys.exit(1)

if __name__ == "__main__":
    verify()
