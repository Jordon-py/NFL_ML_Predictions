import sklearn
import numpy as np
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

def diagnose():
    print(f"Scikit-learn version: {sklearn.__version__}")

    # 1. Test fresh imputer
    imp = SimpleImputer()
    X = np.array([[1, 2], [np.nan, 3]])
    imp.fit(X)
    print(f"Freshly fitted imputer has _fill_dtype: {hasattr(imp, '_fill_dtype')}")
    if hasattr(imp, '_fill_dtype'):
        print(f"Value of _fill_dtype: {imp._fill_dtype}")

    # 2. Test serialization
    dump_path = Path("temp_imputer.joblib")
    joblib.dump(imp, dump_path)
    loaded_imp = joblib.load(dump_path)
    print(f"Loaded imputer has _fill_dtype: {hasattr(loaded_imp, '_fill_dtype')}")

    if dump_path.exists():
        dump_path.unlink()

if __name__ == "__main__":
    diagnose()
