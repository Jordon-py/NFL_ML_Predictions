import joblib
from pathlib import Path
MODELS_DIR = Path(__file__).resolve().parents[1] / 'backend' / 'models'
path = MODELS_DIR / 'preprocessor.joblib'
print('Preprocessor path exists:', path.exists(), path)
if not path.exists():
    raise SystemExit(1)
pre = joblib.load(path)
print('Preprocessor type:', type(pre))
# For ColumnTransformer, a fitted instance has attribute 'transformers_'
print('Has attribute transformers_?', hasattr(pre, 'transformers_'))
print('Has attribute feature_names_in_?', hasattr(pre, 'feature_names_in_'))
try:
    # try calling transform on a tiny dummy DataFrame with zero rows but correct columns
    import pandas as pd
    cols = []
    # If pre has feature_names_in_, use it
    if hasattr(pre, 'feature_names_in_'):
        cols = list(getattr(pre, 'feature_names_in_'))
    elif hasattr(pre, 'transformers'):
        # Try to extract column specs from transformers
        for name, transformer, colspec in getattr(pre, 'transformers'):
            if isinstance(colspec, (list, tuple)):
                cols.extend(list(colspec))
    if not cols:
        print('No feature names available to test transform; skipping transform test.')
    else:
        df = pd.DataFrame(columns=cols)
        print('Attempting transform on empty DataFrame with cols:', len(cols))
        try:
            out = pre.transform(df)
            print('Transform succeeded on empty dataframe, output shape:', getattr(out, 'shape', 'unknown'))
        except Exception as e:
            print('Transform raised:', type(e), e)
except Exception as ee:
    print('Error while testing transform:', ee)
