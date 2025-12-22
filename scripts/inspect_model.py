import joblib
from pathlib import Path
import json
BASE = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE / 'backend' / 'models'

home = joblib.load(MODELS_DIR / 'home_model.joblib')
print('Home model type:', type(home))
print('Has named_steps:', hasattr(home, 'named_steps'))
if hasattr(home, 'named_steps'):
    print('Named steps:', list(home.named_steps.keys()))
    pre = home.named_steps.get('pre')
    print('Preprocessor type:', type(pre))
    try:
        # If this is a ColumnTransformer, list transformers
        from sklearn.compose import ColumnTransformer
        if isinstance(pre, ColumnTransformer):
            print('ColumnTransformer transformers:')
            for name, trans, cols in pre.transformers:
                print('-', name, type(trans), cols)
    except Exception as e:
        print('Could not introspect preprocessor:', e)

print('\nfeature_names_in_ (home):')
print(getattr(home, 'feature_names_in_', None))

win = joblib.load(MODELS_DIR / 'win_clf_calibrated.joblib')
print('\nWin model type:', type(win))
print('Win model feature_names_in:', getattr(win, 'feature_names_in_', None))

# If metadata exists, show a few items
meta = MODELS_DIR / 'metadata.json'
if meta.exists():
    with open(meta, 'r') as f:
        data = json.load(f)
        print('\nMetadata keys:', list(data.keys())[:20])
        print('Metadata sample:', {k: data[k] for k in list(data.keys())[:5]})
