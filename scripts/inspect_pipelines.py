import joblib
from pathlib import Path
MODELS_DIR = Path(__file__).resolve().parents[1] / 'backend' / 'models'
print('Models dir:', MODELS_DIR)
for p in sorted(MODELS_DIR.glob('*.joblib')):
    print('\n---', p.name)
    try:
        obj = joblib.load(p)
        print('Type:', type(obj))
        print('Has named_steps:', hasattr(obj, 'named_steps'))
        print('Has feature_names_in_:', hasattr(obj, 'feature_names_in_'))
        # If pipeline, list steps
        if hasattr(obj, 'named_steps'):
            print('Pipeline steps:', list(obj.named_steps.keys()))
            pre = obj.named_steps.get('pre') or obj.named_steps.get('prep') or obj.named_steps.get('preprocess')
            if pre is not None:
                print('Preprocessor type:', type(pre))
                try:
                    from sklearn.compose import ColumnTransformer
                    if isinstance(pre, ColumnTransformer):
                        print('ColumnTransformer transformers:')
                        for name, trans, cols in pre.transformers:
                            print('-', name, type(trans), cols)
                except Exception as e:
                    print('Could not introspect preprocessor:', e)
        # If a ColumnTransformer itself
        from sklearn.compose import ColumnTransformer
        if isinstance(obj, ColumnTransformer):
            print('Is ColumnTransformer: transformers ->')
            for name, trans, cols in obj.transformers:
                print('-', name, type(trans), cols)
    except Exception as e:
        print('Failed to load or introspect:', e)
print('\nDone')
