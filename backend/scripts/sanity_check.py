from backend.main import MODELS_DIR, load_models, load_dataset, DATASET_PATH, sanity_predict

print('MODELS_DIR=', MODELS_DIR)
print('DATASET_PATH=', DATASET_PATH)
try:
    models = load_models(MODELS_DIR)
    print('Loaded models keys:', list(models.keys()))
    df = load_dataset(DATASET_PATH, models)
    print('Dataset rows:', len(df), 'cols:', df.shape[1] if df is not None else None)
    try:
        sanity_predict(models, df)
        print('Sanity predict succeeded')
    except Exception as e:
        print('Sanity predict failed:', type(e).__name__, e)
except Exception as e:
    print('load_models failed:', type(e).__name__, e)
