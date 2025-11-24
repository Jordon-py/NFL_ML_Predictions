#!/usr/bin/env python3
import json
import csv
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
models_meta = BASE / 'backend' / 'models' / 'metadata.json'
csv_file = BASE / 'backend' / 'data' / 'game_features_20251123.csv'

if not models_meta.exists():
    print('metadata.json not found at', models_meta)
    raise SystemExit(1)
if not csv_file.exists():
    print('CSV file not found at', csv_file)
    raise SystemExit(1)

with open(models_meta, 'r', encoding='utf-8') as f:
    meta = json.load(f)

expected_numeric = meta.get('raw_feature_columns', {}).get('numeric', [])
expected_categorical = meta.get('raw_feature_columns', {}).get('categorical', [])
expected = set(expected_numeric + expected_categorical)

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    csv_cols = [c.strip() for c in header]

csv_set = set(csv_cols)

missing = sorted(list(expected - csv_set))
extra = sorted(list(csv_set - expected))

print('CSV columns count:', len(csv_cols))
print('Expected feature columns count (from metadata):', len(expected))
print('\nSample CSV columns (first 50):')
print(csv_cols[:50])
print('\nSample expected features (first 50):')
print((expected_numeric + expected_categorical)[:50])

print('\nMissing expected columns in CSV (count={}):'.format(len(missing)))
for c in missing[:50]:
    print('-', c)
if len(missing) > 50:
    print('... ({} more)'.format(len(missing)-50))

print('\nExtra columns in CSV not listed in metadata (count={}):'.format(len(extra)))
for c in extra[:100]:
    print('-', c)
if len(extra) > 100:
    print('... ({} more)'.format(len(extra)-100))

# Quick sanity checks
overlap = expected & csv_set
print('\nOverlap count:', len(overlap))

# Exit code 0
print('\nDone.')
