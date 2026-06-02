"""Audit model inference rows against a small hand-picked slate.

Data shape:
- Input CSV: one game per row with identity columns (`season`, `week`,
  `home_team`, `away_team`) plus the raw feature columns expected by the model
  bundle.
- Model input: single-row pandas DataFrames returned by
  `build_model_input_row`.
- Output: console-only diagnostics with selected raw features, transformed
  feature shape, predicted scores, and home-win probability.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.main_helpers import load_inference_bundle, load_dataset_df
from backend.services.inference_row import build_model_input_row


def audit():
    """Load a model bundle and print inference diagnostics for sample games."""
    print("Auditing Model Predictions...")
    models_dir = BACKEND_DIR / "20260115" / "models"
    
    if not models_dir.exists():
        models_dir = BACKEND_DIR / "models"
        
    bundle = load_inference_bundle(models_dir)
    data_path = BACKEND_DIR / "data" / "datasets" / "legacy" / "game_features_20260213.csv"
    if not data_path.exists():
        data_path = BACKEND_DIR / "data" / "datasets" / "game_features_20260531_clean.csv"
    df = load_dataset_df(data_path, bundle.raw_feature_columns)
    
    games = [
        ("NYG", "WAS"),
        ("MIA", "IND"),
        ("DET", "GB"),
        ("MIN", "CHI"),
        ("BAL", "BUF")
    ]
    
    season = 2025
    week = 1
    
    print("\n--- INFERENCE ROW AUDIT (2025 Week 1) ---\n")
    
    for home, away in games:
        print(f"\n======================================")
        print(f"GAME: {away} @ {home}")
        print(f"======================================")
        
        try:
            row, source, info = build_model_input_row(
                dataset_df=df,
                preprocessor=bundle.preprocessor,
                season=season,
                week=week,
                home_team=home,
                away_team=away,
                debug=True
            )
            
            raw_row = row
            
            print("\n[RAW FEATURES]")
            home_priors = [c for c in raw_row.columns if c.startswith("home_prior") or c.startswith("home_rolling")]
            away_priors = [c for c in raw_row.columns if c.startswith("away_prior") or c.startswith("away_rolling")]
            team_cols = [c for c in raw_row.columns if "team" in c.lower() or "abbr" in c.lower()]
            
            print("Team columns:")
            for c in team_cols:
                print(f"  {c}: {raw_row.iloc[0].get(c)}")
            
            print(f"Sample Home Priors ({len(home_priors)} total):")
            for c in home_priors[:5]:
                print(f"  {c}: {raw_row.iloc[0].get(c)}")
                
            print(f"Sample Away Priors ({len(away_priors)} total):")
            for c in away_priors[:5]:
                print(f"  {c}: {raw_row.iloc[0].get(c)}")
            
            if bundle.preprocessor:
                row_processed = bundle.preprocessor.transform(raw_row)
            else:
                row_processed = raw_row
                
            print("\n[PREPROCESSED FEATURES]")
            print(f"Row shape: {row_processed.shape}")
            
            # Predict
            home_score = bundle.home_model.predict(row_processed)[0]
            away_score = bundle.away_model.predict(row_processed)[0]
            
            home_win_prob = 0.5
            if bundle.hist_win_clf:
                if hasattr(bundle.hist_win_clf, "predict_proba"):
                    classes = bundle.hist_win_clf.classes_
                    proba = bundle.hist_win_clf.predict_proba(row_processed)[0]
                    if 1 in classes:
                        idx = list(classes).index(1)
                        home_win_prob = proba[idx]
                else:
                    win_pred = bundle.hist_win_clf.predict(row_processed)[0]
                    home_win_prob = 1.0 if win_pred == 1 else 0.0
                    
            print(f"\n[PREDICTIONS]")
            print(f"Home Score: {home_score:.2f}")
            print(f"Away Score: {away_score:.2f}")
            print(f"Home Win Prob: {home_win_prob:.2%}")
            
        except Exception as e:
            print(f"Error processing {away} @ {home}: {e}")

if __name__ == "__main__":
    audit()
