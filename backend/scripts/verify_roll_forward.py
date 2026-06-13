
import sys
import pandas as pd
import joblib
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.services.inference_row import build_model_input_row

def verify():
    print("Verifying roll-forward logic...")
    
    # Load resources
    data_path = Path("backend/data/datasets/game_features_20260115.csv")
    models_dir = Path("backend/20260115/models")
    prep_path = models_dir / "preprocessor.joblib"
    
    if not data_path.exists():
        print(f"Dataset not found at {data_path}")
        return
        
    if not prep_path.exists():
        print(f"Preprocessor not found at {prep_path}")
        return
        
    df = pd.read_csv(data_path)
    preprocessor = joblib.load(prep_path)
    
    # Test Case 1: Past Game (Should have no missing values ideally, or low count)
    # 2025 Week 1: KC vs BAL
    row, source, info = build_model_input_row(
        dataset_df=df,
        preprocessor=preprocessor,
        season=2025,
        week=1,
        home_team="KC",
        away_team="BAL",
        debug=True
    )
    print(f"\nPast Game (2025 W1 KC vs BAL): Source={source}")
    print(f"Missing after impute: {info.get('missing_after_impute')}")
    
    # Test Case 2: Future/Hypothetical Game (Should trigger roll-forward)
    # 2026 Week 1 (Assuming dataset stops at 2025/2026 playoffs)
    row_f, source_f, info_f = build_model_input_row(
        dataset_df=df,
        preprocessor=preprocessor,
        season=2026,
        week=1,
        home_team="KC",
        away_team="BUF",
        debug=True
    )
    print(f"\nFuture Game (2026 W1 KC vs BUF): Source={source_f}")
    print(f"Missing after impute: {info_f.get('missing_after_impute')}")
    print(f"Prior counts (Home/Away): {info_f.get('missing_home_prior_count')} / {info_f.get('missing_away_prior_count')}")

if __name__ == "__main__":
    verify()
