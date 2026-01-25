
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parents[2]))

from backend.main import _roll_forward_team_features, _dataset_means

class TestInference(unittest.TestCase):
    def test_roll_forward_features(self):
        # Setup mock dataframe
        df = pd.DataFrame({
            "season": [2024, 2024, 2024],
            "week": [1, 2, 3],
            "home_team": ["KC", "KC", "LV"],
            "away_team": ["BAL", "CIN", "KC"],
            "home_points_for": [20, 25, 10], 
            "away_points_for": [10, 20, 30],
            "is_completed": [True, True, True],
            "home_feature_A": [1.0, 1.2, 1.5], # KC is home W1, W2. Away W3.
            "away_feature_A": [0.8, 0.9, 1.1], # BAL W1, CIN W2, KC W3
        })
        
        numeric_cols = ["home_feature_A", "away_feature_A"]
        
        # Test KC (Home in prediction)
        features_kc = _roll_forward_team_features(
            df=df,
            team="KC",
            season=2024, 
            week=4,
            target_side="home",
            numeric_cols=numeric_cols
        )
        
        self.assertEqual(features_kc.get("home_feature_A"), 1.1)
        
        features_bal = _roll_forward_team_features(
            df=df,
            team="BAL",
            season=2024,
            week=4,
            target_side="away",
            numeric_cols=numeric_cols
        )
        
        self.assertEqual(features_bal.get("away_feature_A"), 0.8)

    def test_dataset_means(self):
        df = pd.DataFrame({
            "col1": [1, 2, 3, np.nan],
            "col2": [10, 20, 30, 40],
            "col3": ["a", "b", "c", "d"]
        })
        
        means = _dataset_means(df, ["col1", "col2", "col3"])
        
        self.assertEqual(means["col1"], 2.0)
        self.assertEqual(means["col2"], 25.0)
        self.assertNotIn("col3", means)

if __name__ == "__main__":
    unittest.main()
