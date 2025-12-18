"""
Test backend predictions directly
"""
import requests
import json

BASE_URL = "http://localhost:8000"

print("=" * 60)
print("TESTING BACKEND PREDICTIONS")
print("=" * 60)

# Test health
try:
    health = requests.get(f"{BASE_URL}/health")
    print(f"\n[1] Health Check: {health.status_code}")
    print(f"    Response: {health.json()}")
except Exception as e:
    print(f"    ERROR: {e}")

# Test a prediction
test_games = [
    {"home_team": "KC", "away_team": "LAC", "season": 2025, "week": 15},
    {"home_team": "CHI", "away_team": "CLE", "season": 2025, "week": 15},
    {"home_team": "TB", "away_team": "ATL", "season": 2025, "week": 15},
]

for i, game in enumerate(test_games, 1):
    print(f"\n[{i+1}] Testing: {game['away_team']} @ {game['home_team']} (Week {game['week']})")
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=game,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            pred = response.json()
            print(f"    Status: SUCCESS")
            print(f"    Home Score: {pred.get('home_score')}")
            print(f"    Away Score: {pred.get('away_score')}")
            print(f"    Point Diff: {pred.get('point_diff')}")
            print(f"    Home Win Prob: {pred.get('home_win_probability'):.3f}")
            print(f"    Source: {pred.get('prediction_source')}")
            print(f"    Win Classifier Used: {pred.get('win_classifier_used')}")
        else:
            print(f"    Status: FAILED ({response.status_code})")
            print(f"    Error: {response.text[:200]}")
    except Exception as e:
        print(f"    ERROR: {e}")

print("\n" + "=" * 60)
