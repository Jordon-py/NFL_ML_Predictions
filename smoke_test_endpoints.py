import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_endpoint(name, method, path, payload=None, params=None):
    url = f"{BASE_URL}{path}"
    print(f"Testing {name}: {method} {path}")
    try:
        if method == "GET":
            res = requests.get(url, params=params)
        else:
            res = requests.post(url, json=payload)
        
        print(f"  Status: {res.status_code}")
        if res.status_code < 400:
            try:
                data = res.json()
                print(f"  Response: {str(data)[:100]}...")
                return True, data
            except:
                print(f"  Response: (Non-JSON)")
                return True, res.text
        else:
            print(f"  Error: {res.text[:200]}")
            return False, res.text
    except Exception as e:
        print(f"  Exception: {e}")
        return False, str(e)

def run_suite():
    print("=" * 60)
    print("NFL PREDICTION APP: FULL SMOKE TEST SUITE")
    print("=" * 60)

    results = []

    # 1. Health
    results.append(test_endpoint("Health", "GET", "/health"))

    # 2. Teams
    results.append(test_endpoint("Teams List", "GET", "/teams"))

    # 3. Debug
    results.append(test_endpoint("Debug Info", "GET", "/debug"))

    # 4. Training Report
    results.append(test_endpoint("Training Report", "GET", "/report/training"))

    # 5. Calibration Report
    results.append(test_endpoint("Calibration Report", "GET", "/report/calibration"))

    # 6. Schedule (New)
    results.append(test_endpoint("Schedule (Canonical)", "GET", "/schedule/next-week"))

    # 7. Schedule (Old/Compat)
    results.append(test_endpoint("Schedule (Compat)", "GET", "/api/games/next-week"))

    # 8. Predict (POST)
    results.append(test_endpoint("Predict (POST)", "POST", "/predict", payload={
        "home_team": "TB", "away_team": "ATL", "season": 2025, "week": 15
    }))

    # 9. Predict (GET)
    results.append(test_endpoint("Predict (GET)", "GET", "/predict", params={
        "home_team": "TB", "away_team": "ATL", "season": 2025, "week": 15
    }))

    # 10. History
    results.append(test_endpoint("History", "GET", "/history", params={"limit": 5}))

    # 11. Status Overview
    results.append(test_endpoint("Status Overview", "GET", "/status/overview"))

    # 12. Batch Predict
    results.append(test_endpoint("Batch Predict", "GET", "/predict/next-week"))

    # 13. Train (Expected 501)
    results.append(test_endpoint("Train (Expect 501)", "POST", "/train"))

    print("=" * 60)
    success_count = sum(1 for r in results if r[0])
    print(f"Total: {len(results)}, Success: {success_count}, Failed: {len(results) - success_count}")
    print("=" * 60)

if __name__ == "__main__":
    run_suite()
