import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(name, method, path, payload=None):
    url = f"{BASE_URL}{path}"
    print(f"Testing {name}: {method} {url}")
    try:
        if method == "GET":
            res = requests.get(url)
        else:
            res = requests.post(url, json=payload)
        
        print(f"  Status: {res.status_code}")
        if res.status_code == 200:
            print(f"  Success: {json.dumps(res.json(), indent=2)[:200]}...")
            return True
        else:
            print(f"  Error: {res.text}")
            return False
    except Exception as e:
        print(f"  Failed: {e}")
        return False

def main():
    success = True
    
    # 1. Health
    success &= test_endpoint("Health", "GET", "/health")
    
    # 2. Schedule
    success &= test_endpoint("Schedule", "GET", "/schedule/next-week")
    
    # 3. Predict (Sample KC vs BUF)
    payload = {
        "home_team": "LAC",
        "away_team": "KC",
        "season": 2025,
        "week": 1
    }
    success &= test_endpoint("Predict", "POST", "/predict", payload)
    
    # 4. History
    success &= test_endpoint("History", "GET", "/history")
    
    # 5. Status Overview
    success &= test_endpoint("Status Overview", "GET", "/status/overview")
    
    # 6. Debug
    success &= test_endpoint("Debug", "GET", "/debug")
    
    if not success:
        print("\nSome tests failed!")
        sys.exit(1)
    else:
        print("\nAll endpoints resolved successfully.")

if __name__ == "__main__":
    main()
