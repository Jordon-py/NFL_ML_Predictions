import requests
import json

# Test the predict endpoint
url = "http://127.0.0.1:8001/predict"
payload = {
    "home_team": "KC",
    "away_team": "BUF",
    "season": 2025,
    "week": 16
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
