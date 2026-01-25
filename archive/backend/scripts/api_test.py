# ==========================================
# File: backend/scripts/api_test.py
# Role: Backend utility script.
# Input Data: CLI args and input files.
# Output Data: Reports, charts, or artifacts.
# Dependencies: requests, json
# Notes: Standalone execution.
# ==========================================

import requests
import json

def test_predict_endpoint():
    url = "http://127.0.0.1:8000/predict"
    payload = {
        "home_team": "Kansas City Chiefs",
        "away_team": "Baltimore Ravens",
        "season": 2025,
        "week": 1
    }
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Response JSON:")
            print(response.json())
        else:
            print("Error Response:")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_predict_endpoint()
