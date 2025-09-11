import json
import subprocess
import sys

# Test data
test_data = {
    "home_team": "KC",
    "away_team": "BUF",
    "season": 2025,
    "week": 1
}

# Convert to JSON string
json_data = json.dumps(test_data)

# Make curl request
curl_cmd = [
    "curl", "-X", "POST", "http://127.0.0.1:8001/predict",
    "-H", "Content-Type: application/json",
    "-d", json_data
]

print(f"Running: {' '.join(curl_cmd)}")
result = subprocess.run(curl_cmd, capture_output=True, text=True)
print(f"Status Code: {result.returncode}")
print(f"Stdout: {result.stdout}")
print(f"Stderr: {result.stderr}")
