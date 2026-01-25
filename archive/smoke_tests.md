# API Smoke Test Suite

Run these commands in your terminal (PowerShell) to verify the endpoints.

## 1. System Health

```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/status/overview
```

## 2. Team Assets (Fixed)

```powershell
curl http://127.0.0.1:8000/teams/KC
```

*Expected Output:* JSON with team colors and logo URLs.

## 3. Prediction (Fixed)

```powershell
$body = @{ home_team="KC"; away_team="BUF"; season=2024; week=11 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body
```

*Expected Output:* Prediction JSON or 503 if models aren't loaded (but NOT a 500 NameError).

## 4. Explanation (Client Wrapper Added)

```powershell
# Backend endpoint check
$body = @{ home_team="KC"; away_team="BUF"; season=2024; week=11 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict/explain" -ContentType "application/json" -Body $body
```
