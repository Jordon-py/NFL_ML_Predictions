Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get | ConvertTo-Json -Depth 4
Invoke-RestMethod -Uri "http://127.0.0.1:8000/schedule/next-week" -Method Get | ConvertTo-Json -Depth 4
$payload = @{ home_team='CLE'; away_team='BAL'; season=2025; week=11 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Body $payload -ContentType 'application/json' | ConvertTo-Json -Depth 6
Invoke-RestMethod -Uri "http://127.0.0.1:8000/history?limit=10" -Method Get | ConvertTo-Json -Depth 6