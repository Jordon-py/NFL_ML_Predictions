import json
import urllib.request

url = 'http://127.0.0.1:8000/predict'
payload = {
    'home_team': 'HOU',
    'away_team': 'BUF',
    'season': 2025,
    'week': 12,
}
req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type':'application/json'}, method='POST')
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        print('STATUS', r.status)
        print(r.read().decode('utf-8'))
except Exception as e:
    print('ERROR', e)
    try:
        # If this is an HTTPError we can read the body for details
        import urllib.error
        if isinstance(e, urllib.error.HTTPError):
            body = e.read().decode('utf-8', errors='replace')
            print('ERROR BODY:', body)
    except Exception:
        pass
