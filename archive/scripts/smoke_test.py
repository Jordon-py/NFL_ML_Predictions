import json
from urllib import request, error

BASE = "http://127.0.0.1:8000"

def fetch(method, path, body=None, headers=None, timeout=10):
    url = BASE + path
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/json")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in (headers or {}).items():
            req.add_header(k, v)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            raw = resp.read()
            if "application/json" in ctype:
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception:
                    payload = raw.decode("utf-8", errors="ignore")
            else:
                payload = raw.decode("utf-8", errors="ignore")
            return resp.status, payload
    except error.HTTPError as e:
        raw = e.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = raw.decode("utf-8", errors="ignore")
        return e.code, payload
    except Exception as e:
        return None, str(e)

results = {}

# 1) /health
status, payload = fetch("GET", "/health")
results["health"] = {"status": status, "payload": payload}

# 2) /schedule/next-week
status, payload = fetch("GET", "/schedule/next-week")
results["schedule"] = {"status": status, "count": (len(payload) if isinstance(payload, list) else None)}

# 3) /predict sample (KC vs BUF, 2025 W9)
body = {"home_team": "KC", "away_team": "BUF", "season": 2025, "week": 9}
status, payload = fetch("POST", "/predict", body)
prov = None
if isinstance(payload, dict):
    prov = payload.get("prediction_source")
    payload = {k: payload.get(k) for k in ("home_score","away_score","home_win_probability","prediction_source","mode")}
results["predict"] = {"status": status, "payload": payload, "prediction_source": prov}

print(json.dumps(results, indent=2))
