"""Quick test script to verify backend endpoints for schedule/history/predict.

Run with the project's Python (preferably the backend venv) so network permissions and SSL are consistent.
"""
import json
import sys
from urllib import request, error

BASE = 'http://127.0.0.1:8000'


def http_get(path):
    url = BASE + path
    req = request.Request(url, method='GET', headers={ 'Accept': 'application/json' })
    try:
        with request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
            text = raw.decode('utf-8') if raw else ''
            try:
                return json.loads(text)
            except Exception:
                return text
    except error.HTTPError as e:
        print(f'HTTPError GET {path}: {e.code} {e.reason}', file=sys.stderr)
        try:
            print(e.read().decode('utf-8'), file=sys.stderr)
        except Exception:
            pass
        return None
    except Exception as e:
        print(f'Error GET {path}: {e}', file=sys.stderr)
        return None


def http_post(path, payload):
    url = BASE + path
    data = json.dumps(payload).encode('utf-8')
    req = request.Request(url, data=data, method='POST', headers={ 'Content-Type': 'application/json', 'Accept': 'application/json' })
    try:
        with request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
            text = raw.decode('utf-8') if raw else ''
            try:
                return json.loads(text)
            except Exception:
                return text
    except error.HTTPError as e:
        print(f'HTTPError POST {path}: {e.code} {e.reason}', file=sys.stderr)
        try:
            print(e.read().decode('utf-8'), file=sys.stderr)
        except Exception:
            pass
        return None
    except Exception as e:
        print(f'Error POST {path}: {e}', file=sys.stderr)
        return None


def main():
    print('Testing backend endpoints on', BASE)

    print('\n1) GET /health')
    health = http_get('/health')
    print('->', json.dumps(health, indent=2) if health is not None else 'No response')

    print('\n2) GET /schedule/next-week')
    schedule = http_get('/schedule/next-week')
    if isinstance(schedule, list):
        print(f'-> schedule length: {len(schedule)}')
        if len(schedule) > 0:
            print('-> sample:', json.dumps(schedule[0], indent=2))
    else:
        print('-> schedule returned:', schedule)

    print('\n3) GET /history?limit=5')
    history = http_get('/history?limit=5')
    print('->', json.dumps(history, indent=2) if history is not None else 'No response')

    # If we have a schedule entry, try POST /predict using that game
    if isinstance(schedule, list) and schedule:
        g = schedule[0]
        home = g.get('home_team') or g.get('home_abbr') or g.get('home') or g.get('home_abbr')
        away = g.get('away_team') or g.get('away_abbr') or g.get('away')
        season = g.get('season') or g.get('season_num') or g.get('season_num')
        week = g.get('week') or g.get('week_num')
        if home and away and season and week:
            payload = {
                'home_team': str(home).strip().upper(),
                'away_team': str(away).strip().upper(),
                'season': int(season),
                'week': int(week)
            }
            print('\n4) POST /predict with payload:', payload)
            pred = http_post('/predict', payload)
            print('->', json.dumps(pred, indent=2) if pred is not None else 'No response')
        else:
            print('\n4) Skipping /predict test - insufficient schedule data')

if __name__ == '__main__':
    main()
