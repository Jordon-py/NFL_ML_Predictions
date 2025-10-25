import json
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

backend = 'http://127.0.0.1:8000'
proxy = 'http://localhost:3000'

def get(url):
    try:
        with urlopen(url, timeout=10) as resp:
            data = resp.read().decode('utf-8')
            print(f'GET {url} -> {len(data)} bytes')
            print(data[:1000])
    except HTTPError as e:
        print(f'HTTPError GET {url}: {e.code} - {e.reason}')
        try:
            print(e.read().decode())
        except Exception:
            pass
    except URLError as e:
        print(f'URLError GET {url}: {e}')
    except Exception as e:
        print(f'Error GET {url}: {e}')


def post(url, payload):
    try:
        data = json.dumps(payload).encode('utf-8')
        req = Request(url, data=data, headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=15) as resp:
            out = resp.read().decode('utf-8')
            print(f'POST {url} -> {len(out)} bytes')
            print(out[:1000])
    except HTTPError as e:
        print(f'HTTPError POST {url}: {e.code} - {e.reason}')
        try:
            print(e.read().decode())
        except Exception:
            pass
    except URLError as e:
        print(f'URLError POST {url}: {e}')
    except Exception as e:
        print(f'Error POST {url}: {e}')


if __name__ == '__main__':
    print('Checking backend schedule...')
    get(backend + '/schedule/next-week')
    print('\nChecking backend predict...')
    payload = {"home_team": "NYG", "away_team": "PHI", "season": 2025, "week": 6}
    post(backend + '/predict', payload)
    print('\nChecking frontend proxy schedule (localhost:3000)...')
    get(proxy + '/schedule/next-week')
