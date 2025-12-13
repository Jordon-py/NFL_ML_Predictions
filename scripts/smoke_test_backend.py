import urllib.request, json, sys

def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            print('URL:', url)
            print('STATUS:', r.status)
            print(r.read().decode('utf-8'))
    except Exception as e:
        print('ERROR', url, e)

if __name__ == '__main__':
    urls = [
        'http://127.0.0.1:8000/health',
        'http://127.0.0.1:8000/status/overview',
        'http://127.0.0.1:8000/schedule/next-week',
        'http://127.0.0.1:8000/history'
        'http://127.0.0.1:8000/predict'
    ]

    for u in urls:
        fetch(u)

    # Try predict using first schedule item if available
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/schedule/next-week', timeout=5) as r:
            sched = json.loads(r.read().decode('utf-8'))
            if sched:
                g = sched[0]
                body = json.dumps({
                    'home_team': g.get('home_abbr') or g.get('home_team'),
                    'away_team': g.get('away_abbr') or g.get('away_team'),
                    'season': int(g.get('season')),
                    'week': int(g.get('week'))
                }).encode('utf-8')
                req = urllib.request.Request('http://127.0.0.1:8000/predict', data=body, method='POST', headers={'Content-Type':'application/json'})
                try:
                    with urllib.request.urlopen(req, timeout=10) as r2:
                        print('PREDICT STATUS', r2.status)
                        print(r2.read().decode('utf-8'))
                except Exception as e:
                    print('ERROR /predict', e)
            else:
                print('No schedule items to predict')
    except Exception as e:
        print('Error fetching schedule for predict test:', e)
