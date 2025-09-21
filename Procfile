web: python backend/startup.py && gunicorn backend.main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 300 --max-requests 1000 --max-requests-jitter 100 --preload
