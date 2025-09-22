# Heroku Deployment Guide

## 🚀 Quick Deployment to Heroku

### Prerequisites

- [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) installed
- Git repository initialized
- Heroku account

### Method 1: Standard Python Buildpack (Recommended)

This method uses the files: `Procfile`, `requirements.txt`, `runtime.txt`, and `app.json`.

```bash
# 1. Login to Heroku
heroku login

# 2. Create a new Heroku app
heroku create your-nfl-prediction-app

# 3. Set environment variables
heroku config:set LOG_LEVEL=INFO
heroku config:set ENVIRONMENT=production

# 4. Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# 5. Check logs
heroku logs --tail
```

### Method 2: Docker Container (Alternative)

Use this if you prefer Docker-based deployment with `heroku.yml` and `Dockerfile`.

```bash
# 1. Set stack to container
heroku stack:set container

# 2. Deploy with Docker
git add .
git commit -m "Deploy with Docker"
git push heroku main
```

## 📁 Deployment Files Explained

### Core Files (Method 1)

- **`Procfile`**: Tells Heroku how to run your app

  ```javascript
  web: gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120
  ```

- **`requirements.txt`**: Production dependencies only
- **`runtime.txt`**: Specifies Python version (`python-3.12.5`)
- **`app.json`**: Heroku app metadata and configuration

### Docker Files (Method 2)

- **`heroku.yml`**: Docker-based Heroku configuration
- **`Dockerfile`**: Container build instructions
- **`.dockerignore`**: Files to exclude from Docker build

### Configuration Files

- **`.env.example`**: Environment variables template
- **`requirements-dev.txt`**: Development dependencies

## ⚙️ Environment Variables

Set these in Heroku Config Vars:

```bash
heroku config:set LOG_LEVEL=INFO
heroku config:set ENVIRONMENT=production
heroku config:set CORS_ORIGINS=https://your-frontend.herokuapp.com
```

## 🔧 Post-Deployment

1. **Check app status:**

   ```bash
   heroku ps:scale web=1
   heroku open
   ```

2. **Test endpoints:**

   ```bash
   curl https://your-app.herokuapp.com/health
   curl https://your-app.herokuapp.com/
   ```

3. **Monitor logs:**

   ```bash
   heroku logs --tail --app your-app
   ```

## 📊 Performance Tuning

### Scaling

```bash
# Scale up workers
heroku ps:scale web=2

# Use performance dynos for production
heroku ps:resize web=standard-1x
```

### Add-ons (Optional)

```bash
# PostgreSQL (if needed later)
heroku addons:create heroku-postgresql:essential-0

# Redis (for caching)
heroku addons:create heroku-redis:mini
```

## 🚨 Troubleshooting

### Common Issues

1. **Build failures**: Check `requirements.txt` for conflicting versions
2. **App crashes**: Review logs with `heroku logs --tail`
3. **Slow startup**: Consider reducing TensorFlow model size
4. **Memory issues**: Upgrade to performance dynos

### Debug Commands

```bash
# Check dyno status
heroku ps

# Restart app
heroku restart

# Access Heroku bash
heroku run bash

# Check environment variables
heroku config
```

## 📈 Production Checklist

- [ ] Models and data files committed to repository
- [ ] Environment variables configured
- [ ] CORS origins set to production URLs
- [ ] Logging configured appropriately
- [ ] Health check endpoint working
- [ ] Performance monitoring setup
- [ ] Database backups (if using PostgreSQL)

## 🔄 CI/CD Integration

For automated deployments, connect your GitHub repository to Heroku:

1. Go to Heroku Dashboard → Your App → Deploy
2. Connect GitHub repository
3. Enable automatic deploys from `main` branch
4. Optional: Enable "Wait for CI to pass before deploy"

## 📝 Notes

- **Data Files**: Large CSV/model files are included in the repo. For production, consider using external storage (S3) for better performance.
- **Model Training**: The `/retrain` endpoint may timeout on Heroku's 30-second limit. Consider using background jobs for training.
- **Scaling**: Start with `eco` dynos for testing, upgrade to `basic` or `standard` for production load.
