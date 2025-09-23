# Heroku Deployment Troubleshooting Guide

## 🚨 Common Issues & Solutions

### Issue 1: Dependency Conflicts (numpy version errors)

**Symptoms:**

```
ERROR: Cannot install numpy==1.19.3 
pip._vendor.pyproject_hooks._impl.BackendUnavailable: Cannot import 'setuptools.build_meta'
```

**Root Cause:** Version conflicts between TensorFlow, numpy, and Python version

**Solutions:**

1. **Use `requirements-lock.txt` (Recommended):**

   ```bash
   # Rename current requirements.txt and use locked versions
   mv requirements.txt requirements-flexible.txt
   mv requirements-lock.txt requirements.txt
   git commit -am "Use locked dependency versions for Heroku"
   git push heroku main
   ```

2. **Force tensorflow-cpu instead of tensorflow:**

   ```bash
   # In requirements.txt, change:
   # tensorflow>=2.13.0,<2.16.0
   # to:
   tensorflow-cpu==2.15.0
   ```

3. **Use Python 3.10 instead of 3.11:**

   ```bash
   # In runtime.txt:
   python-3.10.12
   ```

### Issue 2: Build Timeout

**Symptoms:**

```
Build timed out (exceeded 15 minutes)
```

**Solutions:**

1. **Use precompiled packages only:**

   ```bash
   # Add to pip.conf:
   [install]
   only-binary = :all:
   ```

2. **Reduce worker count in Procfile:**

   ```
   web: gunicorn backend.main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 300
   ```

### Issue 3: Memory Issues

**Symptoms:**

```
Process exceeded memory quota
R14 (Memory quota exceeded)
```

**Solutions:**

1. **Upgrade dyno type:**

   ```bash
   heroku ps:scale web=1:standard-1x
   ```

2. **Optimize model loading:**

   ```python
   # In main.py, add memory optimization
   import gc
   import tensorflow as tf
   
   # After model loading
   tf.keras.backend.clear_session()
   gc.collect()
   ```

### Issue 4: CORS Errors

**Symptoms:**

```
Access to fetch blocked by CORS policy
```

**Solutions:**

1. **Set CORS environment variable:**

   ```bash
   heroku config:set CORS_ORIGINS="https://your-frontend.herokuapp.com,http://localhost:3000"
   ```

2. **For development, use wildcard:**

   ```bash
   heroku config:set CORS_ORIGINS="*"
   ```

## 🔧 Quick Fixes

### Emergency Deployment (Minimal Dependencies)

If all else fails, use this minimal requirements.txt:

```plaintext
fastapi==0.104.1
uvicorn[standard]==0.24.0
gunicorn==21.2.0
numpy==1.24.3
pandas==2.1.4
scikit-learn==1.3.2
joblib==1.3.2
pydantic==2.5.2
```

Then comment out TensorFlow-dependent code temporarily.

### Test Locally First

```bash
# Create a test environment
python -m venv heroku-test
source heroku-test/bin/activate  # or heroku-test\Scripts\activate on Windows

# Install from your requirements.txt
pip install -r requirements.txt

# Test the app
python -m uvicorn backend.main:app --port 8000
```

### Heroku Build Logs

```bash
# View detailed build logs
heroku logs --tail --app your-app-name

# Check specific processes
heroku ps --app your-app-name

# Restart if needed
heroku restart --app your-app-name
```

## 📋 Pre-deployment Checklist

- [ ] Python version is 3.10 or 3.11 (avoid 3.12+)
- [ ] All dependencies have compatible version ranges
- [ ] tensorflow-cpu is used instead of tensorflow
- [ ] setuptools, wheel, pip are pinned in requirements.txt
- [ ] CORS_ORIGINS is set as config var
- [ ] Models and data files are committed to repo
- [ ] Local testing passes with identical requirements

## 🆘 Last Resort Solutions

1. **Use Docker deployment (heroku.yml):**

   ```bash
   heroku stack:set container
   ```

2. **Split into microservices:**
   - Deploy API without ML models
   - Use separate model service

3. **Use external model hosting:**
   - AWS SageMaker, Google AI Platform
   - Call external API from your Heroku app

## 📞 Getting Help

If issues persist:

1. Check [Heroku Dev Center](https://devcenter.heroku.com/)
2. Review [Python buildpack docs](https://github.com/heroku/heroku-buildpack-python)
3. Search [Stack Overflow](https://stackoverflow.com/questions/tagged/heroku) with error messages
4. Consider using Heroku's [Container Registry](https://devcenter.heroku.com/articles/container-registry-and-runtime) for complex builds
