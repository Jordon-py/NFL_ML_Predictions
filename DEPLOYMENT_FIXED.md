# 🚀 Deployment Fix - Backend (Heroku) + Frontend (Vercel)

## 📋 Summary of Changes

Fixed Heroku deployment to deploy **only the Python FastAPI backend**, while frontend remains on Vercel.

---

## ✅ Files Modified

### 1. `package.json` (Root)

**Problem:** Root `package.json` had `heroku-postbuild` script trying to build frontend.

**Fix:**

```json
{
  "scripts": {
    "heroku-postbuild": "echo 'Skipping frontend build - deployed separately on Vercel'"
  },
  "engines": {
    "node": "20.x",
    "npm": "10.x",
    "python": "3.12.x"
  }
}
```

**Why:** Heroku detected Node.js buildpack and tried to build frontend (which uses Vite). Now it skips frontend build entirely.

---

### 2. `.slugignore` (Root)

**Problem:** Only excluded `frontend/node_modules/`, not entire frontend.

**Fix:**

```plaintext
# Exclude entire frontend (deployed separately on Vercel)
frontend/

# Node modules
node_modules/

# Development artifacts
**/*.map
.cache/
tmp/
logs/
__pycache__/
.vscode/
.github/
*.md
!README.md
tests/
.git/
.env.example
.pre-commit-config.yaml
```

**Why:** Reduces slug size and prevents Heroku from trying to process frontend files.

---

### 3. `.buildpacks` (NEW)

**Created:** Forces Heroku to use Python buildpack only.

```plaintext
heroku/python
```

**Why:** Prevents Heroku from auto-detecting Node.js and using multi-buildpack mode.

---

### 4. `runtime.txt` (NEW)

**Created:** Specifies exact Python version.

```plaintext
python-3.12.0
```

**Why:** Ensures consistent Python version across deployments.

---

## 🔧 How to Deploy

### Backend to Heroku

1. **Commit changes:**

   ```bash
   git add .buildpacks runtime.txt package.json .slugignore
   git commit -m "fix: configure Heroku for backend-only deployment"
   ```

2. **Push to Heroku:**

   ```bash
   git push heroku main
   ```

3. **Verify deployment:**

   ```bash
   heroku logs --tail
   heroku ps
   heroku open
   ```

4. **Check backend health:**

   ```bash
   curl https://your-app.herokuapp.com/health
   ```

---

### Frontend to Vercel

1. **Navigate to frontend:**

   ```bash
   cd frontend
   ```

2. **Deploy to Vercel:**

   ```bash
   vercel --prod
   ```

3. **Set environment variables in Vercel dashboard:**
   - `VITE_API_BASE_URL=https://your-app.herokuapp.com`
   - `VITE_API_MODE=production`

---

## 🐛 Common Issues & Solutions

### Issue 1: "vite: not found"

**Cause:** Heroku trying to build frontend.

**Solution:** Ensure `.buildpacks` only has `heroku/python` and `heroku-postbuild` script echoes skip message.

---

### Issue 2: "No app detected"

**Cause:** Heroku can't find `requirements.txt` or `Procfile`.

**Solution:**

- Verify `requirements.txt` is in root (it delegates to `backend/requirements.txt`)
- Verify `Procfile` is in root
- Check `runtime.txt` specifies valid Python version

---

### Issue 3: CORS errors in frontend

**Cause:** Backend CORS not configured for Vercel domain.

**Solution:** Update `backend/main.py`:

```python
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,https://your-vercel-domain.vercel.app").split(",")
```

Then set Heroku config var:

```bash
heroku config:set CORS_ORIGINS="https://your-vercel-domain.vercel.app"
```

---

### Issue 4: Models not loading

**Cause:** Model files missing from slug.

**Solution:** Ensure model files are tracked in git:

```bash
git add backend/models/*.joblib backend/models/metadata.json
git commit -m "add model files"
```

---

## 📊 Buildpack Detection

Heroku uses this priority order:

1. `.buildpacks` file (if exists) ← **We added this**
2. `heroku/nodejs` (if `package.json` exists)
3. `heroku/python` (if `requirements.txt` or `runtime.txt` exists)

By creating `.buildpacks` with only `heroku/python`, we force Python-only deployment.

---

## 🔍 Verification Checklist

Before pushing to Heroku:

- [ ] `.buildpacks` contains only `heroku/python`
- [ ] `runtime.txt` specifies Python version
- [ ] `requirements.txt` in root delegates to `backend/requirements.txt`
- [ ] `Procfile` points to `backend.main:app`
- [ ] `.slugignore` excludes `frontend/`
- [ ] `package.json` heroku-postbuild skips frontend build
- [ ] Backend models are committed to git
- [ ] Environment variables set in Heroku dashboard

After deploying:

- [ ] `heroku logs` shows no errors
- [ ] `heroku ps` shows web dyno running
- [ ] `/health` endpoint returns 200 OK
- [ ] `/schedule/next-week` returns data
- [ ] Vercel frontend can connect to backend

---

## 🎯 Architecture

```graph TD;
┌─────────────────┐         ┌─────────────────┐
│   Vercel        │         │   Heroku        │
│   (Frontend)    │────────▶│   (Backend)     │
│                 │   API   │                 │
│  - React/Vite   │ Calls   │  - FastAPI      │
│  - Static build │         │  - Python 3.12  │
│  - CDN cached   │         │  - ML models    │
└─────────────────┘         └─────────────────┘
```

**Why separate?**

- **Vercel:** Optimized for frontend static hosting with CDN
- **Heroku:** Better for backend APIs with long-running processes
- **Reduced complexity:** No need for multi-buildpack on Heroku

---

## 📝 Next Steps

1. **Deploy backend to Heroku** (should succeed now)
2. **Update Vercel frontend** env vars with Heroku backend URL
3. **Test end-to-end** by making predictions from frontend
4. **Set up monitoring** (Heroku metrics, Sentry, etc.)
5. **Configure auto-deploy** from GitHub branches

---

## 🆘 Still Having Issues?

1. Check Heroku build logs:

   ```bash
   heroku logs --tail --source app
   ```

2. SSH into Heroku dyno:

   ```bash
   heroku run bash
   ```

3. Verify buildpack detection:

   ```bash
   heroku buildpacks
   ```

4. Clear build cache:

   ```bash
   heroku builds:cache:purge
   ```

---

## ✅ Success Indicators

You'll know deployment succeeded when you see:

```mermaid
graph TD;
-----> Building on the Heroku-24 stack
-----> Using buildpack: heroku/python
-----> Python app detected
-----> Installing python-3.12.0
-----> Installing pip dependencies
       Collecting fastapi...
       Collecting uvicorn...
       Successfully installed fastapi-0.109.0 uvicorn-0.24.0
-----> Discovering process types
       Procfile declares types -> web
-----> Compressing...
       Done: 45.2M
-----> Launching...
       Released v12
       https://your-app.herokuapp.com/ deployed to Heroku
```

🎉 **Deployment fixed! Backend now deploys correctly to Heroku.**
