# 🏈 NFL Prediction System - Deployment Guide

## Architecture Overview

- **Frontend**: React app hosted on Vercel
- **Backend**: FastAPI app hosted on Heroku
- **Communication**: Frontend makes API calls to backend with proper CORS configuration

## Quick Deployment

### 1. Backend (Heroku)

Your backend is configured to deploy to: `https://nfl-predict-ecf5a5bd34fe.herokuapp.com`

**Steps:**

```bash
# 1. Commit your changes
git add .
git commit -m "Configure for production deployment"

# 2. Push to Heroku
git push heroku main

# 3. Set environment variables (if not already set)
heroku config:set CORS_ORIGINS="http://localhost:3000,https://localhost:3000,https://YOUR_VERCEL_DOMAIN.vercel.app" --app nfl-predict-ecf5a5bd34fe

# 4. Check deployment
heroku logs --tail --app nfl-predict-ecf5a5bd34fe
```

### 2. Frontend (Vercel)

#### **Option A: Vercel Dashboard (Recommended)**

1. Go to [https://vercel.com](https://vercel.com)
2. Sign in with GitHub
3. Click "Add New Project"
4. Import your GitHub repository
5. Configure build settings:
   - **Framework Preset**: Create React App
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `build`
6. Set environment variables:
   - `VITE_API_URL` = `https://nfl-predict-ecf5a5bd34fe.herokuapp.com`
7. Deploy!

### **Option B: Vercel CLI**

```bash
# Install Vercel CLI
npm i -g vercel

# Go to frontend directory
cd frontend

# Deploy
vercel --prod
```

### 3. Update CORS Configuration

Once you get your Vercel domain (e.g., `https://your-app.vercel.app`), update the backend CORS:

```bash
heroku config:set CORS_ORIGINS="http://localhost:3000,https://localhost:3000,https://YOUR_ACTUAL_VERCEL_DOMAIN.vercel.app" --app nfl-predict-ecf5a5bd34fe
```

## Testing the Setup

### Backend Health Check

```bash
curl https://nfl-predict-ecf5a5bd34fe.herokuapp.com/health
```

Should return: `{"status":"healthy","mode":"models","reason":"models loaded"}`

### CORS Debug

```bash
curl https://nfl-predict-ecf5a5bd34fe.herokuapp.com/cors-debug
```

Should show your configured CORS origins.

### Frontend API Connection

1. Open your Vercel-deployed frontend
2. Try making a prediction
3. Check browser dev tools for any CORS errors

## Configuration Files Summary

### Backend (`backend/.env`)

```env
CORS_ORIGINS=http://localhost:3000,https://localhost:3000,https://your-vercel-domain.vercel.app
DATASET_PATH=backend/data/Nfl_data_sorted.csv
```

### Frontend (`frontend/.env.production`)

```env
VITE_API_URL=https://nfl-predict-ecf5a5bd34fe.herokuapp.com
```

### Vercel (`frontend/vercel.json`)

- Configured for SPA routing
- Build settings for Create React App

## Troubleshooting

### CORS Errors

- Check that your Vercel domain is in the `CORS_ORIGINS` environment variable
- Use the `/cors-debug` endpoint to verify configuration
- Ensure no trailing slashes in URLs

### Build Failures

- **Backend**: Check Heroku logs with `heroku logs --tail --app nfl-predict-ecf5a5bd34fe`
- **Frontend**: Check Vercel build logs in the Vercel dashboard

### API Connection Issues

- Verify the `VITE_API_URL` environment variable in Vercel
- Test the backend health endpoint directly
- Check browser network tab for failed requests

## Environment Variables Checklist

### Heroku (Backend)

- ✅ `CORS_ORIGINS` - Your frontend domains
- ✅ `DATASET_PATH` - Path to your dataset
- ✅ `NFL_API_KEY` - Your API key (if needed)

### Vercel (Frontend)

- ✅ `VITE_API_URL` - Your Heroku backend URL

## Security Notes

- CORS is configured to only allow your specific domains
- All API communication happens over HTTPS in production
- Environment variables are securely managed by each platform

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review Heroku and Vercel logs
3. Test individual components (backend health, frontend build)
4. Verify environment variables are correctly set
