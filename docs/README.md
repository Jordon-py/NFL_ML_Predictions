# Documentation Index

This directory contains comprehensive documentation for the NFL ML Predictions project.

## 📚 Quick Navigation

### CORS & API Configuration (NEW - 2025-10-13)

| Document | Description | Use Case |
|----------|-------------|----------|
| [CORS_CONFIGURATION_SUMMARY.md](CORS_CONFIGURATION_SUMMARY.md) | **Start here!** Complete overview of CORS fixes and architecture | Understanding what was done |
| [CORS_API_CONFIGURATION.md](CORS_API_CONFIGURATION.md) | Comprehensive CORS and API guide (300+ lines) | Detailed configuration reference |
| [API_CORS_CHECKLIST.md](API_CORS_CHECKLIST.md) | Verification checklist and deployment steps | Pre-deployment verification |
| [CORS_QUICK_REFERENCE.md](CORS_QUICK_REFERENCE.md) | Quick reference card with essential commands | Daily development tasks |

### Project Documentation

| Document | Description | Use Case |
|----------|-------------|----------|
| [report.md](report.md) | Comprehensive change log with function reference | Understanding project history |
| [enhancement_workflow.md](enhancement_workflow.md) | Enhancement workflow documentation | Development process |
| [session_completion_report.md](session_completion_report.md) | Session completion reports | Project status tracking |

---

## 🎯 Common Tasks

### I want to...

**...understand the CORS configuration**  
→ Read [CORS_CONFIGURATION_SUMMARY.md](CORS_CONFIGURATION_SUMMARY.md)

**...deploy the application**  
→ Follow [API_CORS_CHECKLIST.md](API_CORS_CHECKLIST.md)

**...quickly fix a CORS issue**  
→ Use [CORS_QUICK_REFERENCE.md](CORS_QUICK_REFERENCE.md)

**...understand all configuration details**  
→ Read [CORS_API_CONFIGURATION.md](CORS_API_CONFIGURATION.md)

**...see what changed recently**  
→ Check [report.md](report.md)

---

## 🔧 Tools & Scripts

| Tool | Location | Purpose |
|------|----------|---------|
| API Verification Script | `../scripts/verify_api_cors.py` | Automated CORS and API testing |
| Deployment Script | `../scripts/deploy.ps1` | Automated deployment to Heroku and Vercel |
| Dataset Builder | `../backend/build_csv_datasets.py` | Generate NFL game features dataset |

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                             │
│  Vercel: https://nfl-ml-predictions.vercel.app             │
│  Tech: React + Vite                                         │
│  Config: VITE_API_URL → Backend                            │
└─────────────────────────────────────────────────────────────┘
                            ↓ HTTP/JSON
┌─────────────────────────────────────────────────────────────┐
│                        BACKEND                              │
│  Heroku: https://nfl-predict-ecf5a5bd34fe.herokuapp.com    │
│  Tech: FastAPI + Python                                     │
│  Config: CORS_ORIGINS ← Frontend URLs                      │
│  Models: LightGBM (home/away score predictors)             │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ CORS Configuration Status

**Current Status:** ✅ Properly Configured

**Backend CORS_ORIGINS:**
```bash
http://localhost:3000,https://localhost:3000,https://nfl-ml-predictions.vercel.app,https://nfl-predict-frontend.vercel.app
```

**Frontend VITE_API_URL:**
```bash
# Production
https://nfl-predict-ecf5a5bd34fe.herokuapp.com

# Development
http://127.0.0.1:8000
```

---

## 🚀 Quick Start

### 1. Local Development

```bash
# Start backend
cd /path/to/NFL_ML_Predictions
uvicorn backend.main:app --reload --port 8000

# Start frontend (in new terminal)
cd frontend
npm run dev
```

### 2. Verify Configuration

```bash
# Run verification script
python scripts/verify_api_cors.py

# Test manually
curl http://localhost:8000/health
```

### 3. Deploy

```bash
# Use automated deployment script
pwsh -File scripts/deploy.ps1

# Or deploy manually (see API_CORS_CHECKLIST.md)
```

---

## 📝 Recent Changes (2025-10-13)

### Configuration Fixes
- ✅ Fixed `.env` CORS_ORIGINS (backend URL → frontend URLs)
- ✅ Created `backend/.env` with proper CORS config
- ✅ Fixed `frontend/.env.production` (removed comma-separated URL)

### Documentation Added
- ✅ Complete CORS guide (300+ lines)
- ✅ Verification checklist (250+ lines)
- ✅ Quick reference (100+ lines)
- ✅ Summary document (300+ lines)
- ✅ Verification script (350+ lines Python)

### Total Changes
- **Files Modified:** 3
- **Files Created:** 5
- **Documentation Lines:** 1,600+
- **Code Lines:** 350+

---

## 🆘 Troubleshooting

### CORS Error in Browser?
1. Check backend CORS_ORIGINS: `heroku config:get CORS_ORIGINS -a nfl-predict`
2. See [CORS_QUICK_REFERENCE.md](CORS_QUICK_REFERENCE.md) for fixes

### API Not Working?
1. Run verification: `python scripts/verify_api_cors.py`
2. See [API_CORS_CHECKLIST.md](API_CORS_CHECKLIST.md) for solutions

### Deployment Issues?
1. See [CORS_API_CONFIGURATION.md](CORS_API_CONFIGURATION.md) troubleshooting section
2. Check logs: `heroku logs --tail -a nfl-predict`

---

## 📞 Getting Help

1. **Start with:** [CORS_CONFIGURATION_SUMMARY.md](CORS_CONFIGURATION_SUMMARY.md) for overview
2. **Deep dive:** [CORS_API_CONFIGURATION.md](CORS_API_CONFIGURATION.md) for details
3. **Quick fix:** [CORS_QUICK_REFERENCE.md](CORS_QUICK_REFERENCE.md) for common issues
4. **Deploy:** [API_CORS_CHECKLIST.md](API_CORS_CHECKLIST.md) for step-by-step guide

---

## 📈 Project Status

**Completion Metrics:**
- CORS & API Configuration: 90% ✅
- Documentation: 95% ✅
- Deployment Readiness: 70% ✅
- Overall Project: 60% ⬆️

**Last Updated:** 2025-10-13  
**Documentation Version:** 1.0
