# NFL Prediction System - Production Deployment Guide

## 🚀 Quick Deploy to Heroku

### One-Click Deploy
[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)

### Manual Deploy Steps

1. **Clone and Setup**
   ```bash
   git clone https://github.com/Jordon-py/NFL_ML_Predictions.git
   cd NFL_ML_Predictions
   ```

2. **Pre-deployment Health Check**
   ```bash
   python health_check.py
   ```

3. **Deploy to Heroku**
   ```bash
   heroku create your-nfl-predictions-app
   git push heroku main
   ```

4. **Verify Deployment**
   ```bash
   curl https://your-nfl-predictions-app.herokuapp.com/health
   ```

## 🏗️ Architecture Overview

### Production Stack
- **Runtime**: Python 3.11.9
- **Web Server**: Gunicorn + Uvicorn workers
- **Framework**: FastAPI with async support
- **ML Models**: LightGBM + TensorFlow (Keras)
- **Data Pipeline**: nfl-data-py + pandas
- **Logging**: Structured JSON logging
- **Monitoring**: Built-in health checks

### Key Production Features
- ✅ **Zero-Fallback Architecture** - Fails fast on missing dependencies
- ✅ **Comprehensive Logging** - Request tracking with correlation IDs
- ✅ **Robust Error Handling** - Structured error responses
- ✅ **Model Validation** - Startup validation of all ML artifacts
- ✅ **Health Monitoring** - `/health` endpoint with detailed status
- ✅ **Request Middleware** - Automatic request/response logging
- ✅ **CORS Configuration** - Environment-based origin control

## 🔧 Configuration

### Required Environment Variables
```bash
# Server Configuration
PORT=8000
HOST=0.0.0.0

# Data Paths (optional - defaults provided)
DATASET_PATH=backend/data/Nfl_data_sorted.csv
SCHEDULE_PATH=backend/data/Nfl_schedule_2025_2026.csv

# Logging
LOG_LEVEL=INFO

# CORS (production)
CORS_ORIGINS=https://your-frontend-domain.com
```

### Heroku Config Vars
```bash
heroku config:set LOG_LEVEL=INFO
heroku config:set CORS_ORIGINS=https://your-domain.com
```

## 📊 API Endpoints

### Core Endpoints
- `GET /` - API information and status
- `GET /health` - Health check with model status
- `POST /predict` - Single game prediction
- `GET /predict/next-week` - Batch predictions for upcoming week
- `GET /schedule/next-week` - Next week's game schedule

### Admin Endpoints
- `POST /retrain` - Retrain models with existing data
- `POST /update_data` - Rebuild datasets and retrain models
- `GET /debug` - System debug information

### Example API Usage
```bash
# Health check
curl https://your-app.herokuapp.com/health

# Predict a game
curl -X POST https://your-app.herokuapp.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Kansas City Chiefs",
    "away_team": "Buffalo Bills", 
    "season": 2025,
    "week": 1
  }'
```

## 🛡️ Production Monitoring

### Health Check Validation
```bash
# Run comprehensive pre-deployment check
python health_check.py

# Expected output:
# ✓ All dependencies are available
# ✓ All model files are present  
# ✓ Dataset has all required columns
# ✓ Heroku config is correctly configured
# 🎉 All checks passed! Ready for production deployment.
```

### Application Logs
```bash
# View live logs
heroku logs --tail

# Search for errors
heroku logs --source app | grep ERROR
```

### Performance Monitoring
- **Request Timing**: All requests logged with duration
- **Error Tracking**: Structured error responses with correlation IDs
- **Model Performance**: Training metrics stored in metadata
- **Resource Usage**: Gunicorn worker management

## 🔄 Data Pipeline

### Dataset Updates
The system automatically handles:
1. **Historical Data**: NFL games from 2014-present
2. **Future Games**: Scheduled games for prediction
3. **Rolling Features**: Team performance metrics (3-game, 5-game windows)
4. **Data Validation**: Comprehensive checks for data integrity

### Model Retraining
```bash
# Retrain with existing data
curl -X POST https://your-app.herokuapp.com/retrain

# Update data and retrain
curl -X POST https://your-app.herokuapp.com/update_data
```

## 🚨 Troubleshooting

### Common Issues

**Startup Failures**
- Check model files exist in `backend/models/`
- Verify dataset at `backend/data/Nfl_data_sorted.csv`
- Run `python health_check.py` for detailed diagnostics

**Prediction Errors**
- Validate team names match expected format
- Check season/week parameters are reasonable
- Review API logs for detailed error messages

**Performance Issues**
- Monitor Heroku dyno metrics
- Check for memory usage spikes during model loading
- Consider scaling to larger dyno size if needed

### Debug Commands
```bash
# Check model metadata
curl https://your-app.herokuapp.com/debug

# Validate specific prediction
curl -X POST https://your-app.herokuapp.com/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "ARI", "away_team": "ATL", "season": 2025, "week": 1}'
```

## 📈 Scaling Considerations

### Heroku Dyno Types
- **Eco ($5/month)**: Development and light usage
- **Basic ($7/month)**: Production with light traffic  
- **Standard-1X ($25/month)**: Production with moderate traffic
- **Standard-2X ($50/month)**: High traffic or complex models

### Performance Optimization
- **Model Loading**: Models loaded once at startup
- **Request Caching**: Consider Redis for frequently requested predictions
- **Database**: Consider PostgreSQL for storing historical predictions
- **CDN**: Use Heroku's CDN for static assets

## 🔐 Security

### Production Hardening
- ✅ No hardcoded secrets or API keys
- ✅ Environment-based configuration
- ✅ CORS properly configured for production domains
- ✅ Input validation on all endpoints
- ✅ Structured error responses (no sensitive data leakage)
- ✅ Request logging for audit trails

### Best Practices
1. Set specific CORS origins (not `*`) in production
2. Use HTTPS-only for all API calls
3. Monitor logs for suspicious activity
4. Regularly update dependencies
5. Use Heroku's security features (private spaces, etc.)

---

🏈 **Ready for Production!** This system is designed to handle real-world NFL prediction workloads with robust error handling, comprehensive monitoring, and zero-downtime deployments.