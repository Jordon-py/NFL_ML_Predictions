#!/bin/bash
# Deployment script for NFL Prediction System
# This script helps deploy both frontend (Vercel) and backend (Heroku)

echo "🏈 NFL Prediction System Deployment Helper"
echo "=========================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check required tools
echo "Checking required tools..."
if ! command_exists git; then
    echo "❌ Git not found. Please install Git."
    exit 1
fi

if ! command_exists heroku; then
    echo "❌ Heroku CLI not found. Please install Heroku CLI."
    exit 1
fi

if ! command_exists vercel; then
    echo "⚠️ Vercel CLI not found. Install with: npm i -g vercel"
fi

echo "✅ Tools check complete"
echo ""

# Backend Deployment (Heroku)
echo "🚀 Backend Deployment (Heroku)"
echo "------------------------------"

# Check if Heroku app exists
HEROKU_APP="nfl-predict-ecf5a5bd34fe"
if heroku apps:info $HEROKU_APP >/dev/null 2>&1; then
    echo "✅ Heroku app '$HEROKU_APP' found"
else
    echo "❌ Heroku app '$HEROKU_APP' not found or not accessible"
    echo "Please check your Heroku app name and permissions"
    exit 1
fi

# Set Heroku environment variables
echo "Setting Heroku environment variables..."
heroku config:set CORS_ORIGINS="http://localhost:3000,https://localhost:3000,https://nfl-predict-frontend.vercel.app" --app $HEROKU_APP
heroku config:set DATASET_PATH="backend/data/Nfl_data_sorted.csv" --app $HEROKU_APP

echo "✅ Heroku environment variables set"
echo ""

# Frontend Deployment Instructions
echo "📱 Frontend Deployment (Vercel)"
echo "-------------------------------"
echo "To deploy your frontend to Vercel:"
echo ""
echo "1. Go to your frontend directory:"
echo "   cd frontend"
echo ""
echo "2. If using Vercel CLI:"
echo "   vercel --prod"
echo ""
echo "3. If using Vercel dashboard:"
echo "   - Go to https://vercel.com"
echo "   - Import your GitHub repository"
echo "   - Set build settings:"
echo "     - Framework Preset: Create React App"
echo "     - Root Directory: frontend"
echo "     - Build Command: npm run build"
echo "     - Output Directory: build"
echo ""
echo "4. Set environment variables in Vercel:"
echo "   VITE_API_URL=https://nfl-predict-ecf5a5bd34fe.herokuapp.com"
echo ""

# Git status check
echo "📦 Git Status"
echo "-------------"
git status --short

echo ""
echo "🎯 Next Steps:"
echo "1. Commit your changes: git add . && git commit -m 'Configure for production deployment'"
echo "2. Push to Heroku: git push heroku main"
echo "3. Deploy frontend to Vercel following instructions above"
echo "4. Update CORS_ORIGINS on Heroku with your Vercel domain"
echo ""
echo "📊 Useful Commands:"
echo "- Check Heroku logs: heroku logs --tail --app $HEROKU_APP"
echo "- Check Heroku config: heroku config --app $HEROKU_APP"
echo "- Test backend health: curl https://nfl-predict-ecf5a5bd34fe.herokuapp.com/health"
echo "- Test CORS debug: curl https://nfl-predict-ecf5a5bd34fe.herokuapp.com/cors-debug"