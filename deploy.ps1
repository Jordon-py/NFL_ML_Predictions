# NFL Prediction System Deployment Helper (PowerShell)
# This script helps deploy both frontend (Vercel) and backend (Heroku)

Write-Host "🏈 NFL Prediction System Deployment Helper" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# Function to check if command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Check required tools
Write-Host "Checking required tools..."
if (-not (Test-Command "git")) {
    Write-Host "❌ Git not found. Please install Git." -ForegroundColor Red
    exit 1
}

if (-not (Test-Command "heroku")) {
    Write-Host "❌ Heroku CLI not found. Please install Heroku CLI." -ForegroundColor Red
    exit 1
}

if (-not (Test-Command "vercel")) {
    Write-Host "⚠️ Vercel CLI not found. Install with: npm i -g vercel" -ForegroundColor Yellow
}

Write-Host "✅ Tools check complete" -ForegroundColor Green
Write-Host ""

# Backend Deployment (Heroku)
Write-Host "🚀 Backend Deployment (Heroku)" -ForegroundColor Cyan
Write-Host "------------------------------" -ForegroundColor Cyan

# Check if Heroku app exists
$HEROKU_APP = "nfl-predict-ecf5a5bd34fe"
try {
    heroku apps:info $HEROKU_APP 2>$null | Out-Null
    Write-Host "✅ Heroku app '$HEROKU_APP' found" -ForegroundColor Green
} catch {
    Write-Host "❌ Heroku app '$HEROKU_APP' not found or not accessible" -ForegroundColor Red
    Write-Host "Please check your Heroku app name and permissions" -ForegroundColor Red
    exit 1
}

# Set Heroku environment variables
Write-Host "Setting Heroku environment variables..."
heroku config:set CORS_ORIGINS="http://localhost:3000,https://localhost:3000,https://nfl-predict-frontend.vercel.app" --app $HEROKU_APP
heroku config:set DATASET_PATH="backend/data/Nfl_data_sorted.csv" --app $HEROKU_APP

Write-Host "✅ Heroku environment variables set" -ForegroundColor Green
Write-Host ""

# Frontend Deployment Instructions
Write-Host "📱 Frontend Deployment (Vercel)" -ForegroundColor Magenta
Write-Host "-------------------------------" -ForegroundColor Magenta
Write-Host "To deploy your frontend to Vercel:"
Write-Host ""
Write-Host "1. Go to your frontend directory:"
Write-Host "   cd frontend" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. If using Vercel CLI:"
Write-Host "   vercel --prod" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. If using Vercel dashboard:"
Write-Host "   - Go to https://vercel.com"
Write-Host "   - Import your GitHub repository"
Write-Host "   - Set build settings:"
Write-Host "     - Framework Preset: Create React App"
Write-Host "     - Root Directory: frontend"
Write-Host "     - Build Command: npm run build"
Write-Host "     - Output Directory: build"
Write-Host ""
Write-Host "4. Set environment variables in Vercel:"
Write-Host "   VITE_API_URL=https://nfl-predict-ecf5a5bd34fe.herokuapp.com" -ForegroundColor Yellow
Write-Host ""

# Git status check
Write-Host "📦 Git Status" -ForegroundColor Blue
Write-Host "-------------" -ForegroundColor Blue
git status --short

Write-Host ""
Write-Host "🎯 Next Steps:" -ForegroundColor Green
Write-Host "1. Commit your changes: git add . && git commit -m 'Configure for production deployment'"
Write-Host "2. Push to Heroku: git push heroku main"
Write-Host "3. Deploy frontend to Vercel following instructions above"
Write-Host "4. Update CORS_ORIGINS on Heroku with your actual Vercel domain"
Write-Host ""
Write-Host "📊 Useful Commands:" -ForegroundColor Blue
Write-Host "- Check Heroku logs: heroku logs --tail --app $HEROKU_APP"
Write-Host "- Check Heroku config: heroku config --app $HEROKU_APP" 
Write-Host "- Test backend health: curl https://nfl-predict-ecf5a5bd34fe.herokuapp.com/health"
Write-Host "- Test CORS debug: curl https://nfl-predict-ecf5a5bd34fe.herokuapp.com/cors-debug"