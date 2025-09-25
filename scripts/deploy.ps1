<#
.SYNOPSIS
  One-command deploy of backend (Heroku) and frontend (Vercel).

.DESCRIPTION
  - Ensures Heroku CORS origins include your Vercel domains and localhost
  - Installs and builds the frontend
  - Pushes changes to GitHub (main and master)
  - Deploys backend to Heroku
  - Deploys frontend to Vercel using root vercel.json
  - Verifies backend /health and prints the deployed frontend URL

.PREREQS
  - Logged in to Heroku CLI (heroku login)
  - Logged in to Vercel CLI (vercel login)
  - Git remotes configured for origin and heroku

.USAGE
  pwsh -File scripts/deploy.ps1
#>

param(
  [string]$HerokuApp = "nfl-predict",
  [string]$VercelProject = "nfl-ml-predictions",
  [string]$VercelOrg = "christopher-jordons-projects",
  [string]$ApiBaseUrl = "https://nfl-predict-ecf5a5bd34fe.herokuapp.com",
  [string]$VercelProdDomain = "https://nfl-ml-predictions.vercel.app"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-Cli($name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    throw "Required CLI not found: $name"
  }
}

Write-Host "[1/8] Checking CLIs..." -ForegroundColor Cyan
Assert-Cli git
Assert-Cli heroku
Assert-Cli vercel
Assert-Cli npm

Write-Host "[2/8] Ensuring Heroku CORS_ORIGINS includes Vercel + localhost..." -ForegroundColor Cyan
$origins = @(
  'http://localhost:3000',
  'https://localhost:3000',
  'https://nfl-predict-frontend.vercel.app',
  $VercelProdDomain
) -join ','
heroku config:set CORS_ORIGINS="$origins" -a $HerokuApp | Out-Host

Write-Host "[3/8] Installing frontend dependencies..." -ForegroundColor Cyan
npm install --prefix frontend | Out-Host

Write-Host "[4/8] Building frontend..." -ForegroundColor Cyan
npm run build --prefix frontend | Out-Host

Write-Host "[5/8] Committing any pending changes..." -ForegroundColor Cyan
git add -A
try {
  git commit -m "chore(deploy): automated deploy run" | Out-Null
} catch {
  Write-Host "No changes to commit." -ForegroundColor DarkGray
}

Write-Host "[6/8] Pushing to GitHub (main -> origin, and mirror to master)..." -ForegroundColor Cyan
git push origin main | Out-Host
git push origin main:master | Out-Host

Write-Host "[7/8] Deploying backend to Heroku..." -ForegroundColor Cyan
git push heroku main | Out-Host

Write-Host "[8/8] Deploying frontend to Vercel..." -ForegroundColor Cyan
# Use repo root vercel.json; suppress prompts
$env:VITE_API_URL = $ApiBaseUrl
vercel --prod --yes | Tee-Object -Variable vercelOut | Out-Host

# Try to parse the production URL from the captured output
$vercelText = ($vercelOut | Out-String)
if ($vercelText -match 'Production:\s+(https?://\S+)') {
  $frontendUrl = $matches[1]
} else {
  $frontendUrl = "$VercelProject (see Vercel dashboard)"
}

Write-Host "\n=== Verification ===" -ForegroundColor Green
Write-Host "Backend health: " -NoNewline
try {
  $health = (curl -sS "$ApiBaseUrl/health").Content
  Write-Host $health
} catch { Write-Host "failed" -ForegroundColor Red }

Write-Host "CORS debug: " -NoNewline
try {
  $cors = (curl -sS "$ApiBaseUrl/cors-debug").Content
  Write-Host $cors
} catch { Write-Host "failed" -ForegroundColor Red }

Write-Host "Frontend URL: $frontendUrl" -ForegroundColor Yellow

Write-Host "\nDeploy complete." -ForegroundColor Green
