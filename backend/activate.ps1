# Activate the Python virtual environment in backend
# Usage: .\activate.ps1

$venvPath = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"

if (Test-Path $venvPath) {
    Write-Host "✅ Activating virtual environment..." -ForegroundColor Green
    & $venvPath
    Write-Host "✅ Virtual environment activated!" -ForegroundColor Green
    Write-Host "📦 Python version:" -ForegroundColor Cyan
    python --version
    Write-Host "`n📚 Key packages installed:" -ForegroundColor Cyan
    python -m pip list | Select-String "pandas|numpy|fastapi|uvicorn|scikit|lightgbm|nfl-data"
} else {
    Write-Host "❌ Virtual environment not found at $venvPath" -ForegroundColor Red
}
