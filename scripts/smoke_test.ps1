<#
File: scripts/smoke_test.ps1

Purpose:
  Production-friendly API smoke test for your FastAPI backend.
  It performs a sequence of requests and validates BOTH:
    (1) connectivity and HTTP status codes
    (2) JSON shapes + key fields you rely on in the React client

When something fails, you get:
  - which STEP failed
  - which URL/method failed
  - HTTP status/body (trimmed)
  - missing field/shape hints

Usage:
  pwsh ./scripts/smoke_test.ps1
  pwsh ./scripts/smoke_test.ps1 -BaseUrl "https://your-app.herokuapp.com"
  pwsh ./scripts/smoke_test.ps1 -VerboseBodies
#>

[CmdletBinding()]
param(
  [string]$BaseUrl = "http://127.0.0.1:8000",
  [int]$TimeoutSec = 15,
  [switch]$VerboseBodies
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step($label) {
  Write-Host ""
  Write-Host "==> $label" -ForegroundColor Cyan
}

function Trim-Body([string]$s, [int]$max = 1200) {
  if ([string]::IsNullOrWhiteSpace($s)) { return "" }
  $t = $s.Trim()
  if ($t.Length -le $max) { return $t }
  return $t.Substring(0, $max) + "`n... (trimmed)"
}

function Try-ParseJson([string]$raw) {
  if ([string]::IsNullOrWhiteSpace($raw)) { return $null }
  try { return $raw | ConvertFrom-Json -ErrorAction Stop }
  catch { return $null }
}

function Fail-Step([string]$step, $result, [string]$extraHint = "") {
  Write-Host "FAIL: $step" -ForegroundColor Red
  Write-Host "  Method: $($result.method)"
  Write-Host "  URL:    $($result.url)"
  if ($null -ne $result.status) {
    Write-Host "  Status: $($result.status)"
  }
  if ($result.error) {
    Write-Host "  Error:  $($result.error)"
  }
  if ($result.body) {
    Write-Host "  Body:`n$(Trim-Body $result.body)"
  }
  if ($extraHint) {
    Write-Host "  Hint:   $extraHint" -ForegroundColor Yellow
  }
  throw "Smoke test failed at: $step"
}

function Invoke-HttpJson {
  param(
    [Parameter(Mandatory=$true)][ValidateSet("GET","POST","OPTIONS")][string]$Method,
    [Parameter(Mandatory=$true)][string]$Url,
    [object]$Body = $null,
    [hashtable]$Headers = @{}
  )

  $payload = $null
  $contentType = $null

  if ($null -ne $Body) {
    $payload = ($Body | ConvertTo-Json -Depth 12)
    $contentType = "application/json"
  }

  try {
    $resp = Invoke-WebRequest `
      -Uri $Url `
      -Method $Method `
      -TimeoutSec $TimeoutSec `
      -Headers $Headers `
      -Body $payload `
      -ContentType $contentType `
      -UseBasicParsing

    return [pscustomobject]@{
      ok = $true
      method = $Method
      url = $Url
      status = [int]$resp.StatusCode
      headers = $resp.Headers
      body = [string]$resp.Content
      json = (Try-ParseJson $resp.Content)
      error = $null
    }
  }
  catch {
    $ex = $_.Exception
    $status = $null
    $raw = $null
    $hdrs = $null

    if ($ex.Response) {
      try {
        $status = [int]$ex.Response.StatusCode
        $hdrs = $ex.Response.Headers
        $stream = $ex.Response.GetResponseStream()
        if ($stream) {
          $reader = New-Object System.IO.StreamReader($stream)
          $raw = $reader.ReadToEnd()
        }
      } catch { }
    }

    return [pscustomobject]@{
      ok = $false
      method = $Method
      url = $Url
      status = $status
      headers = $hdrs
      body = [string]$raw
      json = (Try-ParseJson $raw)
      error = $ex.Message
    }
  }
}

function Ensure-Has($obj, [string[]]$keys, [string]$context) {
  foreach ($k in $keys) {
    if ($null -eq $obj) { return $false }
    if (-not ($obj.PSObject.Properties.Name -contains $k)) {
      Write-Host "  Missing field '$k' in $context" -ForegroundColor Yellow
      return $false
    }
  }
  return $true
}

function In-Range01([double]$n) { return ($n -ge 0.0 -and $n -le 1.0) }

Write-Host "NFL Backend Smoke Test" -ForegroundColor Green
Write-Host "BaseUrl: $BaseUrl"
Write-Host "Timeout: ${TimeoutSec}s"

$passed = 0

# STEP 1: /health
Write-Step "STEP 1: GET /health"
$health = Invoke-HttpJson -Method GET -Url "$BaseUrl/health"
if (-not $health.ok) {
  Fail-Step "GET /health" $health "Is the backend running? Try: uvicorn backend.main:app --reload"
}
if ($VerboseBodies) { Write-Host (Trim-Body $health.body) }
if (-not (Ensure-Has $health.json @("status") "/health response")) {
  Fail-Step "Validate /health JSON" $health "Expected JSON like { status: 'healthy' }"
}
$passed++

# STEP 2: /status/overview
Write-Step "STEP 2: GET /status/overview"
$overview = Invoke-HttpJson -Method GET -Url "$BaseUrl/status/overview"
if (-not $overview.ok) { Fail-Step "GET /status/overview" $overview "This powers StatsPage KPI cards." }
if ($VerboseBodies) { Write-Host (Trim-Body $overview.body) }
if (-not (Ensure-Has $overview.json @("health") "/status/overview response")) {
  Fail-Step "Validate /status/overview JSON" $overview "Expected { health: {...}, dataset: {...}, history: {...} }"
}
$passed++

# STEP 3: schedule (prefer /api/games/next-week)
Write-Step "STEP 3: GET schedule (prefer /api/games/next-week)"
$sched = Invoke-HttpJson -Method GET -Url "$BaseUrl/api/games/next-week"
if (-not $sched.ok) {
  Write-Host "  /api/games/next-week failed, trying fallback /schedule/next-week" -ForegroundColor Yellow
  $sched = Invoke-HttpJson -Method GET -Url "$BaseUrl/schedule/next-week"
}
if (-not $sched.ok) {
  Fail-Step "GET schedule" $sched "Schedule must return an array of games."
}
if ($VerboseBodies) { Write-Host (Trim-Body $sched.body) }

if (-not ($sched.json -is [System.Collections.IEnumerable])) {
  Fail-Step "Validate schedule shape" $sched "Expected JSON array from schedule endpoint."
}

$games = @($sched.json)
if ($games.Count -lt 1) {
  Fail-Step "Validate schedule non-empty" $sched "Schedule returned 0 games. Check dataset/schedule loader."
}
$g = $games[0]

$season = $g.season
$week = $g.week
$home = $g.home_abbr; if (-not $home) { $home = $g.home_team }
$away = $g.away_abbr; if (-not $away) { $away = $g.away_team }

if (-not $season -or -not $week -or -not $home -or -not $away) {
  Write-Host "  First schedule row missing one of: season, week, home, away" -ForegroundColor Yellow
  Write-Host "  Row:`n$(Trim-Body ($g | ConvertTo-Json -Depth 6))"
  throw "Cannot build predict payload from schedule row."
}
$passed++

# STEP 4: /predict
Write-Step "STEP 4: POST /predict (using first schedule game)"
$payload = @{
  season = [int]$season
  week = [int]$week
  home_team = [string]$home
  away_team = [string]$away
}

$pred = Invoke-HttpJson -Method POST -Url "$BaseUrl/predict" -Body $payload
if (-not $pred.ok) {
  Fail-Step "POST /predict" $pred "Check model loading and dataset lookup in backend/main.py"
}
if ($VerboseBodies) { Write-Host (Trim-Body $pred.body) }

$needed = @("home_score","away_score","home_win_probability","away_win_probability")
if (-not (Ensure-Has $pred.json $needed "/predict response")) {
  Fail-Step "Validate /predict fields" $pred "Expected keys: $($needed -join ', ')"
}

$hwp = [double]$pred.json.home_win_probability
$awp = [double]$pred.json.away_win_probability
if (-not (In-Range01 $hwp) -or -not (In-Range01 $awp)) {
  Fail-Step "Validate probabilities range" $pred "Probabilities must be in [0,1]."
}
$sum = $hwp + $awp
if ($sum -lt 0.98 -or $sum -gt 1.02) {
  Write-Host "  Warning: probabilities do not sum close to 1.0 (sum=$sum). Check calibration logic." -ForegroundColor Yellow
}
$passed++

# STEP 5: history (prefer /api/history)
Write-Step "STEP 5: GET history (prefer /api/history?limit=10)"
$hist = Invoke-HttpJson -Method GET -Url "$BaseUrl/api/history?limit=10"
if (-not $hist.ok) {
  Write-Host "  /api/history failed, trying fallback /history?limit=10" -ForegroundColor Yellow
  $hist = Invoke-HttpJson -Method GET -Url "$BaseUrl/history?limit=10"
}
if (-not $hist.ok) {
  Fail-Step "GET history" $hist "StatsPage uses this for HistoryChart."
}
if ($VerboseBodies) { Write-Host (Trim-Body $hist.body) }

if ($hist.json -is [System.Collections.IEnumerable]) {
  Write-Host "  History returned an array (legacy shape). Prefer /api/history for {entries,total,limit}." -ForegroundColor Yellow
} else {
  if (-not (Ensure-Has $hist.json @("entries","total") "history envelope")) {
    Fail-Step "Validate history envelope" $hist "Expected { entries: [...], total: <int>, limit: <int> }"
  }
}
$passed++

# STEP 6: CORS probe
Write-Step "STEP 6: OPTIONS /predict (CORS probe)"
$origin = "http://localhost:5173"
$corsHeaders = @{
  "Origin" = $origin
  "Access-Control-Request-Method" = "POST"
  "Access-Control-Request-Headers" = "content-type"
}
$cors = Invoke-HttpJson -Method OPTIONS -Url "$BaseUrl/predict" -Headers $corsHeaders
if (-not $cors.ok) {
  Fail-Step "OPTIONS /predict (CORS)" $cors "If this fails, browsers may block frontend calls."
}

$allowOrigin = $null
try { $allowOrigin = $cors.headers["Access-Control-Allow-Origin"] } catch { }

if (-not $allowOrigin) {
  Write-Host "  Warning: Access-Control-Allow-Origin header not found." -ForegroundColor Yellow
  Write-Host "  If browser calls fail, check FastAPI CORS middleware config." -ForegroundColor Yellow
} else {
  Write-Host "  CORS Allow-Origin: $allowOrigin"
}
$passed++

Write-Host ""
Write-Host "✅ Smoke test complete. Passed steps: $passed/6" -ForegroundColor Green
exit 0
