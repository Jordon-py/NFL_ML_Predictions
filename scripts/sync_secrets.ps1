# Syncs secrets from local .env to GitHub Repo Secrets
# Usage: ./scripts/sync_secrets.ps1

$envFiles = @(".env", "backend/.env")
$targetSecrets = @("VERCEL_TOKEN", "VERCEL_ORG_ID", "VERCEL_PROJECT_ID", "HEROKU_API_KEY", "HEROKU_APP_STAGING", "HEROKU_APP_PROD", "HEROKU_PIPELINE")

foreach ($file in $envFiles) {
    if (Test-Path $file) {
        Write-Host "Reading $file..."
        Get-Content $file | ForEach-Object {
            $line = $_.Trim()
            if ($line -match "^([^#=]+)=(.*)$") {
                $key = $matches[1]
                $val = $matches[2]
                
                if ($targetSecrets -contains $key) {
                    Write-Host "Setting secret: $key"
                    echo $val | gh secret set $key
                }
            }
        }
    }
}

Write-Host "Creating 'production' environment..."
try {
    gh api -X PUT /repos/:owner/:repo/environments/production
} catch {
    Write-Warning "Could not create environment (might already exist or permission issue)"
}

Write-Host "Done."
