<#
.SYNOPSIS
  Interactive helper to set repository secrets for GitHub Actions using the GH CLI.

.NOTES
  - This script must be run locally on your machine where gh is authenticated (gh auth login).
  - It will prompt for secrets securely and call `gh secret set` for each value.
  - It will NOT echo secrets to stdout or write them into the repo.

USAGE
  pwsh ./scripts/set_github_secrets.ps1
#>

function Ensure-GhInstalled {
    $gh = Get-Command gh -ErrorAction SilentlyContinue
    if (-not $gh) {
        Write-Error "GitHub CLI (gh) is not installed or not in PATH. Install it from https://cli.github.com/ and run 'gh auth login' first."
        exit 1
    }
}

Ensure-GhInstalled

Write-Host "This script will set repository secrets using gh (GitHub CLI). You will be prompted to enter values."

$owner = Read-Host "GitHub repo owner (e.g. Jordon-py)"
$repo = Read-Host "GitHub repo name (e.g. NFL_ML_Predictions)"

function Set-SecretInteractive($name, $required=$true) {
    $val = Read-Host -AsSecureString "Enter value for secret '$name' (input hidden)"
    if (-not $val -and $required) {
        Write-Error "No value provided for required secret $name. Skipping."
        return
    }
    # Convert SecureString to plain text in memory (won't be written to disk)
    $ptr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($val)
    try {
        $plain = [System.Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
        gh secret set $name --body $plain --repo "$owner/$repo"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Set secret $name for $owner/$repo"
        } else {
            Write-Warning "gh secret set exited with code $LASTEXITCODE for $name"
        }
    } finally {
        [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) | Out-Null
    }
}

Set-SecretInteractive -name 'HEROKU_API_KEY'
Set-SecretInteractive -name 'HEROKU_APP'
Set-SecretInteractive -name 'VERCEL_TOKEN' -required:$false

Write-Host "Done. Verify the secrets in GitHub → Settings → Secrets and variables → Actions." -ForegroundColor Green
