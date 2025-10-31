param(
  # Git tag or branch, e.g. 'v3.13.0', 'v3.12.6', or 'main'
  [string]$Tag = "v3.13.0",
  # x64 | Win32 | ARM64 (maps to PCBuild output directories)
  [ValidateSet("x64","Win32","ARM64")]
  [string]$Arch = "x64",
  # Debug or Release
  [ValidateSet("Debug","Release")]
  [string]$Config = "Release",
  # Working directory for source + build
  [string]$WorkDir = "$env:USERPROFILE\src"
)

$ErrorActionPreference = "Stop"

function Have($exe) {
  return [bool](Get-Command $exe -ErrorAction SilentlyContinue)
}

function Ensure-Dir($path) {
  if (!(Test-Path $path)) { New-Item -ItemType Directory -Force -Path $path | Out-Null }
}

# Optional: quick VS Build Tools check
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (!(Test-Path $vswhere)) {
  Write-Warning "Visual Studio Build Tools not detected (vswhere missing)."
  Write-Warning "Install VS 2022 Build Tools with 'Desktop development with C++' + a Windows SDK."
  Write-Warning "The build may fail until that's installed."
}

Ensure-Dir $WorkDir
Set-Location $WorkDir

$repoUrl = "https://github.com/python/cpython"
$srcDir  = Join-Path $WorkDir "cpython-$Tag"

# Fetch source (git preferred; fallback to zip)
if (!(Test-Path $srcDir)) {
  if (Have git) {
    Write-Host "Cloning CPython $Tag..."
    git clone --depth 1 --branch $Tag $repoUrl $srcDir
  } else {
    $zipUrl = "$repoUrl/archive/refs/tags/$Tag.zip"
    $zipOut = Join-Path $WorkDir "cpython-$Tag.zip"
    Write-Host "Downloading $zipUrl ..."
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipOut
    Write-Host "Extracting archive..."
    Expand-Archive -Path $zipOut -DestinationPath $WorkDir -Force
    # GitHub tag zips are named without the leading 'v' in the folder
    $unzipped = Join-Path $WorkDir ("cpython-" + $Tag.TrimStart('v'))
    if (!(Test-Path $unzipped)) {
      throw "Expected extracted folder '$unzipped' not found."
    }
    Rename-Item -Path $unzipped -NewName ("cpython-" + $Tag)
  }
}

# Enter PCBuild and fetch third-party externals (OpenSSL, bz2, etc.)
$pcbuild = Join-Path $srcDir "PCBuild"
if (!(Test-Path $pcbuild)) { throw "PCBuild folder not found at $pcbuild" }
Set-Location $pcbuild

Write-Host "Fetching externals (first time only)..."
cmd /c ".\get_externals.bat"

# Build via the official script (auto-locates MSBuild via vswhere)
$buildArgs = "-e -p $Arch -c $Config"
Write-Host "Building: build.bat $buildArgs"
cmd /c ".\build.bat $buildArgs"

# Resolve output directory
$binDir = switch ($Arch) {
  "x64"   { "amd64" }
  "Win32" { "win32" }
  "ARM64" { "arm64" }
}
$pythonExe = Join-Path $pcbuild "$binDir\python.exe"

if (Test-Path $pythonExe) {
  Write-Host "`n✅ Success! Built Python at:`n$pythonExe"
  & $pythonExe -V
  Write-Host "`nTip: Add to PATH (current user):"
  Write-Host "[Environment]::SetEnvironmentVariable('Path', `"$env:Path;$($pcbuild)\$binDir`", 'User')"
} else {
  throw "Build completed, but '$pythonExe' not found. Review build logs above."
}
