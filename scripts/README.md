# Smoke screenshot helper (Puppeteer)

This small script captures three screenshots of the locally-served frontend at common viewports:

- mobile: 375x800
- tablet: 768x1024
- desktop: 1366x900

## Requirements

- Node.js (same Node used for frontend) installed
- The app must be running locally on `http://localhost:3000` (run `npm start` from the repo root to start the frontend dev server)

### Install dependencies (from repo root) via PowerShell

```powershell
cd frontend
npm install puppeteer --save-dev
```

### Run the script

Run the script from the repo root (starts from default url `http://localhost:3000`):

```powershell
node .\scripts\smoke_screenshots.js
```

Or specify a different url:

```powershell
node .\scripts\smoke_screenshots.js http://127.0.0.1:3000
```

Output
#### Output


- PNG files written under `scripts/screenshots/` (mobile-, tablet-, desktop- file names)

#### Notes

- Puppeteer downloads a recent Chromium version; ensure you have network access during install.
- For CI, use `puppeteer-core` with a system Chromium and pass the executablePath option.
