# Process.env Fix - Post-Vite Migration

**Date:** October 4, 2025  
**Issue:** `Uncaught ReferenceError: process is not defined`  
**Status:** ✅ RESOLVED

---

## 🔍 Root Cause

After migrating from Create React App to Vite, the browser console showed:

```bash
Uncaught ReferenceError: process is not defined
```

**Why it happened:**

- **Create React App** (Webpack) automatically polyfills Node.js globals like `process`
- **Vite** does NOT polyfill Node.js globals (intentional design for smaller bundles)
- Our code in `frontend/src/api/client.js` used `process.env.REACT_APP_API_URL`

---

## 🔧 The Fix

### Changed File: `frontend/src/api/client.js`

**Before (Create React App style):**

```javascript
const BASE_URL =
  process.env.REACT_APP_API_URL || process.env.VITE_API_URL ||
  'https://nfl-predict-ecf5a5bd34fe.herokuapp.com';
```

**After (Vite style):**

```javascript
const BASE_URL =
  import.meta.env.VITE_API_URL ||
  'https://nfl-predict-ecf5a5bd34fe.herokuapp.com';
```

### Key Changes

1. ❌ Removed `process.env` (Node.js-only)
2. ✅ Added `import.meta.env` (Vite standard)
3. ❌ Removed `REACT_APP_*` prefix
4. ✅ Added `VITE_*` prefix

---

## 📋 Environment Variable Guidelines

### Vite Requirements

- **Prefix:** MUST start with `VITE_` to be exposed to client
- **Access:** Use `import.meta.env.VITE_*` (NOT `process.env.*`)
- **Security:** Only `VITE_*` vars are bundled (prevents secret leakage)

### Example `.env.local`

```bash
# ✅ Exposed to browser
VITE_API_URL=http://localhost:8000

# ❌ NOT exposed (no VITE_ prefix)
SECRET_KEY=abc123
```

### Built-in Variables

```javascript
import.meta.env.MODE         // 'development' or 'production'
import.meta.env.BASE_URL     // Base path for deployment
import.meta.env.PROD         // true in production
import.meta.env.DEV          // true in development
```

---

## ✅ Verification

### Build Test

```bash
npm run build --prefix frontend
# ✓ built in 7.50s
# ✓ Output: build/assets/index-D1N9sJiM.js (new hash)
```

### Code Search

```bash
grep -r "process.env" frontend/src/
# No matches (all converted to import.meta.env)
```

### Browser Test

1. Start backend: `uvicorn backend.main:app --reload --port 8000`
2. Open: `http://localhost:8000`
3. Open DevTools Console
4. ✅ No `process is not defined` errors
5. ✅ API calls work correctly

---

## 📚 Additional Resources

- [Vite Environment Variables](https://vitejs.dev/guide/env-and-mode.html)
- [Migration from process.env](https://vitejs.dev/guide/migration.html#environment-variables)
- [Security Best Practices](https://vitejs.dev/guide/env-and-mode.html#security-notes)

---

## 🎯 Lessons Learned

1. **Vite is strict about browser globals** - No Node.js polyfills by design
2. **Environment variables MUST be prefixed** - `VITE_*` for security
3. **Build tool changes require code updates** - Not always drop-in replacements
4. **Always test after migration** - Even "zero code changes" migrations need validation

---

## 📁 Files Modified

| File | Change | Status |
|------|--------|--------|
| `frontend/src/api/client.js` | `process.env` → `import.meta.env` | ✅ Fixed |
| `frontend/.env.example` | Created with `VITE_API_URL` docs | ✅ Added |
| `frontend/VITE_MIGRATION.md` | Added troubleshooting section | ✅ Updated |

---

**Next Steps:** Test the production deployment to Vercel with the fixed build! 🚀
