# Simplification_Report.md

## backend/main_helpers.py
- Summary: Normalized model artifact path resolution to handle Windows-style absolute paths on Linux deploys and reduce branching.
- Before -> After: `Path.is_absolute()` only -> added drive-letter detection + basename fallback to models_dir.
- Reasoning: Prevents Heroku crashes from Windows paths embedded in metadata while keeping local portability.

## backend/services/live_predictor.py
- Summary: Reduced branching in live row assembly, removed dead imports, and replaced broken inference path with direct bundle-based inference.
- Before -> After: duplicated home/away prior fill + missing backend.main imports -> single prior-fill helper + local predict logic.
- Reasoning: Eliminates dead code paths and missing imports, keeps functionality intact with fewer moving parts.

## frontend/src/hooks/usePredictionState.js
- Summary: Simplified team meta merge logic and initial data hydration flow.
- Before -> After: repeated per-side assignments + multiple success/failure branches -> single applyMeta helper + unified schedule/logos hydration.
- Reasoning: Less repetitive code, clearer intent, and fewer branches while preserving schedule enrichment.

## frontend/src/pages/StatsPage.jsx + StatsPage.css
- Summary: Streamlined data derivations with memoized selectors and added lightweight logo-aware schedule markup; harmonized page styling with global theme tokens.
- Before -> After: imperative maps/derived values on each render -> memoized history map and schedule list, new team/logo layout with cohesive card styling.
- Reasoning: Reduces render work, improves readability, and visually aligns the stats page with the rest of the UI.
