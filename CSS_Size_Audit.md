# CSS Size & Viewport Audit — TeamGrid.css

This document summarizes targeted changes made to `frontend/src/components/TeamGrid.css` to improve height/width handling and responsive behavior.

## Summary of Rules Updated

1. body
   - Old: max-height: 100vh; max-width: 100vw; width: 100%;
   - New: max-height: 100vh; max-width: 99vw; width: 100%;
   - Justification: Using `99vw` avoids accidental horizontal scrollbars that occur when 100vw plus vertical scrollbar width exceeds the viewport on some browsers.

2. .container
   - Old: max-width: 100vw; width: 100%;
   - New: max-width: 99vw; width: 100%;
   - Justification: Keeps internal container within viewport limits while allowing full-width behavior on smaller devices.

3. .team-grid
   - Old: max-height: 100vh; height: inherit; width: inherit; max-width: 100vw;
   - New: max-height: 100vh; height: inherit; width: 100%; max-width: 99vw;
   - Justification: Set explicit width:100% and use `max-width:99vw` to avoid inheritance from parent that may cause unexpected sizing; preserves full-width layout while preventing overflow.

4. .team-grid.matchups-grid
   - Old: max-width: fit-content;
   - New: max-width: 99vw;
   - Justification: `fit-content` can cause grid children to overflow the viewport. Constraining to `99vw` keeps layout contained while allowing the grid to size responsively.

5. .team-logo
   - Old: width: 180px; height: 180px; (duplicated)
   - New: max-width: 180px; width: 100%; height: auto;
   - Justification: Replace fixed pixel dimensions with responsive sizing that caps the logo to 180px but allows it to shrink on small screens. Preserves aspect ratio via `height:auto`.

6. .team-logo-placeholder
   - Old: width: 90px; height: 90px;
   - New: max-width: 90px; width: 100%; aspect-ratio: 1 / 1; height: auto;
   - Justification: Keeps placeholder square and responsive; `aspect-ratio` ensures consistent shape while allowing flexibility in smaller viewports.

## Notes on Changes and Compatibility

- No color or visual style properties were altered.
- Where `100vw` was used, switched to `99vw` to prevent horizontal scrollbars resulting from vertical scrollbars or browser chrome.
- Replaced fixed px image sizes with `max-width` plus `width:100%` and `height:auto` to keep logos responsive across devices.
- `aspect-ratio` is used for placeholders — this has broad modern browser support; if legacy browser support is required (IE11), we can instead use padding-top technique.

## How Changes Improve Responsiveness

- Prevents accidental horizontal scrollbars by staying slightly below the full viewport width.
- Allows images and placeholders to shrink on narrow screens, reducing layout breakage and overflow.
- Keeps container and grid elements constrained within the viewport while preserving intended spacing and layout.

## Recommendations / Next Steps

- Test on devices and browsers (mobile Safari, Chrome on Android, desktop with/without scrollbars) to confirm no horizontal overflow.
- If you prefer exact `100vw` due to a specific design requirement, consider adding `overflow-x: hidden` to the document root, but that hides genuine overflow too.
- For older browser support where `aspect-ratio` isn't available, replace with a padding-top hack for the placeholder square.

---

If you'd like, I can also:

- Run visual smoke-tests (render screenshots) across a few viewport sizes.
- Apply the `aspect-ratio` fallback for legacy browsers.
- Update other CSS files to follow the same `99vw` convention.

