// ==========================================
// File: frontend/src/components/Button/D_Button.jsx
// Role: React component for UI rendering.
// Input Data: Props (data and callbacks).
// Output Data: JSX markup.
// Dependencies: react, ./Button.css
// Notes: Presentation-focused component.
// ==========================================

import React from "react";
/**
 * Button (Minimal + Spinner)
 * ------------------------------------------------------------
 * Goals:
 *  - Keep API tiny and predictable
 *  - Safe default type="button" (prevents accidental form submits)
 *  - Built-in loading state with accessible semantics
 *  - No external deps; class hooks align with your central CSS
 *
 * Props:
 *  - children (node): label/content inside the button
 *  - loading (bool): when true, shows spinner, disables button, sets aria-busy
 *  - disabled (bool): manual disable; also disables when loading is true
 *  - className (string): optional extra classes ("btn danger", etc.)
 *  - type (string): "button" | "submit" | "reset" (defaults to "button")
 *  - ...rest: any other valid <button> props (onClick, title, etc.)
 *
 * Why these a11y attributes?
 *  - disabled: native disabling (prevents click/focus)
 *  - aria-disabled: communicates disabled state to assistive tech
 *  - aria-busy: indicates the control is processing
 * 
 * // 1) Default usage
<Button onClick={saveDraft}>Save</Button>

// 2) Loading state (auto-disables + shows spinner)
<Button loading>Saving…</Button>

// 3) Submit button in a form (override type safely)
<Button type="submit" loading={isSubmitting}>Submit</Button>

// 4) Custom class variant (hooked to your central CSS)
<Button className="btn danger" onClick={deleteItem}>Delete</Button>

 */
import './Button.css'
export default function D_Button({
  children,
  loading = false,
  disabled,
  className = "btn",
  type = "button",
  ...rest
}) {
  // If loading, we also disable to prevent double-submits
  const isDisabled = disabled || loading;

  return (
    <button
      type={type}                       // safe default for non-form usage
      className={`${className}${loading ? " is-loading" : ""}`}
      disabled={isDisabled}             // native disabling
      aria-disabled={isDisabled || undefined}
      aria-busy={loading || undefined}
      {...rest}
    >
      {/* Spinner is purely visual; aria-hidden avoids noisy announcements */}
      {loading && <span className="btn__spinner" aria-hidden="true" />}

      {/* Keep label/content in its own span for consistent spacing */}
      <span className="btn__content">{children}</span>
    </button>
  );
}
