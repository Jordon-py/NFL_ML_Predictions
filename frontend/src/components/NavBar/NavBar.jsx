import { useEffect, useState, useCallback } from "react";
import "./NavBar.css";

/**
 * NavBar.jsx
 * ----------
 * Purpose:
 *   Persistent navigation header that gains a "sticky" style after scrolling.
 *
 * Notes:
 *   - `isSticking` holds the CSS class name "sticking" or "" (string-based).
 *   - We call `handleScroll()` once on mount to sync initial state.
 *   - Passive scroll listener + SSR guard for safety.
 *
 * Optional:
 *   - Pass `health` as a prop if you want a status indicator:
 *     <NavBar health={{ status: "healthy" }} />
 */
export default function NavBar({ health }) {
  // Keep string type to avoid changing downstream CSS expectations
  const [isSticking, setIsSticking] = useState("");

  // Memoized so the scroll listener always removes the exact same function reference
  const handleScroll = useCallback(() => {
    if (typeof window === "undefined") return; // SSR/defensive guard
    setIsSticking(window.scrollY > 10 ? "sticking" : "");
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;

    // Sync once on mount (covers initial load where user is already scrolled)
    handleScroll();

    // Add passive listener to avoid blocking scroll
    window.addEventListener("scroll", handleScroll, { passive: true });

    // Cleanup on unmount - CRITICAL to prevent memory leaks
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, [handleScroll]);

  // Safe status class (won't crash if health isn't provided)
  const healthStatusClass =
    health?.status === "healthy"
      ? "health-ok"
      : health?.status === "unhealthy"
        ? "health-error"
        : "health-unknown";

  return (
    <nav className={`navBar ${isSticking}`} style={{ position: "sticky", top: 0 }}>
      {/* SVG defs for border animation – render once and reuse via ids */}
      <svg width="0" height="0" aria-hidden="true">
        <defs>
          <linearGradient id="sb3Gradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#7aaaff" />
            <stop offset="50%" stopColor="#b388ff" />
            <stop offset="85%" stopColor="#c6ccd7" />
            <stop offset="100%" stopColor="#7aaaff" />
          </linearGradient>

          <filter id="sb3Sparkle" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur in="SourceAlpha" stdDeviation="0.5" result="a" />
            <feSpecularLighting
              in="a"
              surfaceScale="0.4"
              specularConstant="0.5"
              specularExponent="18"
              lightingColor="white"
              result="b"
            >
              <fePointLight x="-60" y="-40" z="80" />
            </feSpecularLighting>
            <feComposite in="b" in2="SourceAlpha" operator="in" result="spec" />
            <feMerge>
              <feMergeNode in="spec" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
      </svg>

      <div className="navBar__inner">
        <h1 className="navBar__title">NFL Prediction App</h1>

        {/* Optional health indicator (safe even if CSS doesn’t style it yet) */}
        <span className={`navBar__health ${healthStatusClass}`} aria-label="API health status" />

        <div className="navBar__links">
          <ul>
            <li><a href="#home">Home</a></li>
            <li><a href="#about">About</a></li>
            <li><a href="#contact">Contact</a></li>
          </ul>
        </div>
      </div>
    </nav>
  );
}
