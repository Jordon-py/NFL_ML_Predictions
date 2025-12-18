import { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import './NavBar.css';
import HamburgerMenu from '../Hamburger/HamburgerMenu';

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
export default function NavBar({ health } = {})
{
    // Keep string type to avoid changing downstream CSS expectations
    const [ isSticking, setIsSticking ] = useState( '' );

    // EFFECT: toggle the "sticking" class after scrolling a small distance.
    const handleScroll = () =>
    {
        if ( typeof window === 'undefined' ) return; // SSR/defensive guard
        setIsSticking( window.scrollY > 10 ? 'sticking' : '' );
    };

    useEffect( () =>
    {
        if ( typeof window === 'undefined' )
            return;

    // Sync once on mount (covers initial load where user is already scrolled)
    handleScroll();

        // Add passive listener to avoid blocking scroll
        window.addEventListener( 'scroll', handleScroll, { passive: true } );

        // Cleanup on unmount - CRITICAL to prevent memory leaks
        return () =>
        {
            window.removeEventListener( 'scroll', handleScroll );
        };
    }, [] ); // run once on mount

    const healthStatusClass = health?.status === 'healthy'
        ? 'health-ok'
        : health?.status === 'unhealthy'
            ? 'health-error'
            : 'health-unknown';

    return (
        <nav className={ `navBar ${isSticking}` }>
            {/* SVG defs for the border animation – render once and reuse via ids. */ }
            <svg width="0" height="0" aria-hidden="true">
                <defs>
                    <linearGradient id="sb3Gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#9abfffff" />
                        <stop offset="50%" stopColor="#a16affff" />
                        <stop offset="85%" stopColor="#8fabdeff" />
                        <stop offset="100%" stopColor="#2c79ffff" />
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

            <div className="nav-left">
                <h1>NFL Predict</h1>
                <div
                    className={ `health-indicator ${healthStatusClass}` }
                    title={ `Backend Status: ${health?.status ?? 'unknown'}${health?.reason ? ` - ${health.reason}` : ''}` }
                ></div>
            </div>

            {/* Desktop links (hidden on small screens via CSS) */ }
            <div className="navBar__links">
                <NavLink to="/" end>Dashboard</NavLink>
                <NavLink to="/history">History</NavLink>
                <NavLink to="/stats">Stats</NavLink>
            </div>
            {/* Mobile hamburger (shown on small screens via CSS) */ }
            <div className="navBar__hamburger" aria-label="Navigation menu">
                <HamburgerMenu />
            </div>
        </nav>
    );
}
