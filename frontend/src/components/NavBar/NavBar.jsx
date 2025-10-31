import {useEffect, useState} from 'react';
<<<<<<< HEAD
import HamburgerMenu from '../HamburgerMenu';
=======
>>>>>>> c6845983cfbfd1be9afb17b5b47b7331808ca550
import './NavBarr.css';

/**
 * NavBar.jsx
 * ----------
 * Purpose:
 *   Persistent navigation header that gains a "sticky" style after scrolling.
 *
 * Notes:
 *   - `isSticking` holds the CSS class name "sticking" or "" (string-based to avoid breaking CSS).
 *   - We call `handleScroll()` once on mount to sync initial state (in case the page loads scrolled).
 *   - Passive scroll listener + SSR guard for safety.
 */
function NavBar() {
    // Keep string type to avoid changing downstream CSS expectations
    const [isSticking, setIsSticking] = useState('');

    // EFFECT: toggle the "sticking" class after scrolling a small distance.
    const handleScroll = () => {
        if (typeof window === 'undefined') return; // SSR/defensive guard
<<<<<<< HEAD
        setIsSticking(window.scrollY > 10 ? 'sticking' : '');
=======
        setIsSticking(window.scrollY > 25 ? 'sticking' : '');
>>>>>>> c6845983cfbfd1be9afb17b5b47b7331808ca550
    };

    useEffect(() => {
        if (typeof window === 'undefined') return;

        // Sync once on mount (covers initial load where user is already scrolled)
        handleScroll();

        // Add passive listener to avoid blocking scroll
        window.addEventListener('scroll', handleScroll, {passive: true});

        // Cleanup on unmount - CRITICAL to prevent memory leaks
        return () => {
            window.removeEventListener('scroll', handleScroll);
        };
    }, []); // run once on mount

    return (
        <>
<<<<<<< HEAD
            <nav className={`navBar ${isSticking}`} style={{position: 'sticky'}}>
                {/* SVG defs for the border animation – render once and reuse via ids. */}
                <svg width="0" height="0" aria-hidden="true" style={{ position: 'inherit' }}>
                    <defs>
                        <linearGradient id="sb3Gradient" x1="0%" y1="0%" x2="100%" y2="0%">
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

                <h1>NFL Prediction App</h1>
                
=======
            {/* SVG defs for the border animation – render once and reuse via ids. */}
            <svg width="0" height="0" aria-hidden="true" style={{position: 'inherit'}}>
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

            <nav className={`navBar ${isSticking}`} style={{position: 'sticky'}}>
                <h1>NFL Prediction App</h1>
>>>>>>> c6845983cfbfd1be9afb17b5b47b7331808ca550
                <div className="navBar__links">
                    <ul>
                        <li><a href="#home">Home</a></li>
                        <li><a href="#about">About</a></li>
                        <li><a href="#contact">Contact</a></li>
                    </ul>
                </div>
<<<<<<< HEAD
                <HamburgerMenu />
=======
>>>>>>> c6845983cfbfd1be9afb17b5b47b7331808ca550
            </nav>
        </>
    );
}

export default NavBar;
