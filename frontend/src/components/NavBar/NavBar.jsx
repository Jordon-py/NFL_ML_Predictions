// NavBar component for the NFL Prediction app
import { useEffect, useRef, useState } from 'react';
import './NavBar.css';

const NavBar = () => {
    const [isSticking, setIsSticking] = useState(false);
    
    /**
     * BUG FIX: useEffect scroll handler
     * 
     * WRONG PATTERN (original code):
     * ❌ useEffect(() => { async function handleScroll() { ... return <JSX/> } }, [])
     * 
     * WHY IT'S WRONG:
     * 1. useEffect should NOT return JSX - it can only return a cleanup function
     * 2. Using 'async' is unnecessary - setState is synchronous
     * 3. Using 'await' with setState doesn't make sense - setState isn't a Promise
     * 4. The scroll handler was never attached to window.addEventListener
     * 5. Returning JSX from inside useEffect does nothing - React ignores it
     * 
     * CORRECT PATTERN:
     * ✅ Define a scroll handler function
     * ✅ Attach it to window with addEventListener
     * ✅ Return a cleanup function that removes the listener
     * ✅ Use regular synchronous setState (no async/await needed)
     */
    useEffect(() => {
        // Modern property first, fallback for older browsers
        const handleScroll = () => {
            const scrollY = window.pageYOffset || document.documentElement.scrollTop;
            setIsSticking(scrollY > 50); // Trigger sticky state after 50px of scroll
        };

        // Attach the scroll event listener
        window.addEventListener('scroll', handleScroll);
        
        // CRITICAL: Return cleanup function to prevent memory leaks
        // This removes the listener when component unmounts
        return () => {
            window.removeEventListener('scroll', handleScroll);
        };
    }, []); // Empty dependency array = run once on mount

    return (
        <>
            {/* SVG Defs for racetrack border effect - used globally */}
            <svg width="0" height="0" aria-hidden="true" style={{position: 'absolute'}}>
                <defs>
                    <linearGradient id="sb3Gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#7aaaff"/>
                        <stop offset="50%" stopColor="#b388ff"/>
                        <stop offset="85%" stopColor="#c6ccd7"/>
                        <stop offset="100%" stopColor="#7aaaff"/>
                    </linearGradient>
                    <filter id="sb3Sparkle" x="-20%" y="-20%" width="140%" height="140%">
                        <feGaussianBlur in="SourceAlpha" stdDeviation="0.5" result="a"/>
                        <feSpecularLighting in="a" surfaceScale="1.2" specularConstant="0.5"
                            specularExponent="18" lightingColor="white" result="b">
                            <fePointLight x="-60" y="-40" z="80"/>
                        </feSpecularLighting>
                        <feComposite in="b" in2="SourceAlpha" operator="in" result="spec"/>
                        <feMerge>
                            <feMergeNode in="spec"/>
                            <feMergeNode in="SourceGraphic"/>
                        </feMerge>
                    </filter>
                </defs>
            </svg>

            <div className="sb3" style={{borderRadius: '0'}}>
                <nav className={`navBar sb3__content ${isSticking ? 'sticking' : ''}`}>
                    <h1>NFL Prediction App</h1>
                    <div className="navBar__links">
                        <ul>
                            <li><a href="#home">Home</a></li>
                            <li><a href="#about">About</a></li>
                            <li><a href="#contact">Contact</a></li>
                        </ul>
                    </div>
                </nav>
                <svg className="sb3__svg" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                    <rect className="sb3__rect" x="1.5" y="1.5" width="97" height="97" rx="0" ry="0" pathLength="1000"/>
                    <rect className="sb3__rect sb3__rect--car" x="1.5" y="1.5" width="97" height="97" rx="0" ry="0" pathLength="1000"/>
                </svg>
            </div>
        </>
    );
};
export default NavBar;
