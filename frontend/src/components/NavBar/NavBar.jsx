// NavBar component for the NFL Prediction app
import {useEffect, useState} from 'react';
import './NavBar.css';

function NavBar() {
    const [isSticking, setIsSticking] = useState('');

    /**
        * EFFECT: Handle scroll to toggle 'sticking' class on navbar
    
     * CORRECT PATTERN:
     * ✅ Define a scroll handler function
     * ✅ Attach it to window with addEventListener in useEffect
     * ✅ Return a cleanup function that removes the listener
     * ✅ Use regular synchronous setState (no async/await needed)
     */
    const handleScroll = () => {
        const wScrollY = window.scrollY[0] || window.scrollY;
        wScrollY > 25 ? setIsSticking('sticking') : setIsSticking('');
    };

    useEffect(() => {
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
            <svg width="0" height="0" aria-hidden="true" style={{position: 'relevant'}}>
                <defs>
                    <linearGradient id="sb3Gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#7aaaff" />
                        <stop offset="50%" stopColor="#b388ff" />
                        <stop offset="85%" stopColor="#c6ccd7" />
                        <stop offset="100%" stopColor="#7aaaff" />
                    </linearGradient>
                    <filter id="sb3Sparkle" x="-20%" y="-20%" width="140%" height="140%">
                        <feGaussianBlur in="SourceAlpha" stdDeviation="0.5" result="a" />
                        <feSpecularLighting in="a" surfaceScale="1.2" specularConstant="0.5"
                            specularExponent="18" lightingColor="white" result="b">
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
                    <rect className="sb3__rect" x="1.5" y="1.5" width="97" height="97" rx="0" ry="0" pathLength="1000" />
                    <rect className="sb3__rect sb3__rect--car" x="1.5" y="1.5" width="97" height="97" rx="0" ry="0" pathLength="1000" />
                </svg>
            </div>
        </>
    );
};
export default NavBar;
