// ==========================================
// File: frontend/src/hooks/performance.js
// Role: React hook for UI state management.
// Input Data: Hook params and state.
// Output Data: State values and actions.
// Dependencies: react
// Notes: Consumed by components.
// ==========================================

import { useState, useEffect, useRef } from 'react';

/**
 * useThrottle
 * Limits the frequency of updates to a value.
 * @param {any} value - The value to throttle.
 * @param {number} limit - The time limit in milliseconds.
 * @returns {any} - The throttled value.
 */
export function useThrottle(value, limit) {
    const [throttledValue, setThrottledValue] = useState(value);
    const lastRan = useRef(Date.now());

    useEffect(() => {
        const handler = setTimeout(() => {
            if (Date.now() - lastRan.current >= limit) {
                setThrottledValue(value);
                lastRan.current = Date.now();
            }
        }, limit - (Date.now() - lastRan.current));

        return () => clearTimeout(handler);
    }, [value, limit]);

    return throttledValue;
}

/**
 * useDebounce
 * Delays the update of a value until after a specified delay.
 * @param {any} value - The value to debounce.
 * @param {number} delay - The delay in milliseconds.
 * @returns {any} - The debounced value.
 */
export function useDebounce(value, delay) {
    const [debouncedValue, setDebouncedValue] = useState(value);

    useEffect(() => {
        const handler = setTimeout(() => {
            setDebouncedValue(value);
        }, delay);

        return () => clearTimeout(handler);
    }, [value, delay]);

    return debouncedValue;
}
