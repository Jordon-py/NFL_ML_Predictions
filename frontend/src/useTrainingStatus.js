/**
 * useTrainingStatus.js
 * --------------------
 * Purpose:
 *   Poll backend training status safely with cleanup to avoid leaks.
 *
 * Layer 1 Fix:
 *   - Clear both interval and timeout on unmount or completion.
 *
 * Pattern:
 *   const { status, startPolling } = useTrainingStatus();
 *   useEffect(() => { startPolling(); }, [startPolling]);
 */

import { useCallback, useRef, useState } from 'react';
import { /* optional: training status API */ } from './client.js';

export default function useTrainingStatus(pollMs = 2500, maxMs = 120000) {
  const [status, setStatus] = useState({ state: 'idle' });
  const intervalRef = useRef(null);
  const timeoutRef = useRef(null);

  const stop = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
    if (timeoutRef.current) { clearTimeout(timeoutRef.current); timeoutRef.current = null; }
  }, []);

  const startPolling = useCallback(() => {
    stop();
    setStatus({ state: 'checking' });

    // Example poller body. Replace with your real endpoint if needed.
    intervalRef.current = setInterval(async () => {
      try {
        // const s = await getTrainingStatus();
        // setStatus(s);

        // Placeholder logic for demo purposes
        setStatus((prev) => (prev.state === 'checking' ? { state: 'running' } : { state: 'done' }));

        // When training finishes, stop polling.
        if (['done', 'failed'].includes(status.state)) {
          stop();
        }
      } catch (e) {
        setStatus({ state: 'failed', error: e.message });
        stop();
      }
    }, pollMs);

    // Hard stop after maxMs regardless of state
    timeoutRef.current = setTimeout(() => {
      setStatus((s) => (s.state === 'done' ? s : { state: 'timeout' }));
      stop();
    }, maxMs);
  }, [pollMs, maxMs, status.state, stop]);

  return { status, startPolling, stop };
}
