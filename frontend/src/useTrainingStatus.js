/**
 * useTrainingStatus.js
 * --------------------
 * Component Purpose:
 *   Provide a reusable polling helper for long-running training jobs.
 *   It abstracts the interval/timeout wiring so components can focus on UI.
 *
 * Core Logic Overview:
 *   - Exposes `status`, `startPolling`, and `stop` to consumers.
 *   - Stores interval + timeout IDs in refs so we can clear them during cleanup.
 *   - Polls on a cadence (`pollMs`) and enforces an absolute timeout (`maxMs`).
 *
 * Modification Guide:
 *   - Replace the placeholder body inside `setInterval` with your real
 *     `getTrainingStatus` API call once it is available.
 *   - Always update the `stop` function when introducing new timers so the
 *     hook never leaves background work running after unmount.
 *   - If you add more status states, keep them documented so render logic stays predictable.
 */

import {useCallback, useRef, useState} from 'react';
// Example: import { getTrainingStatus } from './api/client.js';

export default function useTrainingStatus(pollMs = 2500, maxMs = 120000) {
  // Store the latest status consumers can read.
  const [status, setStatus] = useState({state: 'idle'});
  // Refs let us persist timer IDs without triggering renders.
  const intervalRef = useRef(null);
  const timeoutRef = useRef(null);

  // Stop helper clears both timers. We memoise so dependants can use it safely.
  const stop = useCallback(() => {
    if (intervalRef.current) {clearInterval(intervalRef.current); intervalRef.current = null;}
    if (timeoutRef.current) {clearTimeout(timeoutRef.current); timeoutRef.current = null;}
  }, []);

  const startPolling = useCallback(() => {
    stop();
    setStatus({state: 'checking'});

    // Example poller body. Replace with your real endpoint if needed.
    intervalRef.current = setInterval(async () => {
      try {
        // const s = await getTrainingStatus();
        // setStatus(s);

        // Placeholder logic for demo purposes
        setStatus((prev) => (prev.state === 'checking' ? {state: 'running'} : {state: 'done'}));

        // When training finishes, stop polling.
        if (['done', 'failed'].includes(status.state)) {
          stop();
        }
      } catch (e) {
        setStatus({state: 'failed', error: e.message});
        stop();
      }
    }, pollMs);

    // Hard stop after maxMs regardless of state
    timeoutRef.current = setTimeout(() => {
      setStatus((s) => (s.state === 'done' ? s : {state: 'timeout'}));
      stop();
    }, maxMs);
  }, [pollMs, maxMs, status.state, stop]);

  return {status, startPolling, stop};
}
