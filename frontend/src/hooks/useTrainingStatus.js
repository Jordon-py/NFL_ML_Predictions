import {useState, useCallback} from 'react';
import {startTraining, getHealthStatus} from '../api/client.js';

/**
 * hooks/useTrainingStatus.js
 * --------------------------
 * Component Purpose:
 *   Orchestrate a full "kick off retraining ➜ poll for completion" workflow.
 *   This is the version that hits real backend endpoints.
 *
 * Core Logic Overview:
 *   - `startRetraining` posts to the backend to begin model work, then kicks off polling.
 *   - `pollTrainingStatus` periodically checks the health endpoint until models reload.
 *   - State slices (`status`, `error`, `isLoading`) allow the UI to show progress, disable buttons, etc.
 *
 * Modification Guide:
 *   - If the backend gains a dedicated training-status endpoint, swap it in inside
 *     `pollTrainingStatus` so we stop depending on health checks.
 *   - Keep error handling exhaustive: set both `status` and `error` so the UI knows what to show.
 *   - When introducing new statuses, document them here so consuming components stay aligned.
 */
export function useTrainingStatus() {
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  /**
   * Start the model retraining process
   * Updates status through the training lifecycle
   */
  const startRetraining = useCallback(async () => {
    if (isLoading) return;

    setIsLoading(true);
    setError(null);
    setStatus('running');

    try {
      // Kick off the job. Backend should respond quickly even if work is long-running.
      // Start training
      const result = await startTraining();
      console.log('[Training] Started:', result);

      // If backend returns 'queued', start polling for status
      if (result.status === 'queued') {
        pollTrainingStatus();
      } else if (result.status === 'started') {
        // Training started immediately, poll for completion
        pollTrainingStatus();
      } else if (result.status === 'done') {
        setStatus('done');
      }
    } catch (err) {
      console.error('[Training] Failed to start:', err);
      setError(err.message);
      setStatus('error');
    } finally {
      setIsLoading(false);
    }
  }, [isLoading]);

  /**
   * Poll training status until completion
   * Checks health endpoint to see if models have been reloaded
   */
  const pollTrainingStatus = useCallback(async () => {
    // Polling loop: keep an interval handle so we can stop it explicitly.
    const pollInterval = setInterval(async () => {
      try {
        const health = await getHealthStatus();

        // If models are loaded and healthy, training is complete
        if (health.status === 'healthy' && health.mode === 'models') {
          clearInterval(pollInterval);
          setStatus('done');
          console.log('[Training] Completed successfully');
        }
      } catch (err) {
        console.warn('[Training] Health check failed, continuing to poll:', err);
      }
    }, 2000); // Poll every 2 seconds

    // Stop polling after 5 minutes to prevent infinite polling
    setTimeout(() => {
      clearInterval(pollInterval);
      if (status === 'running') {
        setStatus('error');
        setError('Training timed out');
        console.error('[Training] Timed out after 5 minutes');
      }
    }, 300000);
  }, [status]);

  return {
    status,
    error,
    isLoading,
    startRetraining,
  };
}
