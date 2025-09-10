import { useState, useCallback } from 'react';
import { startTraining, getHealthStatus } from '../api/client.js';

/**
 * Custom hook for managing model training status and operations
 * Handles the complete training workflow with status tracking and polling
 *
 * @returns {object} Training status and control functions
 * @property {string} status - Current training status ('idle' | 'running' | 'done' | 'error')
 * @property {string|null} error - Error message if training failed
 * @property {Function} startRetraining - Function to initiate model retraining
 * @property {boolean} isLoading - Whether a training operation is in progress
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
