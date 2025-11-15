/*
File: Dashboard.jsx
Purpose:
  Main dashboard container; integrates PredictionContext state, renders
  TeamGrid/HistoryChart/PredictionResult, and builds a small nav "summary"
  object for the NavBar.

Key ideas:
  - PredictionContext is the single source of truth for schedule + predictions.
  - localStorage is used only as a fallback history source if context is empty.
  - Layout and styling come from Dashboard.module.css via CSS Modules.
*/
/**
 * Enhanced Dashboard with better data cohesion and state management
 */

import React, { useMemo, useCallback } from 'react';
import { usePredictions } from '../../PredictionContext';
import TeamGrid from '../Card/TeamGrid';
import PredictionResult from '../PredictionResult';
import HistoryChart from '../HistoryChart';
import NavBar from '../NavBar/NavBar';
import LoadingState from '../LoadingState';
import ErrorDisplay from '../ErrorDisplay';

// Custom hooks for better data separation
/**
 * Derive the dashboard-friendly view of PredictionContext state.
 *
 * @returns {{ isLoading: boolean, error: any, data: any }}
 */
function useDashboardData() {
  const context = /** @type {any} */ (usePredictions());

  return useMemo(() => {
    if (!context) {
      return {
        isLoading: true,
        error: new Error('Prediction context not available'),
        data: null
      };
    }

    const {
      current,
      history = [],
      schedule = [],
      week,
      teams = {},
      predictions = {},
      loading = {},
      errors = {},
      makePrediction,
      health: contextHealth
    } = context;

    // Merge health information
    const mergedHealth = contextHealth || { status: 'unknown' };

    // Transform data for consistency
    const transformedHistory = Array.isArray(history)
      ? history
      : [];

    const transformedGames = Array.isArray(schedule)
      ? schedule.map(game => ({
        ...game,
        id: game.id
          || game.game_id
          || `${(game.home_abbr || game.home_team || '').trim()}-${(game.away_abbr || game.away_team || '').trim()}-${game.week}`,
        status: game.status || 'scheduled'
      }))
      : [];

    const isGlobalLoading = transformedGames.length === 0 && !errors.global;

    return {
      isLoading: Boolean(loading.global || isGlobalLoading),
      error: errors.global || null,
      data: {
        currentPrediction: current,
        predictionHistory: transformedHistory,
        upcomingGames: transformedGames,
        currentWeek: week || (transformedGames[0]?.week ?? 1),
        teamMetadata: teams,
        gamePredictions: predictions,
        health: mergedHealth,
        actions: {
          makePrediction
        }
      }
    };
  }, [context]);
}

/**
 * Build a compact navigation summary from dashboard data.
 *
 * @param {any} dashboardData
 */
function useNavigationState(dashboardData) {
  return useMemo(() => {
    if (!dashboardData) {
      return {
        current: null,
        latest: null,
        count: 0,
        health: { status: 'unknown' }
      };
    }

    const { currentPrediction, predictionHistory, health } = dashboardData;
    const latestPrediction = predictionHistory.length > 0 ? predictionHistory[0] : null;

    return {
      current: currentPrediction,
      latest: latestPrediction,
      count: predictionHistory.length,
      health
    };
  }, [dashboardData]);
}

// Main Dashboard Component
export default function Dashboard() {
  const { isLoading, error, data } = useDashboardData();
  const navState = useNavigationState(data);

  // Memoized event handlers
  const handlePredictionRequest = useCallback((/** @type {any} */ gameData) => {
    if (!data?.actions?.makePrediction) {
      console.error('Prediction action not available');
      return;
    }

    try {
      data.actions.makePrediction(gameData);
    } catch (err) {
      console.error('Prediction request failed:', err);
    }
  }, [data]);

  // Early returns for loading and error states
  if (isLoading) {
    return <LoadingState message="Loading dashboard data..." />;
  }

  if (error) {
    return (
      <ErrorDisplay
        error={error}
        onRetry={() => window.location.reload()}
      />
    );
  }

  if (!data) {
    return <ErrorDisplay error={new Error('No data available')} />;
  }

  const {
    upcomingGames,
    currentWeek,
    teamMetadata,
    gamePredictions,
    predictionHistory,
    currentPrediction,
    health
  } = data;

  const isBackendHealthy = health.status === 'healthy';
  const healthMessage = isBackendHealthy
    ? "Click any matchup to see predicted scores"
    : `Backend issue: ${health.reason || 'Service unavailable'}`;

  return (
    <div className="dashboard-layout">
      <NavBar state={navState} />

      <main className="dashboard-main">
        <header className="dashboard-header">
          <div className="header-content">
            <h1 className="dashboard-title">
              NFL Predictions - Week {currentWeek}
            </h1>
            <p className="dashboard-subtitle">
              {healthMessage}
            </p>
          </div>
        </header>

        <section className="dashboard-content">
          <div className="content-grid">
            <div className="team-grid-section">
              <TeamGrid
                games={upcomingGames}
                week={currentWeek}
                teams={teamMetadata}
                predictions={gamePredictions}
                onPredict={handlePredictionRequest}
              />
            </div>

            <div className="history-section">
              <HistoryChart
                history={predictionHistory}
                state={currentPrediction}
              />
            </div>
          </div>
        </section>

        <section className="prediction-results-section" aria-live="polite">
          <PredictionResult
            entry={currentPrediction}
          />
        </section>
      </main>
    </div>
  );
}
