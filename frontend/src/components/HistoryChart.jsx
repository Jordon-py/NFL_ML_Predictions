import React, { useEffect, useRef } from 'react';
import { Chart } from 'chart.js/auto';

/**
 * HistoryChart renders a line chart of ensemble probabilities over time.
 *
 * Uses Chart.js via the `chart.js/auto` bundle which automatically registers
 * all necessary chart types. Whenever the `history` prop updates, the chart
 * is re-rendered. Each entry in `history` should have a `date` and
 * `ensemble_proba` field.
 */
function HistoryChart({ history }) {
  const chartRef = useRef(null);
  const chartInstanceRef = useRef(null);

  useEffect(() => {
    const ctx = chartRef.current.getContext('2d');
    // Clean up any existing chart to avoid accumulation
    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
    }
    // Extract labels and data
    const labels = history.map((item) => item.date.toLocaleTimeString());
    const dataPoints = history.map((item) => item.ensemble_proba * 100);
    chartInstanceRef.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Ensemble Win Probability (%)',
            data: dataPoints,
            borderColor: '#0074D9',
            backgroundColor: 'rgba(0, 116, 217, 0.1)',
            tension: 0.2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 0,
            max: 100,
            title: {
              display: true,
              text: 'Win Probability (%)'
            }
          },
          x: {
            title: {
              display: true,
              text: 'Time of Prediction'
            }
          }
        },
        plugins: {
          legend: {
            display: false
          }
        }
      }
    });
  }, [history]);

  return (
    <div className="chart-container">
      <h2>Prediction History</h2>
      <canvas ref={chartRef} height="300"></canvas>
    </div>
  );
}

export default HistoryChart;