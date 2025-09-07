import React, { useState } from 'react';

/**
 * Form component for entering game statistics.
 *
 * Accepts a callback `onPredict` via props which is called with the
 * collected form values when the form is submitted.
 */
function PredictionForm({ onPredict }) {
  const [formValues, setFormValues] = useState({
    home_passer_rating: '',
    away_passer_rating: '',
    home_turnovers: '',
    away_turnovers: '',
    home_rushing_yards: '',
    away_rushing_yards: '',
    home_power_rank: '',
    away_power_rank: ''
  });

  /**
   * Handle changes to the input fields by updating local state.
   */
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormValues((prev) => ({ ...prev, [name]: value }));
  };

  /**
   * Handle form submission by invoking the provided callback with
   * parsed numeric values. Basic validation ensures all fields are filled.
   */
  const handleSubmit = (e) => {
    e.preventDefault();
    // Ensure all fields have values
    const entries = Object.entries(formValues);
    for (const [key, val] of entries) {
      if (val === '') {
        alert('Please fill out all fields before submitting.');
        return;
      }
    }
    // Convert string inputs to numbers
    const payload = {};
    for (const [key, val] of entries) {
      payload[key] = parseFloat(val);
    }
    onPredict(payload);
  };

  return (
    <form className="prediction-form" onSubmit={handleSubmit}>
      <h2>Enter Game Statistics</h2>
      <div className="grid">
        <div className="form-group">
          <label>Home Passer Rating</label>
          <input type="number" name="home_passer_rating" value={formValues.home_passer_rating} onChange={handleChange} step="0.1" />
        </div>
        <div className="form-group">
          <label>Away Passer Rating</label>
          <input type="number" name="away_passer_rating" value={formValues.away_passer_rating} onChange={handleChange} step="0.1" />
        </div>
        <div className="form-group">
          <label>Home Turnovers</label>
          <input type="number" name="home_turnovers" value={formValues.home_turnovers} onChange={handleChange} step="1" />
        </div>
        <div className="form-group">
          <label>Away Turnovers</label>
          <input type="number" name="away_turnovers" value={formValues.away_turnovers} onChange={handleChange} step="1" />
        </div>
        <div className="form-group">
          <label>Home Rushing Yards</label>
          <input type="number" name="home_rushing_yards" value={formValues.home_rushing_yards} onChange={handleChange} step="0.1" />
        </div>
        <div className="form-group">
          <label>Away Rushing Yards</label>
          <input type="number" name="away_rushing_yards" value={formValues.away_rushing_yards} onChange={handleChange} step="0.1" />
        </div>
        <div className="form-group">
          <label>Home Power Rank</label>
          <input type="number" name="home_power_rank" value={formValues.home_power_rank} onChange={handleChange} step="0.1" />
        </div>
        <div className="form-group">
          <label>Away Power Rank</label>
          <input type="number" name="away_power_rank" value={formValues.away_power_rank} onChange={handleChange} step="0.1" />
        </div>
      </div>
      <button type="submit">Predict</button>
    </form>
  );
}

export default PredictionForm;