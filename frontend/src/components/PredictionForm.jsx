import React, { useState } from 'react';

/** 
 * PredictionForm
 * - Matches /predict schema
 * - Adds optional team IDs used by the backend for better imputations
 * - Validates numbers, preserves strings
 */
function PredictionForm({ onPredict }) {
  // Initialize every field to '' to keep inputs controlled
  const [formValues, setFormValues] = useState({
    home_passer_rating: '',
    away_passer_rating: '',
    home_turnovers: '',
    away_turnovers: '',
    home_rushing_yards: '',
    away_rushing_yards: '',
    home_power_rank: '',    // accepted but unused by full model; used in fallback
    away_power_rank: '',
    home: '',               // optional team identifier (e.g., "49ers")
    away: '',
  });

  // TIP: keep one handler; parse only at submit
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormValues(v => ({ ...v, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    // Basic required fields for /predict
    const required = [
      'home_passer_rating','away_passer_rating',
      'home_turnovers','away_turnovers',
      'home_rushing_yards','away_rushing_yards'
    ];
    for (const k of required) {
      if (formValues[k] === '') {
        alert('Fill all required fields.');
        return;
      }
    }

    // Build payload with correct types
    const payload = {
      home_passer_rating: parseFloat(formValues.home_passer_rating),
      away_passer_rating: parseFloat(formValues.away_passer_rating),
      home_turnovers: parseInt(formValues.home_turnovers, 10),
      away_turnovers: parseInt(formValues.away_turnovers, 10),
      home_rushing_yards: parseFloat(formValues.home_rushing_yards),
      away_rushing_yards: parseFloat(formValues.away_rushing_yards),
      // legacy fields, optional
      home_power_rank: formValues.home_power_rank === '' ? null : parseFloat(formValues.home_power_rank),
      away_power_rank: formValues.away_power_rank === '' ? null : parseFloat(formValues.away_power_rank),
      // team IDs, optional but helpful
      home: formValues.home || 'UNK',
      away: formValues.away || 'UNK',
    };

    onPredict(payload);
  };

  return (
    <form className="prediction-form" onSubmit={handleSubmit}>
      <h2>Enter Game Statistics</h2>

      {/* Optional team IDs boost model context */}
      <div className="grid">
        <div className="form-group">
          <label>Home Team (ID)</label>
          <input name="home" value={formValues.home} onChange={handleChange} placeholder="e.g., 49ers" />
        </div>
        <div className="form-group">
          <label>Away Team (ID)</label>
          <input name="away" value={formValues.away} onChange={handleChange} placeholder="e.g., Seahawks" />
        </div>

        <div className="form-group">
          <label>Home Passer Rating*</label>
          <input type="number" step="0.1" name="home_passer_rating" value={formValues.home_passer_rating} onChange={handleChange}/>
        </div>
        <div className="form-group">
          <label>Away Passer Rating*</label>
          <input type="number" step="0.1" name="away_passer_rating" value={formValues.away_passer_rating} onChange={handleChange}/>
        </div>
        <div className="form-group">
          <label>Home Turnovers*</label>
          <input type="number" step="1" name="home_turnovers" value={formValues.home_turnovers} onChange={handleChange}/>
        </div>
        <div className="form-group">
          <label>Away Turnovers*</label>
          <input type="number" step="1" name="away_turnovers" value={formValues.away_turnovers} onChange={handleChange}/>
        </div>
        <div className="form-group">
          <label>Home Rushing Yards*</label>
          <input type="number" step="0.1" name="home_rushing_yards" value={formValues.home_rushing_yards} onChange={handleChange}/>
        </div>
        <div className="form-group">
          <label>Away Rushing Yards*</label>
          <input type="number" step="0.1" name="away_rushing_yards" value={formValues.away_rushing_yards} onChange={handleChange}/>
        </div>
        <div className="form-group">
          <label>Home Power Rank (opt)</label>
          <input type="number" step="0.1" name="home_power_rank" value={formValues.home_power_rank} onChange={handleChange}/>
        </div>
        <div className="form-group">
          <label>Away Power Rank (opt)</label>
          <input type="number" step="0.1" name="away_power_rank" value={formValues.away_power_rank} onChange={handleChange}/>
        </div>
      </div>

      <button type="submit">Predict</button>
    </form>
  );
}

export default PredictionForm;
