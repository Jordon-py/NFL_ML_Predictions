# NFL Team-Game Pipeline → CSVs → Models → API → Frontend

Predict NFL win probability using curated team-game features. No scraping. Uses nflverse (`nfl-data-py`), builds expert CSVs, trains models, serves an API, and plugs into a minimal React widget.

---

```bash
# 1) Create CSVs
pip install nfl-data-py pandas numpy
python scripts/build_csvs.py --start 2014 --end 2023 --out-dir data
# Outputs: data/team_game_{base,iter1,iter2,iter3}.csv + schema files
```

## **2) Train models**

```bash
pip install scikit-learn lightgbm joblib
python train_models.py
```
## **Outputs: models/{preprocessor.joblib, nn_model.joblib, gbm_model.txt, metadata.json}**

# 3) Serve API
pip install fastapi uvicorn
uvicorn main:app --reload --port 8000

# **WHATS INSIDE BACKEND**

backend/data/             # CSV artifacts
  team_game_base.csv
  team_game_iter1.csv
  team_game_iter2.csv
  team_game_iter3.csv
  team_game_iter3.schema.json
  team_game_iter3.schema.md

backend/models/           # Trained artifacts
  preprocessor.joblib
  nn_model.joblib
  gbm_model.txt
  metadata.json

backend/scripts/
  build_csvs.py    # Builds the four CSVs and auto-writes schema files
  make_schema.py   # Schema generator (JSON + Markdown)

main.py            # FastAPI service: /health, /predict, /predict_raw, /retrain
train_models.py    # Trains NN + GBM, writes artifacts + metadata
README.md

