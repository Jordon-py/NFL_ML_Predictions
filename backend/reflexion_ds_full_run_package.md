# Reflexion DS — Full Run Package

This package performs a full purged walk-forward test (WFT), engineered-feature audit, train–test drift analysis, and neural network hyperparameter search on your `train.csv` and `test.csv`.

## Quick start
1. Save the two files below into the same folder.
2. Install deps:
   ```bash
   pip install numpy pandas scikit-learn pyyaml matplotlib
   ```
3. Run:
   ```bash
   python full_run.py --config full_run_config.yaml
   ```
4. Outputs land in `outputs_dir` from the YAML (defaults to `/mnt/data/reflexion_ds_full_run_artifacts`).

---

## `full_run_config.yaml`
```yaml
# Reflexion DS full-run configuration
data:
  train_path: /mnt/data/train.csv
  test_path: /mnt/data/test.csv
validation:
  n_splits: 6
  embargo_groups: 1
features:
  drop:
    - home_win
    - season
    - week
    - home_points_for
    - away_points_for
    - group_idx
wft_model:
  type: sgd_logit
  params:
    max_iter: 3000
    tol: 1.0e-3
    alpha: 0.0001
    loss: log_loss
    random_state: 42
nn_hpo:
  model: mlp
  max_iter: 200
  early_stopping: true
  n_iter_no_change: 12
  validation_fraction: 0.1
  grid:
    hidden_layer_sizes:
      - [128]
      - [256]
      - [128, 64]
      - [64, 64]
      - [128, 128]
    alpha: [0.001, 0.0003]
    learning_rate_init: [0.001, 0.0003]
outputs_dir: /mnt/data/reflexion_ds_full_run_artifacts
```

---

## `full_run.py`
```python
#!/usr/bin/env python3
# Reflexion DS – Full run: WFT, feature audit, drift, and MLP HPO
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import yaml


def make_time_key(df: pd.DataFrame) -> pd.Series:
    return (df["season"].astype(int) * 100 + df["week"].astype(int)).astype(int)


def ensure_target(df: pd.DataFrame) -> pd.Series:
    if "home_win" in df.columns:
        return df["home_win"].astype(int)
    if {"winner", "home_team"}.issubset(df.columns):
        return (df["winner"].astype(str).str.strip() == df["home_team"].astype(str).str.strip()).astype(int)
    if {"home_points_for", "away_points_for"}.issubset(df.columns):
        return (df["home_points_for"].astype(float) > df["away_points_for"].astype(float)).astype(int)
    raise ValueError("Cannot derive binary home_win target")


class PurgedGroupTimeSeriesSplit:
    def __init__(self, n_splits=6, embargo_groups=1):
        self.n_splits = n_splits
        self.embargo_groups = embargo_groups

    def split(self, X, y=None, groups=None):
        uniq = np.unique(groups)
        k = self.n_splits
        sizes = np.full(k, len(uniq) // k, dtype=int)
        sizes[: len(uniq) % k] += 1
        parts = []
        s = 0
        for fs in sizes:
            parts.append(uniq[s : s + fs])
            s += fs
        for i in range(k - 1):
            tr_g = np.concatenate(parts[: i + 1])
            va_g = parts[i + 1]
            tr_g = tr_g[tr_g <= (va_g.max() - self.embargo_groups)]
            tr = np.where(np.isin(groups, tr_g))[0]
            va = np.where(np.isin(groups, va_g))[0]
            yield tr, va


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a)
    b = np.sort(b)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    ai = np.searchsorted(a, b, side="right")
    bi = np.arange(1, len(b) + 1)
    return float(np.max(np.abs(ai / len(a) - bi / len(b))))


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    train_path = Path(cfg["data"]["train_path"])
    test_path = Path(cfg["data"]["test_path"])
    outdir = Path(cfg["outputs_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    for df in (train, test):
        df["home_win"] = ensure_target(df)
        df["group_idx"] = make_time_key(df)

    drop = set(cfg["features"]["drop"]) | {"game_id", "home_team", "away_team", "winner", "loser"}
    feature_cols = [c for c in train.select_dtypes(include=[np.number]).columns if c not in drop]
    for c in feature_cols:
        if c not in test.columns:
            test[c] = 0.0

    X = train[feature_cols].copy()
    X_means = X.mean()
    X = X.fillna(X_means)
    X_test = test[feature_cols].copy().fillna(X_means)

    y = train["home_win"].astype(int).values
    y_test = test["home_win"].astype(int).values
    g = train["group_idx"].astype(int).values

    # ---------- Walk-forward testing (SGD Logit) ----------
    cv = PurgedGroupTimeSeriesSplit(cfg["validation"]["n_splits"], cfg["validation"]["embargo_groups"])
    sgd = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(**cfg["wft_model"]["params"]))
    ])

    prob_oof = np.zeros_like(y, dtype=float)
    y_oof = np.zeros_like(y, dtype=float)
    fold_rows = []
    for k, (tr, va) in enumerate(cv.split(X, y, g), 1):
        sgd.fit(X.iloc[tr], y[tr])
        p = np.clip(sgd.predict_proba(X.iloc[va])[:, 1], 1e-6, 1 - 1e-6)
        prob_oof[va] = p
        y_oof[va] = y[va]
        fold_rows.append({
            "fold": k,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "brier": float(brier_score_loss(y[va], p)),
            "logloss": float(log_loss(y[va], p)),
            "roc_auc": float(roc_auc_score(y[va], p)) if len(np.unique(y[va])) > 1 else float("nan"),
            "pr_auc": float(average_precision_score(y[va], p)) if len(np.unique(y[va])) > 1 else float("nan"),
            "accuracy": float(accuracy_score(y[va], (p >= 0.5).astype(int))),
        })

    wft_folds = pd.DataFrame(fold_rows)
    wft_overall = {
        "brier_oof": float(brier_score_loss(y_oof, prob_oof)),
        "logloss_oof": float(log_loss(y_oof, prob_oof)),
        "roc_auc_oof": float(roc_auc_score(y_oof, prob_oof)) if len(np.unique(y_oof)) > 1 else float("nan"),
        "pr_auc_oof": float(average_precision_score(y_oof, prob_oof)) if len(np.unique(y_oof)) > 1 else float("nan"),
        "accuracy_oof": float(accuracy_score(y_oof, (prob_oof >= 0.5).astype(int))),
    }

    wft_folds.to_csv(outdir / "wft_fold_metrics.csv", index=False)
    with open(outdir / "wft_overall_metrics.json", "w") as f:
        json.dump(wft_overall, f, indent=2)

    # ---------- Feature audit ----------
    stats = X.describe().T.reset_index().rename(columns={"index": "feature"})
    stats["missing_pct"] = X.isna().mean().values
    corrs = []
    y_float = y.astype(float)
    for c in X.columns:
        v = X[c].values.astype(float)
        corrs.append(float(np.corrcoef(v, y_float)[0, 1]) if np.std(v) > 0 else 0.0)
    stats["pearson_to_target"] = corrs
    stats.sort_values(by="pearson_to_target", key=np.abs, ascending=False).to_csv(outdir / "feature_audit.csv", index=False)

    drift_rows = [{"feature": c, "ks_train_vs_test": ks_stat(X[c].values, X_test[c].values)} for c in X.columns]
    pd.DataFrame(drift_rows).sort_values("ks_train_vs_test", ascending=False).to_csv(outdir / "train_test_feature_drift.csv", index=False)

    # ---------- MLP HPO (manual grid) ----------
    grid = {
        "hidden_layer_sizes": [tuple(v) for v in cfg["nn_hpo"]["grid"]["hidden_layer_sizes"]],
        "alpha": cfg["nn_hpo"]["grid"]["alpha"],
        "learning_rate_init": cfg["nn_hpo"]["grid"]["learning_rate_init"],
    }
    params = list(ParameterGrid(grid))

    splits = list(PurgedGroupTimeSeriesSplit(cfg["validation"]["n_splits"], cfg["validation"]["embargo_groups"]).split(X, y, g))

    best_cfg = None
    best_cv_ll = float("inf")
    for p in params:
        ll_sum = 0.0
        for tr, va in splits:
            mlp = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=p["hidden_layer_sizes"],
                    activation="relu",
                    solver="adam",
                    alpha=p["alpha"],
                    learning_rate_init=p["learning_rate_init"],
                    max_iter=cfg["nn_hpo"]["max_iter"],
                    early_stopping=cfg["nn_hpo"]["early_stopping"],
                    n_iter_no_change=cfg["nn_hpo"]["n_iter_no_change"],
                    validation_fraction=cfg["nn_hpo"]["validation_fraction"],
                    random_state=42,
                )),
            ])
            mlp.fit(X.iloc[tr], y[tr])
            prob = np.clip(mlp.predict_proba(X.iloc[va])[:, 1], 1e-6, 1 - 1e-6)
            ll_sum += log_loss(y[va], prob)
        if ll_sum < best_cv_ll:
            best_cv_ll = ll_sum
            best_cfg = p

    best_mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=best_cfg["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=best_cfg["alpha"],
            learning_rate_init=best_cfg["learning_rate_init"],
            max_iter=max(250, cfg["nn_hpo"]["max_iter"]),
            early_stopping=cfg["nn_hpo"]["early_stopping"],
            n_iter_no_change=cfg["nn_hpo"]["n_iter_no_change"],
            validation_fraction=cfg["nn_hpo"]["validation_fraction"],
            random_state=42,
        )),
    ])
    best_mlp.fit(X, y)
    prob_test = np.clip(best_mlp.predict_proba(X_test)[:, 1], 1e-6, 1 - 1e-6)
    pred_test = (prob_test >= 0.5).astype(int)

    nn_metrics = {
        "best_params": best_cfg,
        "cv_best_logloss": float(best_cv_ll / len(splits)),
        "test_logloss": float(log_loss(y_test, prob_test)),
        "test_brier": float(brier_score_loss(y_test, prob_test)),
        "test_auc": float(roc_auc_score(y_test, prob_test)) if len(np.unique(y_test)) > 1 else float("nan"),
        "test_pr_auc": float(average_precision_score(y_test, prob_test)) if len(np.unique(y_test)) > 1 else float("nan"),
        "test_accuracy": float(accuracy_score(y_test, pred_test)),
        "test_precision": float(precision_score(y_test, pred_test)),
        "test_recall": float(recall_score(y_test, pred_test)),
        "test_f1": float(f1_score(y_test, pred_test)),
    }
    with open(outdir / "mlp_test_metrics.json", "w") as f:
        json.dump(nn_metrics, f, indent=2)

    preds = test[["season", "week"]].copy()
    if "game_id" in test.columns:
        preds["game_id"] = test["game_id"]
    preds["prob_home_win"] = prob_test
    preds["pred_home_win"] = pred_test
    preds.to_csv(outdir / "mlp_test_predictions.csv", index=False)

    print(json.dumps({
        "artifacts": {
            "fold_metrics_csv": str(outdir / "wft_fold_metrics.csv"),
            "overall_metrics_json": str(outdir / "wft_overall_metrics.json"),
            "feature_audit_csv": str(outdir / "feature_audit.csv"),
            "drift_csv": str(outdir / "train_test_feature_drift.csv"),
            "mlp_metrics_json": str(outdir / "mlp_test_metrics.json"),
            "mlp_predictions_csv": str(outdir / "mlp_test_predictions.csv"),
        },
        "schema": {
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_features": int(len(feature_cols)),
        },
    }, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="full_run_config.yaml")
    args = ap.parse_args()
    main(args.config)
```

