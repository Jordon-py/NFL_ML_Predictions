"""
W1 Unsupervised Lab — merge_dominance.csv (Matplotlib only)
Clustering NFL games using pre-game features; fast & teachable.

RUN WITH:
python w1_unsupervised_merge_dominance.py --csv ./backend/data/merge_dominance.csv
"""

import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
# Re-run (compact): same logic, but split into smaller steps to avoid tool resets.

from sklearn.linear_model import LogisticRegression

# Small utility to display/save tabular outputs in CLI/Jupyter.
def display_dataframe_to_user(title: str, data) -> None:
    """
    Print a DataFrame-like object and save it to a CSV named after the title.
    Accepts pd.DataFrame or pd.Series; falls back to DataFrame(data) if needed.
    """
    if isinstance(data, pd.Series):
        df = data.reset_index()
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)

    print(f"\n=== {title} ===")
    try:
        # If running in a notebook, display nicely; otherwise fallback print.
        from IPython.display import display  # type: ignore
        display(df)
    except Exception:
        print(df.to_string(index=False))

    # Save artifact for later inspection
    slug = "".join(ch if ch.isalnum() else "_" for ch in title.lower()).strip("_")
    out_path = f"{slug}.csv"
    try:
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Warning: could not save {out_path}: {e}")

df = pd.read_csv('backend\\data\\merge_dominance.csv')
# ---- Pre-game numeric matrix
X = df.select_dtypes(include=["number"]).copy()
for col in ("home_points_for","away_points_for","point_diff"):
    if col in X.columns:
        X.drop(columns=col, inplace=True)
X = X.fillna(X.median(numeric_only=True))
X_orig = X.copy()

# ---- Scale + k=2 MiniBatchKMeans
Xs = StandardScaler().fit_transform(X)
km = MiniBatchKMeans(n_clusters=2, batch_size=512, n_init=5, max_iter=200, random_state=42)
labels = km.fit_predict(Xs)

# ---- PCA for readable 2D plot (with explained variance in labels)
pca = PCA(n_components=2, random_state=42).fit(Xs)
X2 = pca.transform(Xs); centers2 = pca.transform(km.cluster_centers_)
pc1_var = pca.explained_variance_ratio_[0]*100; pc2_var = pca.explained_variance_ratio_[1]*100

plt.figure(figsize=(6,5))
plt.scatter(X2[:,0], X2[:,1], c=labels, s=16, alpha=0.9)
plt.scatter(centers2[:,0], centers2[:,1], c="black", s=140, marker="X")
plt.title(f"PCA scatter (k=2). PC1={pc1_var:.1f}%  PC2={pc2_var:.1f}%")
plt.xlabel(f"PC1 ({pc1_var:.1f}% variance)"); plt.ylabel(f"PC2 ({pc2_var:.1f}% variance)")
plt.grid(alpha=.25); plt.tight_layout(); plt.show()

# ---- Biplot-style top-loading arrows
load2 = pca.components_[:2, :]
strength = np.sqrt((load2**2).sum(axis=0))
top_idx = np.argsort(strength)[-10:][::-1]
top_feats = X.columns[top_idx].tolist()
scale = 3.0

plt.figure(figsize=(6,5))
plt.scatter(X2[:,0], X2[:,1], c=labels, s=8, alpha=.7)
plt.scatter(centers2[:,0], centers2[:,1], c="black", s=120, marker="X")
for j, feat in enumerate(top_feats):
    lx, ly = load2[0, top_idx[j]]*scale, load2[1, top_idx[j]]*scale
    plt.arrow(0,0,lx,ly, head_width=0.06, length_includes_head=True)
    plt.text(lx*1.05, ly*1.05, feat, fontsize=8)
plt.title("PCA with top-loading feature arrows")
plt.xlabel(f"PC1 ({pc1_var:.1f}%)"); plt.ylabel(f"PC2 ({pc2_var:.1f}%)")
plt.grid(alpha=.25); plt.tight_layout(); plt.show()

# ---- Non-PCA comparison using two most discriminative original features
mask0 = labels==0; mask1 = labels==1
def smd(col_idx):
    a, b = Xs[mask0, col_idx], Xs[mask1, col_idx]
    mu0, mu1 = a.mean(), b.mean()
    s0, s1 = a.std(ddof=1), b.std(ddof=1)
    sp = np.sqrt(((len(a)-1)*s0**2 + (len(b)-1)*s1**2)/max((len(a)+len(b)-2),1))
    return (mu1-mu0)/(sp+1e-12)

smd_vals = np.array([smd(j) for j in range(Xs.shape[1])])
top2 = np.argsort(np.abs(smd_vals))[-2:][::-1]
f1, f2 = X.columns[top2[0]], X.columns[top2[1]]

plt.figure(figsize=(6,5))
plt.scatter(X_orig[f1], X_orig[f2], c=labels, s=16, alpha=.9)
plt.title(f"Non‑PCA view — top features: {f1} vs {f2}")
plt.xlabel(f1); plt.ylabel(f2)
plt.grid(alpha=.25); plt.tight_layout(); plt.show()

# ---- Top-10 distinguishing features table (by |standardized mean diff|)
rank = np.argsort(np.abs(smd_vals))[::-1][:10]
feat = X.columns[rank]; vals = smd_vals[rank]
table = pd.DataFrame({"feature": feat, "std_mean_diff (cluster1 - cluster0)": np.round(vals,3)})
display_dataframe_to_user("Top-10 distinguishing features", table)

# ---- Optional: linear probe (logistic regression) to confirm drivers
lr = LogisticRegression(max_iter=250, random_state=42)
lr.fit(Xs, labels)
coef_top = pd.Series(lr.coef_[0], index=X.columns).sort_values(key=np.abs, ascending=False).head(10).round(3)
display_dataframe_to_user("Linear probe coefficients (top 10 abs)", coef_top.reset_index().rename(columns={"index":"feature","0":"coef"}))
# -----------------------------
# 0) CLI — point to your CSV
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to merge_dominance.csv")
    p.add_argument("--kmin", type=int, default=2, help="min k for search")
    p.add_argument("--kmax", type=int, default=6, help="max k for search")
    p.add_argument("--sample", type=int, default=1000,
                   help="rows for quick k-selection (MiniBatch & silhouette)")
    return p.parse_args()

# -----------------------------
# 1) Load + numeric selection
# -----------------------------
def load_numeric(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep only numeric columns for unsupervised models
    num = df.select_dtypes(include=["number"]).copy()

    # Remove obvious post-game leakage: these encode outcomes after the game
    for col in ("home_points_for", "away_points_for", "point_diff", "winner"):
        if col in num.columns:
            num.drop(columns=col, inplace=True)

    return df, num

# -----------------------------
# 2) Basic cleaning
# -----------------------------
def clean_numeric(num: pd.DataFrame, max_missing_pct: float = 40.0) -> pd.DataFrame:
    # Drop columns that are mostly missing; they add noise & slow down models
    pct_missing = num.isna().mean() * 100.0
    keep_cols = pct_missing.index[pct_missing <= max_missing_pct].tolist()
    X = num[keep_cols].copy()

    # Median-impute remaining NaNs (robust default for numeric)
    X = X.fillna(X.median(numeric_only=True))
    return X

# -----------------------------
# 3) Scale + fast k-selection
# -----------------------------
def select_k_and_fit(X: pd.DataFrame, kmin=2, kmax=6, sample_n=1000, rng_seed=42):
    # Standardize so features contribute evenly (K-Means + PCA both expect this)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # Subsample for fast silhouette estimation (optional but practical)
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(Xs), size=min(sample_n, len(Xs)), replace=False)
    X_sub = Xs[idx]

    k_list = list(range(kmin, kmax + 1))
    sil_scores = []

    # Use MiniBatchKMeans for speed; good quality for large-ish data
    for k in k_list:
        mb = MiniBatchKMeans(
            n_clusters=k, random_state=rng_seed,
            batch_size=256, n_init=5, max_iter=100
        )
        # Fit & predict on the sample (fast), good enough for k selection
        labels_sub = mb.fit_predict(X_sub)
        sil = silhouette_score(X_sub, labels_sub)
        sil_scores.append(sil)

    # Choose the k with the best (highest) silhouette
    best_k = k_list[int(np.argmax(sil_scores))]

    # Fit final model on *all* rows using the chosen k
    mb_final = MiniBatchKMeans(
        n_clusters=best_k, random_state=rng_seed,
        batch_size=512, n_init=10, max_iter=200
    ).fit(Xs)
    labels_all = mb_final.labels_

    return scaler, labels_all, mb_final, k_list, sil_scores

# -----------------------------
# 4) PCA for 2D plots
# -----------------------------
def pca_2d(X_scaled: np.ndarray, centers_scaled: np.ndarray, seed=42):
    pca = PCA(n_components=2, random_state=seed).fit(X_scaled)
    X2 = pca.transform(X_scaled)
    C2 = pca.transform(centers_scaled)
    return X2, C2

# -----------------------------
# 5) Useful plots (matplotlib)
# -----------------------------
def plot_silhouette(k_list, sil_scores):
    plt.figure(figsize=(6,4))
    plt.plot(k_list, sil_scores, marker="o")
    plt.title("Silhouette vs k (MiniBatchKMeans)")
    plt.xlabel("k"); plt.ylabel("Silhouette")
    plt.grid(alpha=.3); plt.tight_layout(); plt.show()

def plot_pca_scatter(X2, labels, centers2, best_k):
    plt.figure(figsize=(6,5))
    plt.scatter(X2[:,0], X2[:,1], c=labels, s=14, alpha=.9)
    plt.scatter(centers2[:,0], centers2[:,1], c="black", s=140, marker="X")
    plt.title(f"Clusters in PCA 2D (k={best_k})")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.grid(alpha=.2); plt.tight_layout(); plt.show()

# -----------------------------
# 6) Cluster profiling (football)
# -----------------------------
PROFILE_CANDIDATES = [
    # Odds/signals expected before kickoff
    "moneyline_prob_diff", "home_moneyline_prob", "away_moneyline_prob",
    "spread_line", "total_line", "rest_diff",
    # Form / dominance style features in your file
    "dom_home_win_pct", "dom_away_win_pct", "season_home_win_rate",
    "pre_home_win_rate", "pre_away_win_rate"
]

def cluster_profile_table(X: pd.DataFrame, labels: np.ndarray, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact per-cluster mean table over selected pre-game features.
    Falls back to all numeric features in X if candidates are missing.
    """
    # Prefer human-meaningful features if present
    cols = [c for c in PROFILE_CANDIDATES if c in raw_df.columns]
    if not cols:
        cols = list(X.columns)

    tmp = pd.DataFrame({"cluster": labels}).join(raw_df[cols].reset_index(drop=True))
    means = tmp.groupby("cluster")[cols].mean().reset_index()

    # Add cluster sizes for context
    counts = pd.Series(labels).value_counts().sort_index().rename("count").reset_index(drop=True)
    means.insert(1, "count", counts.values)
    return means

def main():
    """
    CLI entrypoint:
    - Load CSV
    - Select numeric pre-game features and clean
    - Pick k via silhouette on MiniBatchKMeans, fit final model
    - PCA for 2D viz
    - Plot silhouette and PCA scatter
    - Produce cluster profile and save artifacts
    """
    args = parse_args()

    # 1) Load + numeric selection
    raw_df, num = load_numeric(args.csv)

    # 2) Basic cleaning
    X = clean_numeric(num, max_missing_pct=40.0)

    # 3) Scale + fast k-selection + final fit
    scaler, labels, model, k_list, sil_scores = select_k_and_fit(
        X, kmin=args.kmin, kmax=args.kmax, sample_n=args.sample
    )

    # 4) PCA for 2D plots
    X_scaled = scaler.transform(X)
    X2, centers2 = pca_2d(X_scaled, model.cluster_centers_)
    best_k = model.n_clusters

    # 5) Plots
    plot_silhouette(k_list, sil_scores)
    plot_pca_scatter(X2, labels, centers2, best_k)

    # 6) Cluster profile table
    profile = cluster_profile_table(X, labels, raw_df)
    display_dataframe_to_user("Cluster Profile (per-cluster means of pre-game features)", profile)

    # 7) Save artifacts
    pd.DataFrame({"k": k_list, "silhouette": sil_scores}).to_csv("silhouette_scores.csv", index=False)
    profile.to_csv("cluster_profile.csv", index=False)
    print("Saved: cluster_profile.csv, silhouette_scores.csv")

# Ensure script entrypoint exists (fixes: 'main' is not defined)
if __name__ == "__main__":
    main()

# --- Change log (dev note):
# - Repaired display_dataframe_to_user indentation and I/O handling.
# - Removed erroneous top-level execution; consolidated into main().
# - Fixed PROFILE_CANDIDATES syntax and added cluster_profile_table().
# - Added main() to wire CLI → pipeline; resolves NameError and runtime issues.
