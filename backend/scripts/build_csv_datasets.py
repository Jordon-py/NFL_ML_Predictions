# build_csvs.py
"""
NFL Team-Game CSV Builder (Expert)
Author: Christopher (Jordon-py)

What this script does (CSV-only)
1) Load nflverse play-by-play via nfl-data-py
2) Aggregate to team-game logs (offense + defense)
3) Join final scores and targets
4) Add features in 3 passes:
   - Iter1: pace-normalized rates + margins
   - Iter2: rolling 3-game form + league-relative pass rate
   - Iter3: matchup differentials (final)
5) Write 4 clean CSVs in --out-dir
6) Generate schema files (.schema.json + .schema.md) for the final CSV

Education notes (read while skimming code)
- GroupBy→Agg turns plays into team-game rows
- shift(1).rolling(3) = last-3 games, no leakage
- Opponent merge uses same-season week alignment

Run:
  pip install nfl-data-py pandas numpy
  python build_csvs.py --start 2014 --end 2023 --out-dir data
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from nfl_data_py import import_pbp_data

# Call schema writer after final CSV is produced
from make_schema import write_schema_files  # <- ensure make_schema.py sits beside this script

# ----------------------------
# Modern nflverse team codes and names
# ----------------------------
TEAM_MAP = {
    "ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills",
    "CAR":"Carolina Panthers","CHI":"Chicago Bears","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns",
    "DAL":"Dallas Cowboys","DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers",
    "HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars","KC":"Kansas City Chiefs",
    "LAC":"Los Angeles Chargers","LAR":"Los Angeles Rams","MIA":"Miami Dolphins","MIN":"Minnesota Vikings",
    "NE":"New England Patriots","NO":"New Orleans Saints","NYG":"New York Giants","NYJ":"New York Jets",
    "PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers","SEA":"Seattle Seahawks","SF":"San Francisco 49ers",
    "TB":"Tampa Bay Buccaneers","TEN":"Tennessee Titans","WAS":"Washington Commanders","LV":"Las Vegas Raiders"
}
DEFAULT_TEAMS = sorted(TEAM_MAP.keys())

# ----------------------------
# Utils
# ----------------------------
def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Division with denominator guard. Returns 0.0 on bad denom."""
    out = a.astype(float) / b.replace({0: np.nan})
    return out.fillna(0.0)

def setup_logger(out_dir: Path) -> None:
    logf = out_dir / "build_csvs.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(logf, encoding="utf-8"), logging.StreamHandler()]
    )
    logging.info("Logging to %s", logf)

def attach_names(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable team names for quick analysis."""
    df = df.copy()
    df["team_name"] = df["team"].map(TEAM_MAP)
    df["opponent_name"] = df["opponent"].map(TEAM_MAP)
    return df

def validate_schema_base(df: pd.DataFrame) -> None:
    """Fast checks: primary key uniqueness and code set conformity."""
    pk = ["season","week","game_id","team"]
    dups = int(df.duplicated(subset=pk).sum())
    assert dups == 0, f"Primary key not unique. Found {dups} duplicates on {pk}"
    bad_codes = sorted(set(df["team"].unique()) - set(TEAM_MAP.keys()))
    assert not bad_codes, f"Unexpected team codes in data: {bad_codes}"

# ----------------------------
# Core builders
# ----------------------------
def load_pbp(seasons: list[int]) -> pd.DataFrame:
    logging.info("Loading pbp for seasons=%s", seasons)
    pbp = import_pbp_data(seasons, downcast=True)
    pbp = pbp[pbp["season_type"].isin(["REG","POST"])].copy()
    need = [
        "season","week","game_id","home_team","away_team","posteam","defteam",
        "yards_gained","rush_attempt","pass_attempt","touchdown","interception",
        "fumble_lost","penalty","qb_spike","qb_kneel","epa",
        "total_home_score","total_away_score"
    ]
    missing = [c for c in need if c not in pbp.columns]
    if missing:
        raise RuntimeError(f"Missing required PBP columns: {missing}")
    return pbp[need].copy()

def build_team_game(pbp_small: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Offense+Defense aggregates per team-game; attach opponent/home and scores."""
    off = (
        pbp_small.groupby(["season","week","game_id","posteam"], as_index=False)
        .agg(
            off_plays=("yards_gained","size"),
            off_yards=("yards_gained","sum"),
            off_pass_att=("pass_attempt","sum"),
            off_rush_att=("rush_attempt","sum"),
            off_tds=("touchdown","sum"),
            off_int=("interception","sum"),
            off_fumbles_lost=("fumble_lost","sum"),
            off_penalties=("penalty","sum"),
            off_qb_spike=("qb_spike","sum"),
            off_qb_kneel=("qb_kneel","sum"),
            off_epa_sum=("epa","sum"),
            off_epa_mean=("epa","mean"),
        ).rename(columns={"posteam":"team"})
    )
    defn = (
        pbp_small.groupby(["season","week","game_id","defteam"], as_index=False)
        .agg(
            def_plays_allowed=("yards_gained","size"),
            def_yards_allowed=("yards_gained","sum"),
            def_pass_att_allowed=("pass_attempt","sum"),
            def_rush_att_allowed=("rush_attempt","sum"),
            def_tds_allowed=("touchdown","sum"),
            def_int_made=("interception","sum"),
            def_fumbles_gained=("fumble_lost","sum"),
            def_penalties_committed=("penalty","sum"),
            def_epa_allowed_sum=("epa","sum"),
            def_epa_allowed_mean=("epa","mean"),
        ).rename(columns={"defteam":"team"})
    )
    tg = off.merge(defn, on=["season","week","game_id","team"], how="inner", validate="one_to_one")

    # Opponent and home flag
    meta = pbp_small[["game_id","home_team","away_team"]].drop_duplicates()
    tg = tg.merge(meta, on="game_id", how="left")
    tg["opponent"] = np.where(tg["team"] == tg["home_team"], tg["away_team"], tg["home_team"])
    tg["home"] = (tg["team"] == tg["home_team"]).astype(int)

    # Restrict to modern nflverse codes
    tg = tg[tg["team"].isin(teams)].reset_index(drop=True)

    # Scores and targets
    scores = (
        pbp_small.groupby("game_id", as_index=False)
        .agg(home_points=("total_home_score","max"), away_points=("total_away_score","max"))
    )
    tg = tg.merge(scores, on="game_id", how="left")
    tg["points_for"] = np.where(tg["team"] == tg["home_team"], tg["home_points"], tg["away_points"])
    tg["points_allowed"] = np.where(tg["team"] == tg["home_team"], tg["away_points"], tg["home_points"])
    tg["point_diff"] = tg["points_for"] - tg["points_allowed"]
    tg["win"] = (tg["point_diff"] > 0).astype(int)

    validate_schema_base(tg)
    return tg

def add_iter1(df: pd.DataFrame) -> pd.DataFrame:
    """Iteration 1: rates + margins."""
    df = df.copy()
    # Offensive rates
    df["off_ypp"]     = safe_div(df["off_yards"], df["off_plays"])
    df["off_pr"]      = safe_div(df["off_pass_att"], df["off_plays"])
    df["off_rr"]      = safe_div(df["off_rush_att"], df["off_plays"])
    df["off_td_rate"] = safe_div(df["off_tds"], df["off_plays"])
    df["off_pen_ypp"] = safe_div(df["off_penalties"], df["off_plays"])
    # Defensive allowed rates
    df["def_ypp_allowed"]     = safe_div(df["def_yards_allowed"], df["def_plays_allowed"])
    df["def_pass_share"]      = safe_div(df["def_pass_att_allowed"], df["def_plays_allowed"])
    df["def_rush_share"]      = safe_div(df["def_rush_att_allowed"], df["def_plays_allowed"])
    df["def_td_rate_allowed"] = safe_div(df["def_tds_allowed"], df["def_plays_allowed"])
    df["def_pen_ypp"]         = safe_div(df["def_penalties_committed"], df["def_plays_allowed"])
    # Margins
    df["turnovers"]       = df["off_int"] + df["off_fumbles_lost"]
    df["takeaways"]       = df["def_int_made"] + df["def_fumbles_gained"]
    df["to_margin"]       = df["takeaways"] - df["turnovers"]
    df["epa_margin"]      = df["off_epa_sum"] - df["def_epa_allowed_sum"]
    df["epa_play_margin"] = df["off_epa_mean"] - df["def_epa_allowed_mean"]
    return df

def add_iter2(df: pd.DataFrame) -> pd.DataFrame:
    """Iteration 2: rolling form + league-relative pass rate."""
    df = df.sort_values(["team","season","week"]).copy()
    roll_feats = [
        "off_ypp","off_pr","off_td_rate","off_epa_mean",
        "def_ypp_allowed","def_td_rate_allowed","def_epa_allowed_mean",
        "to_margin","epa_play_margin"
    ]
    for col in roll_feats:
        df[f"{col}_r3"] = (
            df.groupby(["team","season"])[col]
              .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )
    league_pw = (
        df.groupby(["season","week"], as_index=False)["off_pr"].mean()
          .rename(columns={"off_pr":"league_off_pr"})
    )
    df = df.merge(league_pw, on=["season","week"], how="left")
    df["off_pr_over_lg"] = df["off_pr"] - df["league_off_pr"]
    df["home_pass_interact"] = df["home"] * df["off_pr"]
    return df

def add_iter3(df: pd.DataFrame) -> pd.DataFrame:
    """Iteration 3: matchup differentials using opponent rolling defense."""
    df = df.copy()
    opp = df[["season","week","team",
              "def_ypp_allowed_r3","def_td_rate_allowed_r3","def_epa_allowed_mean_r3","off_pr_r3"]].rename(columns={
        "team":"opp_team",
        "def_ypp_allowed_r3":"opp_def_ypp_allowed_r3",
        "def_td_rate_allowed_r3":"opp_def_td_rate_allowed_r3",
        "def_epa_allowed_mean_r3":"opp_def_epa_allowed_mean_r3",
        "off_pr_r3":"opp_off_pr_r3"
    })
    df = df.merge(opp, left_on=["season","week","opponent"], right_on=["season","week","opp_team"], how="left")
    df.drop(columns=["opp_team"], inplace=True)

    df["m_diff_ypp"] = df["off_ypp_r3"] - df["opp_def_ypp_allowed_r3"]
    df["m_diff_td"]  = df["off_td_rate_r3"] - df["opp_def_td_rate_allowed_r3"]
    df["m_diff_epa"] = df["off_epa_mean_r3"] - df["opp_def_epa_allowed_mean_r3"]
    df["script_pass_edge"] = df["off_pr_r3"] - df["opp_off_pr_r3"]
    return df

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Build expert-level NFL team-game CSVs")
    ap.add_argument("--start", type=int, default=2014, help="Start season (inclusive)")
    ap.add_argument("--end", type=int, default=2023, help="End season (inclusive)")
    ap.add_argument("--out-dir", type=str, default="data", help="Output directory")
    ap.add_argument("--teams", type=str, nargs="*", default=DEFAULT_TEAMS, help="Team abbreviations to include")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(out_dir)

    seasons = list(range(args.start, args.end + 1))
    logging.info("Seasons=%s | Teams=%d | OutDir=%s", seasons, len(args.teams), out_dir)

    pbp_small = load_pbp(seasons)
    tg = build_team_game(pbp_small, args.teams)

    # Base export (+ names)
    base_cols = [
        "season","week","team","opponent","home","game_id",
        "off_plays","off_yards","off_pass_att","off_rush_att","off_tds","off_int",
        "off_fumbles_lost","off_penalties","off_qb_spike","off_qb_kneel","off_epa_sum","off_epa_mean",
        "def_plays_allowed","def_yards_allowed","def_pass_att_allowed","def_rush_att_allowed",
        "def_tds_allowed","def_int_made","def_fumbles_gained","def_penalties_committed",
        "def_epa_allowed_sum","def_epa_allowed_mean",
        "home_team","away_team","points_for","points_allowed","point_diff","win"
    ]
    base = tg[base_cols].copy()
    base = attach_names(base)
    base.to_csv(out_dir / "team_game_base.csv", index=False)
    logging.info("Wrote %s", out_dir / "team_game_base.csv")

    # Iteration 1
    iter1 = add_iter1(base)
    iter1 = attach_names(iter1)
    iter1.to_csv(out_dir / "team_game_iter1.csv", index=False)
    logging.info("Wrote %s", out_dir / "team_game_iter1.csv")

    # Iteration 2
    iter2 = add_iter2(iter1)
    iter2 = attach_names(iter2)
    iter2.to_csv(out_dir / "team_game_iter2.csv", index=False)
    logging.info("Wrote %s", out_dir / "team_game_iter2.csv")

    # Iteration 3 (final)
    iter3 = add_iter3(iter2)
    iter3 = attach_names(iter3)
    final_csv = out_dir / "team_game_iter3.csv"
    iter3.to_csv(final_csv, index=False)
    logging.info("Wrote %s", final_csv)

    # Schema export for the final dataset
    write_schema_files(final_csv, title="Team-Game Dataset Schema", required_core={
        "season","week","team","opponent","game_id","points_for","win"
    })
    logging.info("Schema files written for %s", final_csv)

if __name__ == "__main__":
    main()
