#!/usr/bin/env python
"""
Transform Dataset from Per-Team to Per-Game Format

Purpose:
--------
Transforms the existing per-team NFL dataset into the per-game format expected
by train_models.py. This script:
1. Calculates game scores from TDs, FGs, PATs, and safeties
2. Pivots from per-team rows to per-game rows with home/away structure
3. Adds home_points_for and away_points_for columns

Usage:
------
python backend/transform_dataset.py

Input:  backend/data/merged_game_features.csv (per-team format)
Output: backend/data/merged_game_features.csv (per-game format, backup created)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def calculate_team_score(row: pd.Series) -> float:
    """
    Calculate total points scored by a team based on stats.
    
    NFL Scoring:
    - Touchdown (TD) = 6 points
    - PAT (Point After Touchdown) = 1 point  
    - Field Goal (FG) = 3 points
    - Safety = 2 points (defensive stat)
    - 2-point conversion = 2 points
    
    Args:
        row: Team statistics row
        
    Returns:
        Total points scored
    """
    # Count all touchdowns (6 points each)
    tds = 0
    for col in ['passing_tds', 'rushing_tds', 'receiving_tds', 
                'special_teams_tds', 'def_tds', 'fumble_recovery_tds']:
        tds += row.get(col, 0) if pd.notna(row.get(col, 0)) else 0
    
    td_points = tds * 6
    
    # PATs (1 point each)
    pat_points = row.get('pat_made', 0) if pd.notna(row.get('pat_made', 0)) else 0
    
    # Field goals (3 points each)
    fg_points = (row.get('fg_made', 0) if pd.notna(row.get('fg_made', 0)) else 0) * 3
    
    # 2-point conversions (passing + rushing)
    two_pt = 0
    for col in ['passing_2pt_conversions', 'rushing_2pt_conversions', 
                'receiving_2pt_conversions']:
        two_pt += row.get(col, 0) if pd.notna(row.get(col, 0)) else 0
    two_pt_points = two_pt * 2
    
    # Safeties (2 points each) - these are defensive points
    safety_points = (row.get('def_safeties', 0) if pd.notna(row.get('def_safeties', 0)) else 0) * 2
    
    total = td_points + pat_points + fg_points + two_pt_points + safety_points
    
    return float(total)


def transform_dataset(input_path: Path, output_path: Path) -> None:
    """
    Transform per-team dataset to per-game format.
    
    Args:
        input_path: Path to input CSV (per-team format)
        output_path: Path to output CSV (per-game format)
    """
    log.info("Loading dataset from %s", input_path)
    df = pd.read_csv(input_path)
    
    log.info("Dataset shape: %s", df.shape)
    log.info("Columns: %d", len(df.columns))
    
    # Verify we have the expected columns
    required = ['season', 'week', 'team', 'opponent_team', 'is_home']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Calculate scores for each team row
    log.info("Calculating team scores...")
    df['points_scored'] = df.apply(calculate_team_score, axis=1)
    
    # Separate home and away games
    home_games = df[df['is_home'] == True].copy()
    away_games = df[df['is_home'] == False].copy()
    
    log.info("Home games: %d, Away games: %d", len(home_games), len(away_games))
    
    # Create merge keys
    home_games['merge_key'] = (
        home_games['season'].astype(str) + '_' +
        home_games['week'].astype(str) + '_' +
        home_games['team'].astype(str) + '_' +
        home_games['opponent_team'].astype(str)
    )
    
    away_games['merge_key'] = (
        away_games['season'].astype(str) + '_' +
        away_games['week'].astype(str) + '_' +
        away_games['opponent_team'].astype(str) + '_' +
        away_games['team'].astype(str)
    )
    
    # Rename columns for home side
    home_rename = {
        'team': 'home_team',
        'opponent_team': 'away_team',
        'points_scored': 'home_points_for'
    }
    home_games = home_games.rename(columns=home_rename)
    
    # Rename columns for away side (just need points)
    away_games = away_games.rename(columns={
        'points_scored': 'away_points_for'
    })
    
    # Merge home and away games
    log.info("Merging home and away games...")
    merged = home_games.merge(
        away_games[['merge_key', 'away_points_for']],
        on='merge_key',
        how='inner'
    )
    
    log.info("Merged dataset shape: %s", merged.shape)
    
    # Drop merge key and is_home (always True for per-game format)
    merged = merged.drop(columns=['merge_key', 'is_home'], errors='ignore')
    
    # Ensure score columns are float
    merged['home_points_for'] = merged['home_points_for'].astype(float)
    merged['away_points_for'] = merged['away_points_for'].astype(float)
    
    # Sort by season and week
    merged = merged.sort_values(['season', 'week']).reset_index(drop=True)
    
    # Verify no missing scores for historical games
    missing_scores = merged[
        (merged['home_points_for'].isna()) | (merged['away_points_for'].isna())
    ]
    if len(missing_scores) > 0:
        log.warning("Found %d games with missing scores", len(missing_scores))
        log.warning("Sample: %s", missing_scores[['season', 'week', 'home_team', 'away_team']].head())
    
    # Save transformed dataset
    log.info("Saving transformed dataset to %s", output_path)
    merged.to_csv(output_path, index=False)
    
    log.info("Transformation complete!")
    log.info("Final dataset shape: %s", merged.shape)
    log.info("Score columns: home_points_for=%s, away_points_for=%s",
             'home_points_for' in merged.columns,
             'away_points_for' in merged.columns)
    
    # Quick validation
    sample_scores = merged[['season', 'week', 'home_team', 'away_team', 
                            'home_points_for', 'away_points_for']].head(10)
    log.info("Sample scores:\n%s", sample_scores)


def main():
    """Main entry point."""
    backend_dir = Path(__file__).resolve().parent
    data_dir = backend_dir / "data"
    
    input_path = data_dir / "merged_game_features.csv"
    
    # Create backup
    backup_path = data_dir / f"merged_game_features_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)
    
    # Create backup
    log.info("Creating backup: %s", backup_path)
    import shutil
    shutil.copy(input_path, backup_path)
    
    # Transform dataset
    transform_dataset(input_path, input_path)
    
    log.info("Done! Backup saved to: %s", backup_path)


if __name__ == "__main__":
    main()
