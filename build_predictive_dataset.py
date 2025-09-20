#!/usr/bin/env python3
"""
build_predictive_dataset.py
============================

Purpose
-------
Merge NFL play-by-play data with player tracking data to build a dataset with 
highly predictive properties. This script performs feature engineering and data 
cleaning to create a comprehensive predictive dataset.

Key Functions
-------------
- load_data(): Load play-by-play and player tracking data
- engineer_features(): Create new predictive features (offensive_epa, play_result)
- merge_datasets(): Merge datasets on game_id and play_id
- clean_data(): Handle missing values and data quality issues
- save_dataset(): Save the final predictive dataset

External Dependencies
---------------------
pandas, numpy, logging

Usage
-----
python build_predictive_dataset.py [--data-dir <path>] [--output-dir <path>]

Output
------
Saves predictive_nfl_dataset.csv to the specified output directory
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path) -> None:
    """Set up file and console logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'build_predictive_dataset.log'
    
    # Add file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add to logger
    logger.addHandler(file_handler)
    logger.info(f"Logging initialized. Log file: {log_file}")


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load play-by-play and player tracking data from CSV files.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing the data files
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        play_by_play_df, player_tracking_df
    """
    logger.info("Loading data from %s", data_dir)
    
    # Load play-by-play data
    pbp_file = data_dir / 'play_by_play.csv'
    tracking_file = data_dir / 'player_tracking.csv'
    
    if not pbp_file.exists():
        logger.warning("play_by_play.csv not found. Creating sample data structure...")
        play_by_play_df = create_sample_play_by_play()
    else:
        logger.info("Loading play_by_play.csv")
        play_by_play_df = pd.read_csv(pbp_file)
    
    if not tracking_file.exists():
        logger.warning("player_tracking.csv not found. Creating sample data structure...")
        player_tracking_df = create_sample_player_tracking()
    else:
        logger.info("Loading player_tracking.csv")
        player_tracking_df = pd.read_csv(tracking_file)
    
    logger.info("Loaded %d play-by-play records and %d tracking records", 
                len(play_by_play_df), len(player_tracking_df))
    
    return play_by_play_df, player_tracking_df


def create_sample_play_by_play() -> pd.DataFrame:
    """Create sample play-by-play data structure for demonstration."""
    logger.info("Creating sample play-by-play data")
    
    np.random.seed(42)  # For reproducible results
    n_plays = 1000
    
    game_ids = [f"2024_{week:02d}_{home}_{away}" 
                for week in range(1, 18) 
                for home in ['KC', 'BUF', 'CIN', 'BAL', 'PIT']
                for away in ['LAR', 'SF', 'SEA', 'ARI', 'LV']][:50]
    
    data = []
    play_id = 1
    
    for game_id in game_ids:
        plays_in_game = np.random.randint(120, 180)  # Typical plays per game
        
        for _ in range(plays_in_game):
            # Extract teams from game_id
            parts = game_id.split('_')
            home_team = parts[2]
            away_team = parts[3]
            
            play_data = {
                'game_id': game_id,
                'play_id': play_id,
                'season': 2024,
                'week': int(parts[1]),
                'quarter': np.random.randint(1, 5),
                'down': np.random.choice([1, 2, 3, 4], p=[0.35, 0.25, 0.25, 0.15]),
                'yards_to_go': np.random.randint(1, 21),
                'yardline_100': np.random.randint(1, 100),
                'home_team': home_team,
                'away_team': away_team,
                'posteam': np.random.choice([home_team, away_team]),
                'play_type': np.random.choice(['pass', 'run', 'punt', 'field_goal'], 
                                            p=[0.6, 0.35, 0.03, 0.02]),
                'yards_gained': np.random.randint(-5, 25),
                'touchdown': np.random.choice([0, 1], p=[0.95, 0.05]),
                'interception': np.random.choice([0, 1], p=[0.97, 0.03]),
                'fumble': np.random.choice([0, 1], p=[0.98, 0.02]),
                'sack': np.random.choice([0, 1], p=[0.93, 0.07]),
                'penalty': np.random.choice([0, 1], p=[0.9, 0.1]),
                'epa': np.random.normal(0, 2),  # Expected Points Added
                'wp': np.random.uniform(0, 1),  # Win Probability
                'wpa': np.random.normal(0, 0.1),  # Win Probability Added
            }
            data.append(play_data)
            play_id += 1
    
    return pd.DataFrame(data)


def create_sample_player_tracking() -> pd.DataFrame:
    """Create sample player tracking data structure for demonstration."""
    logger.info("Creating sample player tracking data")
    
    np.random.seed(42)
    n_tracking_records = 5000
    
    # Create tracking data that can be linked to plays
    data = []
    
    for i in range(n_tracking_records):
        # Generate game_id and play_id that match some play-by-play records
        game_week = np.random.randint(1, 18)
        teams = ['KC', 'BUF', 'CIN', 'BAL', 'PIT', 'LAR', 'SF', 'SEA', 'ARI', 'LV']
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])
        game_id = f"2024_{game_week:02d}_{home_team}_{away_team}"
        
        tracking_data = {
            'game_id': game_id,
            'play_id': np.random.randint(1, 1000),
            'player_id': f"player_{np.random.randint(1, 100)}",
            'position': np.random.choice(['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'CB', 'S']),
            'team': np.random.choice([home_team, away_team]),
            'x_position': np.random.uniform(0, 120),  # Field position x
            'y_position': np.random.uniform(0, 53.3),  # Field position y
            'speed': np.random.uniform(0, 25),  # mph
            'acceleration': np.random.uniform(-5, 5),  # mph/s
            'distance_traveled': np.random.uniform(0, 50),  # yards
            'max_speed': np.random.uniform(10, 25),  # mph
            'time_to_tackle': np.random.uniform(0, 5),  # seconds
            'separation_distance': np.random.uniform(0, 15),  # yards from nearest defender
            'pressure_rate': np.random.uniform(0, 1),  # QB pressure metric
            'coverage_rating': np.random.uniform(0, 1),  # Defensive coverage metric
        }
        data.append(tracking_data)
    
    return pd.DataFrame(data)


def engineer_features(play_by_play_df: pd.DataFrame, player_tracking_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Engineer new predictive features for both datasets.
    
    Parameters
    ----------
    play_by_play_df : pd.DataFrame
        Play-by-play data
    player_tracking_df : pd.DataFrame
        Player tracking data
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Enhanced play_by_play_df, enhanced player_tracking_df
    """
    logger.info("Engineering features...")
    
    # Feature 1: Offensive EPA (Expected Points Added for offensive team)
    logger.info("Creating offensive_epa feature")
    play_by_play_df['offensive_epa'] = play_by_play_df.apply(
        lambda row: row['epa'] if row['posteam'] == row['home_team'] 
        else -row['epa'], axis=1
    )
    
    # Feature 2: Play Result (comprehensive outcome classification)
    logger.info("Creating play_result feature")
    def categorize_play_result(row):
        if row['touchdown'] == 1:
            return 'touchdown'
        elif row['interception'] == 1:
            return 'interception'
        elif row['fumble'] == 1:
            return 'fumble'
        elif row['sack'] == 1:
            return 'sack'
        elif row['penalty'] == 1:
            return 'penalty'
        elif row['yards_gained'] >= row['yards_to_go']:
            return 'first_down'
        elif row['yards_gained'] > 0:
            return 'positive_gain'
        elif row['yards_gained'] == 0:
            return 'no_gain'
        else:
            return 'negative_gain'
    
    play_by_play_df['play_result'] = play_by_play_df.apply(categorize_play_result, axis=1)
    
    # Additional play-by-play features
    play_by_play_df['red_zone'] = (play_by_play_df['yardline_100'] <= 20).astype(int)
    play_by_play_df['goal_to_go'] = (play_by_play_df['yards_to_go'] >= play_by_play_df['yardline_100']).astype(int)
    play_by_play_df['long_yardage'] = (play_by_play_df['yards_to_go'] >= 7).astype(int)
    play_by_play_df['scoring_drive'] = play_by_play_df.groupby(['game_id', 'posteam'])['touchdown'].transform('max')
    
    # Player tracking features
    logger.info("Enhancing player tracking features")
    
    # Speed differential (difference from average speed)
    player_tracking_df['speed_differential'] = (
        player_tracking_df['speed'] - player_tracking_df.groupby('position')['speed'].transform('mean')
    )
    
    # Explosive play indicator (high speed + distance)
    player_tracking_df['explosive_play'] = (
        (player_tracking_df['speed'] > 15) & (player_tracking_df['distance_traveled'] > 20)
    ).astype(int)
    
    # Pressure situation for QBs
    qb_mask = player_tracking_df['position'] == 'QB'
    player_tracking_df.loc[qb_mask, 'under_pressure'] = (
        player_tracking_df.loc[qb_mask, 'pressure_rate'] > 0.3
    ).astype(int)
    
    # Defensive advantage (good coverage + close to action)
    def_positions = ['CB', 'S', 'LB']
    def_mask = player_tracking_df['position'].isin(def_positions)
    player_tracking_df.loc[def_mask, 'defensive_advantage'] = (
        (player_tracking_df.loc[def_mask, 'coverage_rating'] > 0.7) &
        (player_tracking_df.loc[def_mask, 'separation_distance'] < 5)
    ).astype(int)
    
    logger.info("Feature engineering completed")
    return play_by_play_df, player_tracking_df


def merge_datasets(play_by_play_df: pd.DataFrame, player_tracking_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge play-by-play and player tracking datasets on game_id and play_id.
    
    Parameters
    ----------
    play_by_play_df : pd.DataFrame
        Enhanced play-by-play data
    player_tracking_df : pd.DataFrame
        Enhanced player tracking data
        
    Returns
    -------
    pd.DataFrame
        Merged dataset
    """
    logger.info("Merging datasets on game_id and play_id...")
    
    # Aggregate tracking data by play to create play-level features
    logger.info("Aggregating player tracking data by play...")
    
    tracking_agg = player_tracking_df.groupby(['game_id', 'play_id']).agg({
        'speed': ['mean', 'max', 'std'],
        'acceleration': ['mean', 'max', 'min'],
        'distance_traveled': ['sum', 'mean', 'max'],
        'max_speed': ['mean', 'max'],
        'separation_distance': ['mean', 'min'],
        'pressure_rate': 'mean',
        'coverage_rating': 'mean',
        'speed_differential': 'mean',
        'explosive_play': 'sum',
        'under_pressure': 'max',
        'defensive_advantage': 'sum'
    }).reset_index()
    
    # Flatten column names
    tracking_agg.columns = [
        '_'.join(col).strip() if col[1] else col[0] 
        for col in tracking_agg.columns.values
    ]
    
    # Rename columns for clarity
    column_mapping = {
        'game_id_': 'game_id',
        'play_id_': 'play_id',
        'speed_mean': 'avg_speed',
        'speed_max': 'max_speed_play',
        'speed_std': 'speed_variance',
        'acceleration_mean': 'avg_acceleration',
        'acceleration_max': 'max_acceleration',
        'acceleration_min': 'min_acceleration',
        'distance_traveled_sum': 'total_distance',
        'distance_traveled_mean': 'avg_distance',
        'distance_traveled_max': 'max_distance',
        'max_speed_mean': 'avg_player_max_speed',
        'max_speed_max': 'play_max_speed',
        'separation_distance_mean': 'avg_separation',
        'separation_distance_min': 'min_separation',
        'pressure_rate_mean': 'avg_pressure',
        'coverage_rating_mean': 'avg_coverage',
        'speed_differential_mean': 'avg_speed_diff',
        'explosive_play_sum': 'explosive_plays_count',
        'under_pressure_max': 'qb_under_pressure',
        'defensive_advantage_sum': 'def_advantage_count'
    }
    
    tracking_agg = tracking_agg.rename(columns=column_mapping)
    
    logger.info("Performing merge...")
    # Merge on game_id and play_id
    merged_df = pd.merge(
        play_by_play_df, 
        tracking_agg, 
        on=['game_id', 'play_id'], 
        how='left'
    )
    
    logger.info("Merge completed. Result shape: %s", merged_df.shape)
    logger.info("Merge statistics:")
    logger.info("- Total plays in play-by-play: %d", len(play_by_play_df))
    logger.info("- Unique plays in tracking: %d", len(tracking_agg))
    logger.info("- Plays with tracking data: %d", merged_df.dropna(subset=['avg_speed']).shape[0])
    logger.info("- Plays without tracking data: %d", merged_df.isna().sum()['avg_speed'])
    
    return merged_df


def clean_data(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the merged dataset by handling missing values and data quality issues.
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged dataset
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataset
    """
    logger.info("Cleaning merged dataset...")
    logger.info("Initial dataset shape: %s", merged_df.shape)
    
    # Log missing values
    missing_values = merged_df.isnull().sum()
    logger.info("Missing values summary:")
    for col, count in missing_values[missing_values > 0].items():
        logger.info("- %s: %d (%.2f%%)", col, count, 100 * count / len(merged_df))
    
    # Fill missing tracking data with play-level defaults
    tracking_columns = [
        'avg_speed', 'max_speed_play', 'speed_variance', 'avg_acceleration',
        'max_acceleration', 'min_acceleration', 'total_distance', 'avg_distance',
        'max_distance', 'avg_player_max_speed', 'play_max_speed', 'avg_separation',
        'min_separation', 'avg_pressure', 'avg_coverage', 'avg_speed_diff',
        'explosive_plays_count', 'qb_under_pressure', 'def_advantage_count'
    ]
    
    # Fill missing values with appropriate defaults based on play type
    for col in tracking_columns:
        if col in merged_df.columns:
            if col in ['explosive_plays_count', 'qb_under_pressure', 'def_advantage_count']:
                # Count columns - fill with 0
                merged_df[col] = merged_df[col].fillna(0)
            elif 'speed' in col or 'acceleration' in col or 'distance' in col:
                # Speed/acceleration/distance - fill with play-type specific medians
                for play_type in merged_df['play_type'].unique():
                    mask = (merged_df['play_type'] == play_type) & merged_df[col].notna()
                    if mask.sum() > 0:
                        median_val = merged_df.loc[mask, col].median()
                        merged_df.loc[(merged_df['play_type'] == play_type) & merged_df[col].isna(), col] = median_val
                
                # Fill remaining with overall median
                merged_df[col] = merged_df[col].fillna(merged_df[col].median())
            else:
                # Other columns - fill with mean
                merged_df[col] = merged_df[col].fillna(merged_df[col].mean())
    
    # Data quality checks and corrections
    logger.info("Performing data quality checks...")
    
    # Remove obviously invalid plays
    initial_count = len(merged_df)
    
    # Remove plays with impossible yard gains (e.g., > 99 yards)
    merged_df = merged_df[merged_df['yards_gained'].between(-20, 99)]
    
    # Remove plays with invalid field position
    merged_df = merged_df[merged_df['yardline_100'].between(1, 99)]
    
    # Remove plays with invalid down
    merged_df = merged_df[merged_df['down'].isin([1, 2, 3, 4])]
    
    # Fix logical inconsistencies
    # If touchdown = 1, yards_gained should be >= yards to goal line
    td_mask = merged_df['touchdown'] == 1
    merged_df.loc[td_mask, 'yards_gained'] = np.maximum(
        merged_df.loc[td_mask, 'yards_gained'],
        merged_df.loc[td_mask, 'yardline_100']
    )
    
    logger.info("Removed %d invalid records during quality checks", initial_count - len(merged_df))
    
    # Create additional derived features after cleaning
    merged_df['yards_per_play'] = merged_df['yards_gained']
    merged_df['success_rate'] = ((merged_df['yards_gained'] >= merged_df['yards_to_go']) | 
                                (merged_df['touchdown'] == 1)).astype(int)
    
    # Normalize some features
    merged_df['epa_normalized'] = (merged_df['epa'] - merged_df['epa'].mean()) / merged_df['epa'].std()
    merged_df['wp_change'] = merged_df['wpa']  # Already represents change in win probability
    
    logger.info("Data cleaning completed. Final dataset shape: %s", merged_df.shape)
    
    return merged_df


def save_dataset(cleaned_df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Save the final predictive dataset to CSV.
    
    Parameters
    ----------
    cleaned_df : pd.DataFrame
        Cleaned and merged dataset
    output_dir : Path
        Output directory
        
    Returns
    -------
    Path
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'predictive_nfl_dataset.csv'
    
    logger.info("Saving dataset to %s", output_file)
    cleaned_df.to_csv(output_file, index=False)
    
    logger.info("Dataset saved successfully")
    logger.info("Final dataset statistics:")
    logger.info("- Rows: %d", len(cleaned_df))
    logger.info("- Columns: %d", len(cleaned_df.columns))
    logger.info("- Memory usage: %.2f MB", cleaned_df.memory_usage(deep=True).sum() / 1024**2)
    
    # Log feature summary
    logger.info("Feature summary:")
    logger.info("- Categorical features: %d", cleaned_df.select_dtypes(include=['object']).shape[1])
    logger.info("- Numerical features: %d", cleaned_df.select_dtypes(include=[np.number]).shape[1])
    
    return output_file


def generate_data_summary(cleaned_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate a summary report of the created dataset."""
    summary_file = output_dir / 'dataset_summary.txt'
    
    with open(summary_file, 'w') as f:
        f.write("NFL Predictive Dataset Summary\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Dataset Shape: {cleaned_df.shape}\n")
        f.write(f"Date Created: {pd.Timestamp.now()}\n\n")
        
        f.write("Column Information:\n")
        f.write("-" * 20 + "\n")
        for col in cleaned_df.columns:
            dtype = cleaned_df[col].dtype
            null_count = cleaned_df[col].isnull().sum()
            unique_count = cleaned_df[col].nunique()
            f.write(f"{col}: {dtype}, {null_count} nulls, {unique_count} unique values\n")
        
        f.write("\nEngineered Features:\n")
        f.write("-" * 20 + "\n")
        f.write("- offensive_epa: Expected Points Added from offensive team perspective\n")
        f.write("- play_result: Comprehensive play outcome classification\n")
        f.write("- avg_speed: Average player speed during play\n")
        f.write("- explosive_plays_count: Number of high-speed, long-distance plays\n")
        f.write("- qb_under_pressure: Whether QB was under pressure\n")
        f.write("- def_advantage_count: Number of defensive players in advantageous position\n")
        
        f.write(f"\nPlay Result Distribution:\n")
        f.write("-" * 30 + "\n")
        result_counts = cleaned_df['play_result'].value_counts()
        for result, count in result_counts.items():
            f.write(f"{result}: {count} ({100*count/len(cleaned_df):.1f}%)\n")
    
    logger.info("Dataset summary saved to %s", summary_file)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Build predictive NFL dataset by merging play-by-play and player tracking data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_predictive_dataset.py
  python build_predictive_dataset.py --data-dir /path/to/data --output-dir /path/to/output
  
For more information, see the README.md file.
        """
    )
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='data',
        help='Directory containing input data files (default: data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Directory to save output files (default: data)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Adjust logging level if verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup logging
    setup_logging(output_dir)
    
    try:
        logger.info("Starting predictive dataset building process...")
        logger.info("Data directory: %s", data_dir.absolute())
        logger.info("Output directory: %s", output_dir.absolute())
        
        # Validate input directory exists
        if not data_dir.exists():
            logger.warning("Data directory %s does not exist. Creating it...", data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load data
        logger.info("Step 1/6: Loading data...")
        play_by_play_df, player_tracking_df = load_data(data_dir)
        
        # Step 2: Engineer features  
        logger.info("Step 2/6: Engineering features...")
        play_by_play_df, player_tracking_df = engineer_features(play_by_play_df, player_tracking_df)
        
        # Step 3: Merge datasets
        logger.info("Step 3/6: Merging datasets...")
        merged_df = merge_datasets(play_by_play_df, player_tracking_df)
        
        # Step 4: Clean data
        logger.info("Step 4/6: Cleaning data...")
        cleaned_df = clean_data(merged_df)
        
        # Step 5: Save dataset
        logger.info("Step 5/6: Saving dataset...")
        output_file = save_dataset(cleaned_df, output_dir)
        
        # Step 6: Generate summary
        logger.info("Step 6/6: Generating summary...")
        generate_data_summary(cleaned_df, output_dir)
        
        logger.info("🎉 Process completed successfully!")
        logger.info("📁 Output file: %s", output_file)
        logger.info("📊 Dataset shape: %s", cleaned_df.shape)
        logger.info("💾 File size: %.2f MB", output_file.stat().st_size / 1024**2)
        
        return output_file
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("❌ Process failed with error: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()