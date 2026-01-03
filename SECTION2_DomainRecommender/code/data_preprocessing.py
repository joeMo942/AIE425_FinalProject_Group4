"""
Data Preprocessing for Live Stream Recommendation System
=========================================================
Team Members:
- [Add your names and IDs here]

This script processes the 100k Twitch interactions dataset and converts
watch time to 1-5 ratings using quantile-based bucketing.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RAW_FILE = DATA_DIR / "100k_a.csv"
OUTPUT_RATINGS = DATA_DIR / "processed_ratings.csv"
OUTPUT_STREAMERS = DATA_DIR / "unique_streamers.txt"


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Load raw interaction data from CSV."""
    print("=" * 60)
    print("PHASE 1: LOADING RAW DATA")
    print("=" * 60)
    
    df = pd.read_csv(
        filepath,
        header=None,
        names=['user_id', 'stream_id', 'streamer_username', 'time_start', 'time_stop']
    )
    
    print(f"[LOADED] {filepath.name}")
    print(f"         Rows: {len(df):,}")
    print(f"         Columns: {df.columns.tolist()}")
    
    return df


def clean_usernames(df: pd.DataFrame) -> pd.DataFrame:
    """Clean streamer usernames (lowercase, strip whitespace)."""
    print("\n" + "-" * 40)
    print("Cleaning streamer usernames...")
    
    df['streamer_username'] = df['streamer_username'].str.lower().str.strip()
    
    print(f"[DONE] Cleaned {df['streamer_username'].nunique():,} unique streamers")
    
    return df


def calculate_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate watch duration in minutes.
    Dataset uses 10-minute intervals, so duration = (stop - start) * 10
    """
    print("\n" + "-" * 40)
    print("Calculating watch duration...")
    
    df['duration_minutes'] = (df['time_stop'] - df['time_start']) * 10
    
    # Remove invalid durations (negative or zero)
    before_count = len(df)
    df = df[df['duration_minutes'] > 0].copy()
    removed = before_count - len(df)
    
    print(f"[DONE] Duration range: {df['duration_minutes'].min():.0f} - {df['duration_minutes'].max():.0f} mins")
    print(f"       Removed {removed:,} invalid rows (duration <= 0)")
    
    return df


def aggregate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate duplicate user-streamer interactions.
    Sum total watch time per user-streamer pair.
    """
    print("\n" + "-" * 40)
    print("Aggregating user-streamer interactions...")
    
    before_rows = len(df)
    
    df_agg = df.groupby(['user_id', 'streamer_username']).agg(
        total_minutes=('duration_minutes', 'sum'),
        interaction_count=('stream_id', 'count')
    ).reset_index()
    
    print(f"[DONE] {before_rows:,} rows → {len(df_agg):,} unique user-streamer pairs")
    print(f"       Avg interactions per pair: {df_agg['interaction_count'].mean():.2f}")
    
    return df_agg


def convert_to_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert total watch time to 1-5 ratings using USER-SPECIFIC MIN-MAX NORMALIZATION.
    
    Logic:
    1. For each user, find their min and max watch time.
    2. Normalize watch time to 0-1 scale: (x - min) / (max - min)
    3. Map normalized score to 1-5 rating bins.
    
    This ensures every user utilizes the full rating scale range relative to their own habits.
    """
    print("\n" + "-" * 40)
    print("Converting watch time to 1-5 ratings (User Min-Max)...")
    
    # Calculate min and max per user
    user_stats = df.groupby('user_id')['total_minutes'].agg(['min', 'max']).reset_index()
    user_stats.columns = ['user_id', 'user_min', 'user_max']
    
    df = df.merge(user_stats, on='user_id', how='left')
    
    # Calculate range
    df['user_range'] = df['user_max'] - df['user_min']
    
    # Handle users with single interaction or 0 range (assign rating 3 or 5)
    # If range is 0, it means all interactions have same duration. 
    # We'll assign rating 4 (Above Average) for these cases (arbitrary but reasonable).
    df['normalized'] = 0.75  # Default for zero range
    
    mask_range = df['user_range'] > 0
    df.loc[mask_range, 'normalized'] = (
        (df.loc[mask_range, 'total_minutes'] - df.loc[mask_range, 'user_min']) / 
        df.loc[mask_range, 'user_range']
    )
    
    # Map normalized score (0-1) to ratings 1-5
    # 0.0 - 0.2 → 1
    # 0.2 - 0.4 → 2
    # 0.4 - 0.6 → 3
    # 0.6 - 0.8 → 4
    # 0.8 - 1.0 → 5
    # usage of pd.cut for clean binning
    df['rating'] = pd.cut(
        df['normalized'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=[1, 2, 3, 4, 5],
        include_lowest=True
    ).astype(int)
    
    # Drop temp columns
    df = df.drop(columns=['user_min', 'user_max', 'user_range', 'normalized'])
    
    # Print distribution
    rating_dist = df['rating'].value_counts().sort_index()
    print("\n       Rating Distribution:")
    for rating, count in rating_dist.items():
        pct = 100 * count / len(df)
        bar = "█" * int(pct / 2)
        print(f"       {rating}: {count:>8,} ({pct:5.1f}%) {bar}")
        
    return df


def save_outputs(df: pd.DataFrame):
    """Save processed ratings and unique streamer list."""
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)
    
    # Save ratings CSV
    output_df = df[['user_id', 'streamer_username', 'rating', 'total_minutes']].copy()
    output_df.to_csv(OUTPUT_RATINGS, index=False)
    print(f"[SAVED] {OUTPUT_RATINGS.name}")
    
    # Save unique streamers list (for scraping)
    unique_streamers = df['streamer_username'].unique()
    with open(OUTPUT_STREAMERS, 'w') as f:
        for streamer in sorted(unique_streamers):
            f.write(f"{streamer}\n")
    print(f"[SAVED] {OUTPUT_STREAMERS.name} ({len(unique_streamers):,} streamers)")


def print_summary(df: pd.DataFrame):
    """Print final dataset summary."""
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    
    n_users = df['user_id'].nunique()
    n_items = df['streamer_username'].nunique()
    n_ratings = len(df)
    
    # Sparsity calculation
    possible = n_users * n_items
    sparsity = 100 * (1 - n_ratings / possible)
    
    print(f"       Users:        {n_users:>10,}  (min required: 5,000)")
    print(f"       Items:        {n_items:>10,}  (min required: 500)")
    print(f"       Interactions: {n_ratings:>10,}  (min required: 50,000)")
    print(f"       Sparsity:     {sparsity:>10.4f}%")
    
    # Validation
    print("\n" + "-" * 40)
    print("VALIDATION:")
    checks = [
        ("Users >= 5,000", n_users >= 5000),
        ("Items >= 500", n_items >= 500),
        ("Interactions >= 50,000", n_ratings >= 50000),
        ("Ratings in 1-5 range", df['rating'].min() >= 1 and df['rating'].max() <= 5),
    ]
    
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"       {status}: {check_name}")


def main():
    """Main execution pipeline."""
    print("\n" + "=" * 60)
    print("LIVE STREAM RECOMMENDATION - DATA PREPROCESSING")
    print("=" * 60)
    
    # Phase 1: Load and process data
    df = load_raw_data(RAW_FILE)
    df = clean_usernames(df)
    df = calculate_duration(df)
    df = aggregate_interactions(df)
    df = convert_to_ratings(df)
    
    # Save outputs
    save_outputs(df)
    
    # Print summary
    print_summary(df)
    
    print("\n" + "=" * 60)
    print("[DONE] Data preprocessing complete!")
    print("=" * 60 + "\n")
    
    return df


if __name__ == "__main__":
    main()
