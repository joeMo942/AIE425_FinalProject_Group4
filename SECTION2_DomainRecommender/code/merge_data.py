"""
Data Merging and Final Dataset Creation
========================================
Team Members:
- [Add your names and IDs here]

Merges processed ratings with streamer metadata to create final datasets.
Supports both archive data (datasetV2.csv) and scraped data.
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

# Input files
RATINGS_FILE = DATA_DIR / "processed_ratings.csv"
ARCHIVE_FILE = DATA_DIR / "datasetV2.csv"
SCRAPED_FILE = DATA_DIR / "streamer_metadata.csv"

# Output files
FINAL_RATINGS = DATA_DIR / "final_ratings.csv"
FINAL_ITEMS = DATA_DIR / "final_items.csv"


def load_and_prepare_archive_data() -> pd.DataFrame:
    """Load streamer metadata from archive (datasetV2.csv)."""
    print("=" * 60)
    print("LOADING ARCHIVE STREAMER DATA")
    print("=" * 60)
    
    df = pd.read_csv(ARCHIVE_FILE)
    print(f"[LOADED] {ARCHIVE_FILE.name}: {len(df)} streamers")
    
    # Standardize column names
    df = df.rename(columns={
        'NAME': 'streamer_username',
        'LANGUAGE': 'language',
        'MOST_STREAMED_GAME': '1st_game',
        '2ND_MOST_STREAMED_GAME': '2nd_game',
        'TYPE': 'type',
        'RANK': 'rank',
        'AVG_VIEWERS_PER_STREAM': 'avg_viewers',
        'TOTAL_FOLLOWERS': 'followers',
        'AVERAGE_STREAM_DURATION': 'avg_stream_duration',
        'TOTAL_TIME_STREAMED': 'hours_streamed',
        'TOTAL_VIEWS': 'total_views',
    })
    
    # Clean username
    df['streamer_username'] = df['streamer_username'].str.lower().str.strip()
    
    # Create text features for TF-IDF (combine games, type, language)
    df['text_features'] = (
        df['type'].fillna('') + ' ' +
        df['1st_game'].fillna('') + ' ' +
        df['2nd_game'].fillna('') + ' ' +
        df['language'].fillna('')
    ).str.strip()
    
    print(f"[DONE] Prepared streamer metadata with text features")
    
    return df


def load_and_prepare_scraped_data() -> pd.DataFrame:
    """Load streamer metadata from web scraping (if available)."""
    if not SCRAPED_FILE.exists():
        print(f"[SKIP] Scraped data not found: {SCRAPED_FILE.name}")
        return None
    
    print("=" * 60)
    print("LOADING SCRAPED STREAMER DATA")
    print("=" * 60)
    
    df = pd.read_csv(SCRAPED_FILE)
    print(f"[LOADED] {SCRAPED_FILE.name}: {len(df)} streamers")
    
    # Clean username
    df['streamer_username'] = df['streamer_username'].str.lower().str.strip()
    
    # Create text features from bio and games
    df['text_features'] = (
        df['bio'].fillna('') + ' ' +
        df['1st_game'].fillna('') + ' ' +
        df['2nd_game'].fillna('') + ' ' +
        df['language'].fillna('')
    ).str.strip()
    
    print(f"[DONE] Prepared scraped data with text features")
    
    return df


def merge_datasets(use_scraped: bool = True) -> tuple:
    """
    Merge ratings with streamer metadata.
    
    Args:
        use_scraped: If True and scraped data exists, prefer it over archive
    
    Returns:
        (df_ratings, df_items) tuple
    """
    print("\n" + "=" * 60)
    print("MERGING DATASETS")
    print("=" * 60)
    
    # Load ratings
    df_ratings = pd.read_csv(RATINGS_FILE)
    print(f"[LOADED] Ratings: {len(df_ratings):,} interactions")
    
    # Try scraped data first, fallback to archive
    df_items = None
    if use_scraped:
        df_items = load_and_prepare_scraped_data()
    
    if df_items is None:
        df_items = load_and_prepare_archive_data()
    
    # Merge ratings with items
    print("\n" + "-" * 40)
    print("Performing inner join...")
    
    df_merged = df_ratings.merge(
        df_items[['streamer_username', 'text_features', 'language', 'rank', 
                  'avg_viewers', 'followers', '1st_game', '2nd_game']],
        on='streamer_username',
        how='inner'
    )
    
    # Stats
    n_users = df_merged['user_id'].nunique()
    n_items = df_merged['streamer_username'].nunique()
    n_ratings = len(df_merged)
    
    print(f"\n[MERGED] Results:")
    print(f"         Users:        {n_users:>10,}")
    print(f"         Items:        {n_items:>10,}")
    print(f"         Interactions: {n_ratings:>10,}")
    
    # Create final dataframes
    df_final_ratings = df_merged[['user_id', 'streamer_username', 'rating']].copy()
    
    df_final_items = df_items[df_items['streamer_username'].isin(
        df_merged['streamer_username'].unique()
    )].copy()
    
    # Validation
    print("\n" + "-" * 40)
    print("VALIDATION:")
    checks = [
        ("Users >= 5,000", n_users >= 5000),
        ("Items >= 500", n_items >= 500),
        ("Interactions >= 50,000", n_ratings >= 50000),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"       {status}: {check_name}")
        if not passed:
            all_passed = False
    
    if not all_passed:
        print("\n[WARNING] Some requirements not met. Consider using scraped data for more coverage.")
    
    return df_final_ratings, df_final_items


def save_final_datasets(df_ratings: pd.DataFrame, df_items: pd.DataFrame):
    """Save final datasets to CSV."""
    print("\n" + "=" * 60)
    print("SAVING FINAL DATASETS")
    print("=" * 60)
    
    df_ratings.to_csv(FINAL_RATINGS, index=False)
    print(f"[SAVED] {FINAL_RATINGS.name} ({len(df_ratings):,} rows)")
    
    df_items.to_csv(FINAL_ITEMS, index=False)
    print(f"[SAVED] {FINAL_ITEMS.name} ({len(df_items):,} items)")


def print_sample_data(df_ratings: pd.DataFrame, df_items: pd.DataFrame):
    """Print sample of final datasets."""
    print("\n" + "=" * 60)
    print("SAMPLE DATA")
    print("=" * 60)
    
    print("\nRatings sample:")
    print(df_ratings.head(5).to_string())
    
    print("\nItems sample:")
    display_cols = ['streamer_username', 'language', 'rank', 'avg_viewers', '1st_game', 'text_features']
    available_cols = [c for c in display_cols if c in df_items.columns]
    print(df_items[available_cols].head(3).to_string())


def main(use_scraped: bool = True):
    """Main execution pipeline."""
    print("\n" + "=" * 60)
    print("DATA MERGING PIPELINE")
    print("=" * 60)
    
    # Merge datasets
    df_ratings, df_items = merge_datasets(use_scraped=use_scraped)
    
    # Save final datasets
    save_final_datasets(df_ratings, df_items)
    
    # Print samples
    print_sample_data(df_ratings, df_items)
    
    print("\n" + "=" * 60)
    print("[DONE] Data merging complete!")
    print("=" * 60 + "\n")
    
    return df_ratings, df_items


if __name__ == "__main__":
    import sys
    
    # Check for --scraped flag
    use_scraped = '--scraped' in sys.argv or '-s' in sys.argv
    
    if use_scraped:
        print("[MODE] Using scraped data (if available)")
    else:
        print("[MODE] Using archive data")
    
    main(use_scraped=use_scraped)
