"""
Data Preprocessing, Merging, and EDA Pipeline
=============================================
This script performs the entire data preparation pipeline for Section 2:
1. Preprocessing: Converts raw logs to ratings.
2. Merging: Enriches ratings with streamer metadata.
3. EDA: Generates statistics and visualizations.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Input Files
RAW_FILE = DATA_DIR / "100k_a.csv"
ARCHIVE_FILE = DATA_DIR / "datasetV2.csv"
SCRAPED_FILE = DATA_DIR / "streamer_metadata.csv"
GAME_METADATA_FILE = DATA_DIR / "game_metadata.csv"

# Intermediate Output Files
OUTPUT_RATINGS = DATA_DIR / "processed_ratings.csv"
OUTPUT_STREAMERS = DATA_DIR / "unique_streamers.txt"

# Final Output Files
FINAL_RATINGS = DATA_DIR / "final_ratings.csv"
FINAL_ITEMS = DATA_DIR / "final_items_enriched.csv"


# ============================================================================
# PART 1: PREPROCESSING (Raw Logs -> Ratings)
# ============================================================================

def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Load raw interaction data from CSV."""
    print("=" * 60)
    print("PHASE 1: LOADING RAW DATA")
    print("=" * 60)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found: {filepath}")

    df = pd.read_csv(
        filepath,
        header=None,
        names=['user_id', 'stream_id', 'streamer_username', 'time_start', 'time_stop']
    )
    print(f"[LOADED] {filepath.name}")
    print(f"         Rows: {len(df):,}")
    return df

def clean_usernames(df: pd.DataFrame) -> pd.DataFrame:
    """Clean streamer usernames."""
    print("\nCleaning streamer usernames...")
    df['streamer_username'] = df['streamer_username'].str.lower().str.strip()
    return df

def calculate_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate watch duration."""
    print("Calculating watch duration...")
    df['duration_minutes'] = (df['time_stop'] - df['time_start']) * 10
    df = df[df['duration_minutes'] > 0].copy()
    return df

def aggregate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate duplicate user-streamer interactions."""
    print("Aggregating interactions...")
    df_agg = df.groupby(['user_id', 'streamer_username']).agg(
        total_minutes=('duration_minutes', 'sum'),
        interaction_count=('stream_id', 'count')
    ).reset_index()
    return df_agg

def convert_to_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert watch time to 1-5 ratings using user-specific log-min-max normalization."""
    print("Converting to ratings...")
    
    # Log transformation
    df['log_minutes'] = np.log1p(df['total_minutes'])
    
    # Min-max per user
    df['user_log_min'] = df.groupby('user_id')['log_minutes'].transform('min')
    df['user_log_max'] = df.groupby('user_id')['log_minutes'].transform('max')
    df['user_log_range'] = df['user_log_max'] - df['user_log_min']
    
    df['normalized'] = 0.75  # Default
    mask_range = df['user_log_range'] > 0
    df.loc[mask_range, 'normalized'] = (
        (df.loc[mask_range, 'log_minutes'] - df.loc[mask_range, 'user_log_min']) / 
        df.loc[mask_range, 'user_log_range']
    )
    
    # Map to buckets
    df['rating'] = pd.cut(
        df['normalized'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=[1, 2, 3, 4, 5],
        include_lowest=True
    ).astype(int)
    
    return df.drop(columns=['log_minutes', 'user_log_min', 'user_log_max', 'user_log_range', 'normalized'])

def save_intermediate_outputs(df: pd.DataFrame):
    """Save processed ratings."""
    print("\nSaving intermediate processed ratings...")
    df[['user_id', 'streamer_username', 'rating', 'total_minutes']].to_csv(OUTPUT_RATINGS, index=False)
    
    with open(OUTPUT_STREAMERS, 'w') as f:
        for streamer in sorted(df['streamer_username'].unique()):
            f.write(f"{streamer}\n")
    print(f"[SAVED] {OUTPUT_RATINGS.name}")


# ============================================================================
# PART 2: MERGING (Ratings + Metadata -> Final Dataset)
# ============================================================================

def load_and_prepare_archive_data() -> pd.DataFrame:
    """Load and prep archive metadata."""
    print("\nLoading archive data...")
    df = pd.read_csv(ARCHIVE_FILE)
    
    df = df.rename(columns={
        'NAME': 'streamer_username', 'LANGUAGE': 'language',
        'MOST_STREAMED_GAME': '1st_game', '2ND_MOST_STREAMED_GAME': '2nd_game',
        'TYPE': 'type', 'RANK': 'rank', 'AVG_VIEWERS_PER_STREAM': 'avg_viewers',
        'TOTAL_FOLLOWERS': 'followers'
    })
    df['streamer_username'] = df['streamer_username'].str.lower().str.strip()
    
    # Simple text features for archive data
    df['text_features'] = (
        df['type'].fillna('') + ' ' + df['1st_game'].fillna('') + ' ' +
        df['2nd_game'].fillna('') + ' ' + df['language'].fillna('')
    ).str.strip()
    return df

def load_and_prepare_scraped_data() -> pd.DataFrame:
    """Load and prep scraped data (with IGDB enrichment)."""
    if not SCRAPED_FILE.exists():
        return None
        
    print("Loading scraped data...")
    df = pd.read_csv(SCRAPED_FILE)
    
    df = df.rename(columns={
        'NAME': 'streamer_username', 'LANGUAGE': 'language',
        'MOST_STREAMED_GAME': '1st_game', '2ND_MOST_STREAMED_GAME': '2nd_game',
        'RANK': 'rank', 'AVG_VIEWERS_PER_STREAM': 'avg_viewers',
        'TOTAL_FOLLOWERS': 'followers'
    })
    df['streamer_username'] = df['streamer_username'].astype(str).str.lower().str.strip()
    
    # Load IGDB enrichment
    game_lookup = {}
    if GAME_METADATA_FILE.exists():
        print(f"[ENRICH] Loading game metadata...")
        df_games = pd.read_csv(GAME_METADATA_FILE)
        for _, row in df_games.iterrows():
            gname = str(row['game_name']).strip()
            desc = f"{row['summary']} {row['genres']} {row['themes']} {row['keywords']}"
            game_lookup[gname] = str(desc).replace('nan', '').strip()
            
    # Create enriched text features
    def create_features(row):
        base = f"{row.get('language', '')} {row.get('1st_game', '')} {row.get('2nd_game', '')}"
        enrich1 = game_lookup.get(str(row.get('1st_game', '')), '')
        enrich2 = game_lookup.get(str(row.get('2nd_game', '')), '')
        return f"{base} {enrich1} {enrich2}".strip()

    df['text_features'] = df.apply(create_features, axis=1)
    return df

def merge_datasets(df_ratings: pd.DataFrame, use_scraped: bool = True):
    """Merge ratings with best available metadata."""
    print("=" * 60)
    print("PHASE 2: MERGING DATASETS")
    print("=" * 60)
    
    df_items = None
    if use_scraped:
        df_items = load_and_prepare_scraped_data()
    
    if df_items is None:
        print("[INFO] Using archive data as fallback.")
        df_items = load_and_prepare_archive_data()
        
    print("Performing inner join...")
    df_merged = df_ratings.merge(
        df_items[['streamer_username', 'text_features', 'language', 'rank', 
                  'avg_viewers', 'followers', '1st_game', '2nd_game']],
        on='streamer_username',
        how='inner'
    )
    
    df_final_ratings = df_merged[['user_id', 'streamer_username', 'rating']].copy()
    df_final_items = df_items[df_items['streamer_username'].isin(
        df_merged['streamer_username'].unique()
    )].copy()
    
    print(f"[MERGED] Final Users: {df_final_ratings['user_id'].nunique():,}")
    print(f"[MERGED] Final Items: {df_final_ratings['streamer_username'].nunique():,}")
    
    return df_final_ratings, df_final_items

def save_final_datasets(df_ratings, df_items):
    print("\nSaving final datasets...")
    df_ratings.to_csv(FINAL_RATINGS, index=False)
    df_items.to_csv(FINAL_ITEMS, index=False)
    print(f"[SAVED] {FINAL_RATINGS.name}")
    print(f"[SAVED] {FINAL_ITEMS.name}")


# ============================================================================
# PART 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

def run_eda(df_ratings, df_items):
    """Run Exploratory Data Analysis."""
    print("=" * 60)
    print("PHASE 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Basic Stats
    n_users = df_ratings['user_id'].nunique()
    n_items = df_ratings['streamer_username'].nunique()
    sparsity = 100 * (1 - len(df_ratings) / (n_users * n_items))
    
    print(f"Sparsity: {sparsity:.4f}%")
    with open(RESULTS_DIR / "Sec2_basic_statistics.txt", 'w') as f:
        f.write(f"Users: {n_users}\nItems: {n_items}\nSparsity: {sparsity:.4f}%\n")
    
    # Rating Distribution
    counts = df_ratings['rating'].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    counts.plot(kind='bar', color='coral', edgecolor='black')
    plt.title('Rating Distribution')
    plt.xlabel('Rating'); plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "Sec2_rating_distribution.png")
    plt.close()
    print("[PLOT] Rating distribution saved.")
    
    # User Activity Long-Tail
    user_counts = df_ratings.groupby('user_id').size().sort_values(ascending=False)
    cumulative = np.cumsum(user_counts.values) / user_counts.sum()
    
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(cumulative)) / len(cumulative) * 100, cumulative * 100)
    plt.axhline(80, color='red', linestyle='--')
    plt.title('User Activity (Long-Tail)')
    plt.xlabel('% Users'); plt.ylabel('% Ratings')
    plt.grid(True)
    plt.savefig(RESULTS_DIR / "Sec2_user_activity.png")
    plt.close()
    print("[PLOT] User activity saved.")
    
    # Item Popularity
    item_counts = df_ratings.groupby('streamer_username').size().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(item_counts)), item_counts.values, color='steelblue')
    plt.title('Item Popularity')
    plt.xlabel('Items'); plt.ylabel('Ratings')
    plt.savefig(RESULTS_DIR / "Sec2_item_popularity.png")
    plt.close()
    print("[PLOT] Item popularity saved.")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("STARTING FULL PREPROCESSING PIPELINE...")
    
    # 1. Preprocess
    df_raw = load_raw_data(RAW_FILE)
    df_clean = clean_usernames(df_raw)
    df_dur = calculate_duration(df_clean)
    df_agg = aggregate_interactions(df_dur)
    df_processed = convert_to_ratings(df_agg)
    save_intermediate_outputs(df_processed)
    
    # 2. Merge
    df_final_ratings, df_final_items = merge_datasets(df_processed, use_scraped=True)
    save_final_datasets(df_final_ratings, df_final_items)
    
    # 3. EDA
    run_eda(df_final_ratings, df_final_items)
    
    print("\n[DONE] Pipeline execution complete!")

if __name__ == "__main__":
    main()
