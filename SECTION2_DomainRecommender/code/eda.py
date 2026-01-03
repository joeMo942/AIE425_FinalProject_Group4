"""
Exploratory Data Analysis (EDA)
===============================
Team Members:
- [Add your names and IDs here]

Performs exploratory analysis for Part 1 of the assignment:
- Rating distribution
- User activity distribution (long-tail)
- Item popularity distribution
- Sparsity analysis
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RATINGS_FILE = DATA_DIR / "final_ratings.csv"
ITEMS_FILE = DATA_DIR / "final_items.csv"


def load_data():
    """Load final datasets."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    df_ratings = pd.read_csv(RATINGS_FILE)
    df_items = pd.read_csv(ITEMS_FILE)
    
    print(f"[LOADED] Ratings: {len(df_ratings):,}")
    print(f"[LOADED] Items: {len(df_items):,}")
    
    return df_ratings, df_items


def basic_statistics(df_ratings, df_items):
    """Print basic statistics."""
    print("\n" + "=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)
    
    n_users = df_ratings['user_id'].nunique()
    n_items = df_ratings['streamer_username'].nunique()
    n_ratings = len(df_ratings)
    
    # Sparsity
    possible = n_users * n_items
    sparsity = 100 * (1 - n_ratings / possible)
    density = 100 - sparsity
    
    stats = {
        'Users': n_users,
        'Items': n_items,
        'Interactions': n_ratings,
        'Sparsity (%)': round(sparsity, 4),
        'Density (%)': round(density, 4),
        'Avg ratings per user': round(n_ratings / n_users, 2),
        'Avg ratings per item': round(n_ratings / n_items, 2),
    }
    
    print("\n       Dataset Summary:")
    for key, value in stats.items():
        print(f"       {key:.<25} {value:>12,}" if isinstance(value, int) else f"       {key:.<25} {value:>12}")
    
    # Save to file
    with open(RESULTS_DIR / "Sec2_basic_statistics.txt", 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("BASIC STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n[SAVED] Sec2_basic_statistics.txt")
    
    return stats


def rating_distribution(df_ratings):
    """Analyze and plot rating distribution."""
    print("\n" + "=" * 60)
    print("RATING DISTRIBUTION")
    print("=" * 60)
    
    rating_counts = df_ratings['rating'].value_counts().sort_index()
    
    print("\n       Rating Distribution:")
    for rating, count in rating_counts.items():
        pct = 100 * count / len(df_ratings)
        bar = "█" * int(pct / 2)
        print(f"       {rating}: {count:>8,} ({pct:5.1f}%) {bar}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 5))
    
    ax.bar(rating_counts.index, rating_counts.values, color=colors, edgecolor='black')
    ax.set_xlabel('Rating', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Rating Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks([1, 2, 3, 4, 5])
    
    # Add count labels
    for i, (rating, count) in enumerate(rating_counts.items()):
        ax.text(rating, count + 1000, f'{count:,}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "Sec2_rating_distribution.png", dpi=150)
    plt.close()
    
    print(f"\n[PLOT] Sec2_rating_distribution.png")


def user_activity_distribution(df_ratings):
    """Analyze user activity (long-tail analysis)."""
    print("\n" + "=" * 60)
    print("USER ACTIVITY DISTRIBUTION (Long-Tail Analysis)")
    print("=" * 60)
    
    user_counts = df_ratings.groupby('user_id').size()
    
    print("\n       User Activity Statistics:")
    print(f"       Min ratings per user:    {user_counts.min():>8}")
    print(f"       Max ratings per user:    {user_counts.max():>8}")
    print(f"       Mean ratings per user:   {user_counts.mean():>8.2f}")
    print(f"       Median ratings per user: {user_counts.median():>8.0f}")
    
    # Percentiles
    print("\n       Percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"       {p}th percentile: {user_counts.quantile(p/100):.0f} ratings")
    
    # Long-tail check
    top_10_pct_users = int(len(user_counts) * 0.1)
    top_users = user_counts.nlargest(top_10_pct_users)
    top_coverage = top_users.sum() / user_counts.sum() * 100
    
    print(f"\n       Long-Tail Analysis:")
    print(f"       Top 10% users contribute {top_coverage:.1f}% of ratings")
    
    is_long_tail = top_coverage > 40
    print(f"       Long-tail problem: {'YES' if is_long_tail else 'NO'}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(user_counts, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Ratings per User', fontsize=12)
    ax1.set_ylabel('Number of Users', fontsize=12)
    ax1.set_title('User Activity Distribution', fontsize=14, fontweight='bold')
    ax1.axvline(user_counts.median(), color='red', linestyle='--', label=f'Median: {user_counts.median():.0f}')
    ax1.legend()
    
    # Sorted cumulative plot (long-tail visualization)
    sorted_counts = user_counts.sort_values(ascending=False).values
    cumulative = np.cumsum(sorted_counts) / sorted_counts.sum() * 100
    x = np.arange(len(cumulative)) / len(cumulative) * 100
    
    ax2.plot(x, cumulative, color='steelblue', linewidth=2)
    ax2.axhline(50, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(80, color='orange', linestyle='--', alpha=0.7)
    ax2.set_xlabel('% of Users (sorted by activity)', fontsize=12)
    ax2.set_ylabel('Cumulative % of Ratings', fontsize=12)
    ax2.set_title('User Activity Long-Tail', fontsize=14, fontweight='bold')
    ax2.fill_between(x, cumulative, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "Sec2_user_activity.png", dpi=150)
    plt.close()
    
    print(f"\n[PLOT] Sec2_user_activity.png")


def item_popularity_distribution(df_ratings):
    """Analyze item (streamer) popularity distribution."""
    print("\n" + "=" * 60)
    print("ITEM POPULARITY DISTRIBUTION")
    print("=" * 60)
    
    item_counts = df_ratings.groupby('streamer_username').size()
    
    print("\n       Item Popularity Statistics:")
    print(f"       Min ratings per item:    {item_counts.min():>8}")
    print(f"       Max ratings per item:    {item_counts.max():>8}")
    print(f"       Mean ratings per item:   {item_counts.mean():>8.2f}")
    print(f"       Median ratings per item: {item_counts.median():>8.0f}")
    
    # Top items
    print("\n       Top 10 Most Popular Streamers:")
    for streamer, count in item_counts.nlargest(10).items():
        print(f"       {streamer:.<25} {count:>8,} ratings")
    
    # Long-tail check
    top_10_pct_items = max(1, int(len(item_counts) * 0.1))
    top_items = item_counts.nlargest(top_10_pct_items)
    top_coverage = top_items.sum() / item_counts.sum() * 100
    
    print(f"\n       Long-Tail Analysis:")
    print(f"       Top 10% items receive {top_coverage:.1f}% of ratings")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sorted_counts = item_counts.sort_values(ascending=False).values
    ax.bar(range(len(sorted_counts)), sorted_counts, color='coral', alpha=0.7)
    ax.set_xlabel('Items (sorted by popularity)', fontsize=12)
    ax.set_ylabel('Number of Ratings', fontsize=12)
    ax.set_title('Item Popularity Distribution (Long-Tail)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "Sec2_item_popularity.png", dpi=150)
    plt.close()
    
    print(f"\n[PLOT] Sec2_item_popularity.png")


def language_analysis(df_items):
    """Analyze language distribution of items."""
    print("\n" + "=" * 60)
    print("LANGUAGE DISTRIBUTION")
    print("=" * 60)
    
    if 'language' not in df_items.columns:
        print("[SKIP] No language column found")
        return
    
    lang_counts = df_items['language'].value_counts()
    
    print("\n       Language Distribution:")
    for lang, count in lang_counts.head(10).items():
        pct = 100 * count / len(df_items)
        print(f"       {lang:.<20} {count:>5} ({pct:5.1f}%)")


def main():
    """Main EDA pipeline."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    # Load data
    df_ratings, df_items = load_data()
    
    # Run analyses
    basic_statistics(df_ratings, df_items)
    rating_distribution(df_ratings)
    user_activity_distribution(df_ratings)
    item_popularity_distribution(df_ratings)
    language_analysis(df_items)
    
    print("\n" + "=" * 60)
    print("[DONE] EDA complete! Check results/ folder for plots.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
