"""
================================================================================
SECTION 1: Dataset Preprocessing and Statistical Analysis
AIE425 Final Project - Dimensionality Reduction and Matrix Factorization
================================================================================
Team Members:
- [Add team member names and IDs here]
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import pickle

# ============================================================================
# CONFIGURATION
# ============================================================================
RESULTS_DIR = "../results"
PLOTS_DIR = "../plots"
TABLES_DIR = "../tables"
DATA_DIR = "../data"

# Create output directories
for dir_path in [RESULTS_DIR, PLOTS_DIR, TABLES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def round_val(value, decimals=2):
    """Round value to specified decimal places."""
    return round(float(value), decimals)

def print_header(title):
    """Print a major section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_subheader(title):
    """Print a sub-section header."""
    print(f"\n--- {title} ---")

def print_kv(key, value, width=35):
    """Print key-value pair with alignment."""
    print(f"  {key:<{width}} : {value}")

# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================

def load_movielens_data(data_path):
    """
    Load MovieLens dataset.
    Supports multiple formats: .dat, .csv
    Expected columns: userId, movieId, rating, timestamp
    """
    print_header("SECTION 1: DATA LOADING")
    
    # Try to find ratings file
    possible_files = [
        os.path.join(data_path, "ratings.dat"),
        os.path.join(data_path, "ratings.csv"),
        os.path.join(data_path, "ml-10M100K", "ratings.dat"),
    ]
    
    ratings_file = None
    for f in possible_files:
        if os.path.exists(f):
            ratings_file = f
            break
    
    if ratings_file is None:
        print("[ERROR] No ratings file found!")
        print("Please download MovieLens 10M dataset from:")
        print("https://grouplens.org/datasets/movielens/10m/")
        print(f"Extract to: {data_path}")
        return None
    
    print(f"[INFO] Loading data from: {ratings_file}")
    
    # Load based on file extension
    if ratings_file.endswith(".dat"):
        df = pd.read_csv(ratings_file, sep="::", 
                         names=["userId", "movieId", "rating", "timestamp"],
                         engine="python")
    else:
        df = pd.read_csv(ratings_file)
    
    print(f"[DONE] Loaded {len(df):,} ratings")
    return df

# ============================================================================
# SECTION 2: DATASET VALIDATION
# ============================================================================

def validate_dataset(df):
    """
    Validate dataset meets minimum requirements:
    - ≥10,000 users
    - ≥500 items
    - ≥100,000 interactions/ratings
    - Ratings on 1-5 scale
    """
    print_header("SECTION 2: DATASET VALIDATION")
    
    n_users = df['userId'].nunique()
    n_items = df['movieId'].nunique()
    n_ratings = len(df)
    rating_min = df['rating'].min()
    rating_max = df['rating'].max()
    
    print_subheader("Dataset Statistics")
    print_kv("Number of users", f"{n_users:,}")
    print_kv("Number of items", f"{n_items:,}")
    print_kv("Number of ratings", f"{n_ratings:,}")
    print_kv("Rating range", f"{rating_min} - {rating_max}")
    
    # Validation checks
    print_subheader("Validation Results")
    
    checks = [
        ("Users ≥ 10,000", n_users >= 10000, n_users),
        ("Items ≥ 500", n_items >= 500, n_items),
        ("Ratings ≥ 100,000", n_ratings >= 100000, n_ratings),
        ("Rating scale 1-5", rating_min >= 1 and rating_max <= 5, f"{rating_min}-{rating_max}"),
    ]
    
    all_passed = True
    for check_name, passed, value in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print_kv(check_name, f"{status} ({value})")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n[DONE] Dataset meets all requirements!")
    else:
        print("\n[WARNING] Dataset does not meet all requirements!")
    
    return all_passed, n_users, n_items, n_ratings

# ============================================================================
# SECTION 3: STATISTICAL ANALYSIS
# ============================================================================

def compute_statistical_analysis(df, n_users, n_items, n_ratings):
    """
    Perform complete statistical analysis as in Assignment 1, Section ONE.
    """
    print_header("SECTION 3: STATISTICAL ANALYSIS")
    
    stats = {}
    
    # 3.1 Rating Distribution Statistics
    print_subheader("3.1 Rating Distribution")
    rating_stats = {
        "mean": round_val(df['rating'].mean()),
        "std": round_val(df['rating'].std()),
        "min": round_val(df['rating'].min()),
        "max": round_val(df['rating'].max()),
        "median": round_val(df['rating'].median()),
        "q1": round_val(df['rating'].quantile(0.25)),
        "q3": round_val(df['rating'].quantile(0.75)),
    }
    stats['rating_distribution'] = rating_stats
    
    for key, value in rating_stats.items():
        print_kv(key.capitalize(), value)
    
    # Rating value counts
    rating_counts = df['rating'].value_counts().sort_index()
    print("\n  Rating Value Counts:")
    for rating, count in rating_counts.items():
        pct = round_val(count / n_ratings * 100)
        print(f"    Rating {rating}: {count:>10,} ({pct}%)")
    
    # 3.2 User Activity Statistics
    print_subheader("3.2 User Activity Distribution")
    user_ratings = df.groupby('userId').size()
    user_stats = {
        "mean_ratings_per_user": round_val(user_ratings.mean()),
        "std_ratings_per_user": round_val(user_ratings.std()),
        "min_ratings_per_user": int(user_ratings.min()),
        "max_ratings_per_user": int(user_ratings.max()),
        "median_ratings_per_user": round_val(user_ratings.median()),
    }
    stats['user_activity'] = user_stats
    
    for key, value in user_stats.items():
        print_kv(key.replace("_", " ").title(), value)
    
    # 3.3 Item Popularity Statistics
    print_subheader("3.3 Item Popularity Distribution")
    item_ratings = df.groupby('movieId').size()
    item_stats = {
        "mean_ratings_per_item": round_val(item_ratings.mean()),
        "std_ratings_per_item": round_val(item_ratings.std()),
        "min_ratings_per_item": int(item_ratings.min()),
        "max_ratings_per_item": int(item_ratings.max()),
        "median_ratings_per_item": round_val(item_ratings.median()),
    }
    stats['item_popularity'] = item_stats
    
    for key, value in item_stats.items():
        print_kv(key.replace("_", " ").title(), value)
    
    # 3.4 Sparsity Analysis
    print_subheader("3.4 Matrix Sparsity Analysis")
    total_possible = n_users * n_items
    sparsity = round_val((1 - n_ratings / total_possible) * 100)
    density = round_val(n_ratings / total_possible * 100)
    
    sparsity_stats = {
        "total_possible_entries": total_possible,
        "actual_entries": n_ratings,
        "sparsity_percentage": sparsity,
        "density_percentage": density,
    }
    stats['sparsity'] = sparsity_stats
    
    print_kv("Total possible entries", f"{total_possible:,}")
    print_kv("Actual entries", f"{n_ratings:,}")
    print_kv("Sparsity", f"{sparsity}%")
    print_kv("Density", f"{density}%")
    
    return stats, user_ratings, item_ratings

# ============================================================================
# SECTION 4: TARGET USER SELECTION
# ============================================================================

def select_target_users(df, user_ratings, n_items):
    """
    Select target users based on rating criteria:
    - U1 (Cold user): ≤2% ratings
    - U2 (Medium user): 2% < ratings ≤ 5%
    - U3 (Rich user): >10% ratings
    """
    print_header("SECTION 4: TARGET USER SELECTION")
    
    # Calculate rating percentage for each user
    user_rating_pct = (user_ratings / n_items * 100).round(2)
    
    # Define thresholds
    cold_threshold = 2.0
    medium_low = 2.0
    medium_high = 5.0
    rich_threshold = 10.0
    
    print_subheader("Selection Criteria")
    print_kv("Cold user (U1)", f"≤ {cold_threshold}% ratings")
    print_kv("Medium user (U2)", f"{medium_low}% < ratings ≤ {medium_high}%")
    print_kv("Rich user (U3)", f"> {rich_threshold}% ratings")
    
    # Find candidates for each category
    cold_users = user_rating_pct[user_rating_pct <= cold_threshold]
    medium_users = user_rating_pct[(user_rating_pct > medium_low) & (user_rating_pct <= medium_high)]
    rich_users = user_rating_pct[user_rating_pct > rich_threshold]
    
    print_subheader("Candidate Counts")
    print_kv("Cold user candidates", len(cold_users))
    print_kv("Medium user candidates", len(medium_users))
    print_kv("Rich user candidates", len(rich_users))
    
    # Select one user from each category (pick median rating count within category)
    target_users = {}
    
    if len(cold_users) > 0:
        # Select cold user with median rating count
        cold_sorted = cold_users.sort_values()
        u1_id = cold_sorted.index[len(cold_sorted) // 2]
        target_users['U1'] = {
            'user_id': int(u1_id),
            'num_ratings': int(user_ratings[u1_id]),
            'rating_pct': round_val(user_rating_pct[u1_id]),
            'type': 'Cold'
        }
    
    if len(medium_users) > 0:
        medium_sorted = medium_users.sort_values()
        u2_id = medium_sorted.index[len(medium_sorted) // 2]
        target_users['U2'] = {
            'user_id': int(u2_id),
            'num_ratings': int(user_ratings[u2_id]),
            'rating_pct': round_val(user_rating_pct[u2_id]),
            'type': 'Medium'
        }
    
    if len(rich_users) > 0:
        rich_sorted = rich_users.sort_values()
        u3_id = rich_sorted.index[len(rich_sorted) // 2]
        target_users['U3'] = {
            'user_id': int(u3_id),
            'num_ratings': int(user_ratings[u3_id]),
            'rating_pct': round_val(user_rating_pct[u3_id]),
            'type': 'Rich'
        }
    
    print_subheader("Selected Target Users")
    for user_key, user_info in target_users.items():
        print(f"\n  {user_key} ({user_info['type']} User):")
        print_kv("    User ID", user_info['user_id'])
        print_kv("    Number of ratings", user_info['num_ratings'])
        print_kv("    Rating percentage", f"{user_info['rating_pct']}%")
    
    return target_users

# ============================================================================
# SECTION 5: TARGET ITEM SELECTION
# ============================================================================

def select_target_items(df, item_ratings):
    """
    Select target items based on popularity criteria:
    - I1 (Low popularity): Bottom 25% of ratings count
    - I2 (High popularity): Top 25% of ratings count
    """
    print_header("SECTION 5: TARGET ITEM SELECTION")
    
    # Calculate popularity percentiles
    item_percentiles = item_ratings.rank(pct=True) * 100
    
    # Define thresholds
    low_threshold = 25
    high_threshold = 75
    
    print_subheader("Selection Criteria")
    print_kv("Low popularity (I1)", f"Bottom {low_threshold}%")
    print_kv("High popularity (I2)", f"Top {100-high_threshold}%")
    
    # Find candidates
    low_pop_items = item_ratings[item_percentiles <= low_threshold]
    high_pop_items = item_ratings[item_percentiles >= high_threshold]
    
    print_subheader("Candidate Counts")
    print_kv("Low popularity candidates", len(low_pop_items))
    print_kv("High popularity candidates", len(high_pop_items))
    
    target_items = {}
    
    # Select items at median of each category
    if len(low_pop_items) > 0:
        low_sorted = low_pop_items.sort_values()
        i1_id = low_sorted.index[len(low_sorted) // 2]
        target_items['I1'] = {
            'item_id': int(i1_id),
            'num_ratings': int(item_ratings[i1_id]),
            'percentile': round_val(item_percentiles[i1_id]),
            'type': 'Low Popularity'
        }
    
    if len(high_pop_items) > 0:
        high_sorted = high_pop_items.sort_values()
        i2_id = high_sorted.index[len(high_sorted) // 2]
        target_items['I2'] = {
            'item_id': int(i2_id),
            'num_ratings': int(item_ratings[i2_id]),
            'percentile': round_val(item_percentiles[i2_id]),
            'type': 'High Popularity'
        }
    
    print_subheader("Selected Target Items")
    for item_key, item_info in target_items.items():
        print(f"\n  {item_key} ({item_info['type']}):")
        print_kv("    Item ID", item_info['item_id'])
        print_kv("    Number of ratings", item_info['num_ratings'])
        print_kv("    Popularity percentile", f"{item_info['percentile']}%")
    
    return target_items

# ============================================================================
# SECTION 6: VISUALIZATION
# ============================================================================

def generate_plots(df, user_ratings, item_ratings, stats):
    """Generate visualization plots."""
    print_header("SECTION 6: GENERATING VISUALIZATIONS")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 6.1 Rating Distribution
    print_subheader("6.1 Rating Distribution Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    rating_counts = df['rating'].value_counts().sort_index()
    bars = ax.bar(rating_counts.index, rating_counts.values, color='steelblue', edgecolor='black')
    ax.set_xlabel('Rating', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Rating Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(rating_counts.index)
    for bar, count in zip(bars, rating_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                f'{count:,}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/Sec1_rating_distribution.png", dpi=150)
    plt.close()
    print(f"  [SAVED] {PLOTS_DIR}/Sec1_rating_distribution.png")
    
    # 6.2 User Activity Distribution
    print_subheader("6.2 User Activity Distribution Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(user_ratings, bins=50, color='forestgreen', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Ratings per User', fontsize=12)
    ax.set_ylabel('Number of Users', fontsize=12)
    ax.set_title('User Activity Distribution', fontsize=14, fontweight='bold')
    ax.axvline(user_ratings.mean(), color='red', linestyle='--', label=f'Mean: {user_ratings.mean():.1f}')
    ax.axvline(user_ratings.median(), color='orange', linestyle='--', label=f'Median: {user_ratings.median():.1f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/Sec1_user_activity_distribution.png", dpi=150)
    plt.close()
    print(f"  [SAVED] {PLOTS_DIR}/Sec1_user_activity_distribution.png")
    
    # 6.3 Item Popularity Distribution
    print_subheader("6.3 Item Popularity Distribution Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(item_ratings, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Ratings per Item', fontsize=12)
    ax.set_ylabel('Number of Items', fontsize=12)
    ax.set_title('Item Popularity Distribution (Long-Tail)', fontsize=14, fontweight='bold')
    ax.axvline(item_ratings.mean(), color='red', linestyle='--', label=f'Mean: {item_ratings.mean():.1f}')
    ax.axvline(item_ratings.median(), color='orange', linestyle='--', label=f'Median: {item_ratings.median():.1f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/Sec1_item_popularity_distribution.png", dpi=150)
    plt.close()
    print(f"  [SAVED] {PLOTS_DIR}/Sec1_item_popularity_distribution.png")
    
    # 6.4 Sparsity Visualization
    print_subheader("6.4 Sparsity Visualization")
    fig, ax = plt.subplots(figsize=(8, 6))
    sparsity = stats['sparsity']['sparsity_percentage']
    density = stats['sparsity']['density_percentage']
    colors = ['#ff6b6b', '#4ecdc4']
    ax.pie([sparsity, density], labels=['Sparse (Empty)', 'Dense (Filled)'],
           autopct='%1.2f%%', colors=colors, startangle=90, explode=[0.05, 0])
    ax.set_title('User-Item Matrix Sparsity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/Sec1_sparsity_pie.png", dpi=150)
    plt.close()
    print(f"  [SAVED] {PLOTS_DIR}/Sec1_sparsity_pie.png")
    
    print("\n[DONE] All visualizations generated!")

# ============================================================================
# SECTION 7: SAVE PREPROCESSING RESULTS
# ============================================================================

def save_preprocessing_results(df, stats, target_users, target_items):
    """Save all preprocessing results for use in Part 1."""
    print_header("SECTION 7: SAVING PREPROCESSING RESULTS")
    
    # 7.1 Save statistical summary
    print_subheader("7.1 Saving Statistical Summary")
    summary_file = f"{RESULTS_DIR}/Sec1_statistical_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SECTION 1: STATISTICAL ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("RATING DISTRIBUTION:\n")
        for key, value in stats['rating_distribution'].items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nUSER ACTIVITY:\n")
        for key, value in stats['user_activity'].items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nITEM POPULARITY:\n")
        for key, value in stats['item_popularity'].items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nSPARSITY ANALYSIS:\n")
        for key, value in stats['sparsity'].items():
            f.write(f"  {key}: {value}\n")
    print(f"  [SAVED] {summary_file}")
    
    # 7.2 Save target users
    print_subheader("7.2 Saving Target Users")
    users_file = f"{RESULTS_DIR}/Sec1_target_users.pkl"
    with open(users_file, 'wb') as f:
        pickle.dump(target_users, f)
    print(f"  [SAVED] {users_file}")
    
    # 7.3 Save target items
    print_subheader("7.3 Saving Target Items")
    items_file = f"{RESULTS_DIR}/Sec1_target_items.pkl"
    with open(items_file, 'wb') as f:
        pickle.dump(target_items, f)
    print(f"  [SAVED] {items_file}")
    
    # 7.4 Save processed dataframe
    print_subheader("7.4 Saving Processed Dataset")
    df_file = f"{RESULTS_DIR}/Sec1_ratings_processed.pkl"
    df.to_pickle(df_file)
    print(f"  [SAVED] {df_file}")
    
    # 7.5 Save statistics dictionary
    print_subheader("7.5 Saving Statistics Dictionary")
    stats_file = f"{RESULTS_DIR}/Sec1_statistics.pkl"
    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f)
    print(f"  [SAVED] {stats_file}")
    
    # 7.6 Save summary table as CSV
    print_subheader("7.6 Saving Summary Tables")
    
    # Target users table
    users_df = pd.DataFrame(target_users).T
    users_df.to_csv(f"{TABLES_DIR}/Sec1_target_users.csv")
    print(f"  [SAVED] {TABLES_DIR}/Sec1_target_users.csv")
    
    # Target items table
    items_df = pd.DataFrame(target_items).T
    items_df.to_csv(f"{TABLES_DIR}/Sec1_target_items.csv")
    print(f"  [SAVED] {TABLES_DIR}/Sec1_target_items.csv")
    
    print("\n[DONE] All preprocessing results saved!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print(" SECTION 1: DIMENSIONALITY REDUCTION - DATA PREPROCESSING")
    print(" AIE425 Final Project")
    print("=" * 70)
    
    # Step 1: Load data
    df = load_movielens_data(DATA_DIR)
    if df is None:
        return
    
    # Step 2: Validate dataset
    is_valid, n_users, n_items, n_ratings = validate_dataset(df)
    
    # Step 3: Statistical analysis
    stats, user_ratings, item_ratings = compute_statistical_analysis(
        df, n_users, n_items, n_ratings
    )
    
    # Step 4: Select target users
    target_users = select_target_users(df, user_ratings, n_items)
    
    # Step 5: Select target items
    target_items = select_target_items(df, item_ratings)
    
    # Step 6: Generate visualizations
    generate_plots(df, user_ratings, item_ratings, stats)
    
    # Step 7: Save all results
    save_preprocessing_results(df, stats, target_users, target_items)
    
    # Final summary
    print_header("PREPROCESSING COMPLETE")
    print("\nSummary:")
    print_kv("Total users", f"{n_users:,}")
    print_kv("Total items", f"{n_items:,}")
    print_kv("Total ratings", f"{n_ratings:,}")
    print_kv("Target users selected", len(target_users))
    print_kv("Target items selected", len(target_items))
    print("\n" + "=" * 70)
    print(" All preprocessing results saved to 'results/' directory")
    print(" Visualizations saved to 'plots/' directory")
    print(" Tables saved to 'tables/' directory")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
