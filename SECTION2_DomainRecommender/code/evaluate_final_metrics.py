
"""
Assignment Part 10-12: Final Evaluation Metrics
================================================
Generates specific comparison tables for the report.
1. Baseline Comparison (Random vs Popularity vs Content-Based vs Collaborative vs Hybrid)
2. Cold-Start Analysis (Hit Rate by Rating Count segments)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import random

# Import functions from other modules
import sys
sys.path.append(str(Path(__file__).parent))
from hybrid import hybrid_recommend, load_models, get_global_popularity_scores

# Directories
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

RATINGS_FILE = DATA_DIR / "final_ratings.csv"
ITEMS_FILE = DATA_DIR / "final_items.csv"

TOP_N = 10
N_TEST_USERS = 500  # Evaluate on 500 users for robustness

def get_random_recommendations(all_items, k=TOP_N):
    return random.sample(all_items, k)

def get_popularity_recommendations(pop_scores, k=TOP_N):
    # Already sorted dict
    return list(pop_scores.keys())[:k]

def evaluate_segment(df_ratings, df_items, models, pop_scores, user_segment, name="All Users"):
    """Evaluate all methods on a specific segment of users."""
    print(f"\nEvaluating segment: {name} ({len(user_segment)} users)")
    
    all_items = df_items['streamer_username'].tolist()
    pop_recs = get_popularity_recommendations(pop_scores)
    
    hits = {
        'Random': 0,
        'Popularity': 0,
        'Hybrid': 0
    }
    total = 0
    
    for user_id in user_segment:
        user_ratings = df_ratings[df_ratings['user_id'] == user_id]
        if len(user_ratings) < 1:
            continue
            
        # Leave-one-out (highest rated)
        highest = user_ratings.loc[user_ratings['rating'].idxmax()]
        target = highest['streamer_username']
        target_idx = highest.name
        
        # 1. Random
        rand_recs = get_random_recommendations(all_items)
        if target in rand_recs:
            hits['Random'] += 1
            
        # 2. Popularity
        if target in pop_recs:
            hits['Popularity'] += 1
            
        # 3. Hybrid
        # CRITICAL: Temporarily hide this rating so it's not filtered out!
        original_uid = df_ratings.at[target_idx, 'user_id']
        df_ratings.at[target_idx, 'user_id'] = -999
        
        try:
            hyb_recs = hybrid_recommend(user_id, models, df_ratings, df_items, n_recommendations=TOP_N)
        finally:
            # Always restore
            df_ratings.at[target_idx, 'user_id'] = original_uid
            
        hyb_items = [r['streamer'] for r in hyb_recs]
        if target in hyb_items:
            hits['Hybrid'] += 1
            
        total += 1
        
    return {k: v/total if total > 0 else 0 for k, v in hits.items()}

def main():
    print("="*60)
    print("FINAL EVALUATION METRICS (PART 10-12)")
    print("="*60)
    
    # Load Data
    df_ratings = pd.read_csv(RATINGS_FILE)
    df_items = pd.read_csv(ITEMS_FILE)
    all_items = df_items['streamer_username'].tolist()
    
    # Load Models
    models = load_models()
    if models['content_based'] is None:
        print("[ERROR] Models not loaded.")
        return

    # Pre-compute popularity ranking
    pop_scores = get_global_popularity_scores(df_ratings, all_items)
    # Sort
    pop_scores = dict(sorted(pop_scores.items(), key=lambda x: x[1], reverse=True))

    # Define Segments
    user_counts = df_ratings.groupby('user_id').size()
    
    seg_cold = user_counts[user_counts <= 3].index.tolist()
    seg_med = user_counts[(user_counts > 3) & (user_counts <= 10)].index.tolist()
    seg_est = user_counts[user_counts > 10].index.tolist()
    
    # Sample if too large
    np.random.seed(42)
    random.seed(42)
    
    def sample(u_list, n):
        if len(u_list) > n:
            return np.random.choice(u_list, n, replace=False)
        return u_list

    seg_cold = sample(seg_cold, 200)
    seg_med = sample(seg_med, 200)
    seg_est = sample(seg_est, 200)
    
    # Run Evaluations
    results_cold = evaluate_segment(df_ratings, df_items, models, pop_scores, seg_cold, "Cold Start (<=3 ratings)")
    results_med = evaluate_segment(df_ratings, df_items, models, pop_scores, seg_med, "Medium (4-10 ratings)")
    results_est = evaluate_segment(df_ratings, df_items, models, pop_scores, seg_est, "Established (>10 ratings)")
    
    # Format Output
    output = []
    output.append("PART 10-12: FINAL EVALUATION METRICS\n")
    output.append("="*50 + "\n\n")
    
    output.append(f"{'Segment':<30} {'Random':<10} {'Popularity':<15} {'Hybrid':<10}")
    output.append("-" * 65)
    
    def add_row(name, res):
        output.append(f"{name:<30} {res['Random']:<10.2%} {res['Popularity']:<15.2%} {res['Hybrid']:<10.2%}")
        
    add_row("Cold Start (<=3 ratings)", results_cold)
    add_row("Medium (4-10 ratings)", results_med)
    add_row("Established (>10 ratings)", results_est)
    
    output.append("\n\nAnalysis:\n")
    output.append("- Cold Start: Hybrid should match Popularity or CB performance.")
    output.append("- Established: Hybrid should significantly outperform Random and Popularity.")
    
    # Save
    out_file = RESULTS_DIR / "Part10_Evaluation_Metrics.txt"
    with open(out_file, 'w') as f:
        f.writelines([line + "\n" for line in output])
        
    print(f"\n[SAVED] {out_file}")
    # Print to console
    print("\n" + "".join([line + "\n" for line in output]))

if __name__ == "__main__":
    main()
