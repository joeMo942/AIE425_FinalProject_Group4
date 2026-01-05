"""
Hybrid Recommendation System
=============================
Team Members:
- [Add your names and IDs here]

Implements a weighted hybrid combining:
- Content-Based Filtering (text + numerical features)
- Collaborative Filtering (user-based)

Handles cold-start by dynamically adjusting weights.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from scipy.sparse import load_npz
import random

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

RATINGS_FILE = DATA_DIR / "final_ratings.csv"
ITEMS_FILE = DATA_DIR / "final_items.csv"

# Hybrid parameters
ALPHA = 0.3  # Weight for content-based (lower = more weight for SVD/CF)
TOP_N = 10



# ============================================================================
# PART 1: MODEL & DATA LOADING
# ============================================================================

def load_models():
    """Load pre-trained content-based and collaborative models."""
    print("=" * 60)
    print("LOADING PRE-TRAINED MODELS")
    print("=" * 60)
    
    # Load content-based model
    cb_path = RESULTS_DIR / "content_based_model.pkl"
    if cb_path.exists():
        with open(cb_path, 'rb') as f:
            cb_model = pickle.load(f)
        print(f"[LOADED] Content-Based model")
    else:
        print(f"[ERROR] Content-Based model not found at {cb_path}")
        cb_model = None
    
    # Load collaborative model
    cf_path = RESULTS_DIR / "collaborative_model.pkl"
    if cf_path.exists():
        with open(cf_path, 'rb') as f:
            cf_model = pickle.load(f)
        print(f"[LOADED] Collaborative model")
    else:
        print(f"[ERROR] Collaborative model not found at {cf_path}")
        cf_model = None
    
    # Load user similarity matrix
    sim_path = RESULTS_DIR / "user_similarity.npy"
    if sim_path.exists():
        user_similarity = np.load(sim_path)
        print(f"[LOADED] User similarity matrix: {user_similarity.shape}")
    else:
        user_similarity = None
    
    # Load user-item matrix
    matrix_path = RESULTS_DIR / "user_item_matrix.npz"
    if matrix_path.exists():
        user_item_matrix = load_npz(matrix_path)
        print(f"[LOADED] User-item matrix: {user_item_matrix.shape}")
    else:
        user_item_matrix = None
    
    return {
        'content_based': cb_model,
        'collaborative': cf_model,
        'user_similarity': user_similarity,
        'user_item_matrix': user_item_matrix
    }


def load_data():
    """Load ratings and items data."""
    df_ratings = pd.read_csv(RATINGS_FILE)
    df_items = pd.read_csv(ITEMS_FILE)
    return df_ratings, df_items



# ============================================================================
# PART 2: SCORING FUNCTIONS
# ============================================================================

def get_content_based_scores(user_id, models, df_ratings, all_items):
    """
    Get content-based recommendation scores for a user.
    Returns dict: {item: score}
    """
    cb_model = models['content_based']
    if cb_model is None:
        return {}
    
    user_profiles = cb_model.get('user_profiles', {})
    item_features = cb_model.get('item_features')
    item_to_idx = cb_model.get('item_to_idx', {})
    idx_to_item = cb_model.get('idx_to_item', {})
    
    if user_id not in user_profiles:
        return {}
    
    user_profile = user_profiles[user_id]
    
    # Import cosine_similarity here to avoid circular imports
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Compute similarity between user profile and all items
    user_profile_2d = user_profile.reshape(1, -1)
    similarities = cosine_similarity(user_profile_2d, item_features)[0]
    
    # Get items user has already rated
    rated_items = set(df_ratings[df_ratings['user_id'] == user_id]['streamer_username'])
    
    # Create score dict for unrated items
    scores = {}
    for idx, score in enumerate(similarities):
        item = idx_to_item.get(idx)
        if item and item not in rated_items:
            scores[item] = score
    
    return scores


def get_svd_scores(user_id, models, df_ratings, all_items):
    """
    Get SVD-based recommendation scores for a user.
    Uses pre-computed SVD predictions matrix.
    Returns dict: {item: predicted_rating normalized to 0-1}
    """
    # Load SVD predictions if not already in models
    svd_pred_path = RESULTS_DIR / "svd_predictions.npy"
    if not svd_pred_path.exists():
        return {}
    
    cf_model = models.get('collaborative')
    if cf_model is None:
        return {}
    
    user_to_idx = cf_model.get('user_to_idx', {})
    idx_to_item = cf_model.get('idx_to_item', {})
    
    if user_id not in user_to_idx:
        return {}
    
    user_idx = user_to_idx[user_id]
    
    # Load SVD predictions
    svd_predictions = np.load(svd_pred_path)
    
    # Get items user has already rated
    rated_items = set(df_ratings[df_ratings['user_id'] == user_id]['streamer_username'])
    
    scores = {}
    for item_idx, pred in enumerate(svd_predictions[user_idx, :]):
        item = idx_to_item.get(item_idx)
        if item and item not in rated_items:
            # Normalize to 0-1 scale
            scores[item] = (pred - 1) / 4
    
    return scores


def get_global_popularity_scores(df_ratings, all_items):
    """
    Get global popularity scores for cold-start users.
    Based on how frequently each item is rated (popularity).
    Returns dict: {item: score normalized to 0-1}
    """
    item_counts = df_ratings.groupby('streamer_username').size()
    max_count = item_counts.max()
    
    scores = {}
    for item in all_items:
        count = item_counts.get(item, 0)
        scores[item] = count / max_count  # Normalize to 0-1
    
    return scores


def get_collaborative_scores(user_id, models, df_ratings, all_items):
    """
    Get collaborative filtering recommendation scores for a user (K-NN fallback).
    Returns dict: {item: predicted_rating normalized to 0-1}
    """
    cf_model = models.get('collaborative')
    user_similarity = models.get('user_similarity')
    user_item_matrix = models.get('user_item_matrix')
    
    if cf_model is None or user_similarity is None or user_item_matrix is None:
        return {}
    
    user_to_idx = cf_model.get('user_to_idx', {})
    idx_to_item = cf_model.get('idx_to_item', {})
    user_means = cf_model.get('user_means')
    
    if user_id not in user_to_idx:
        return {}
    
    user_idx = user_to_idx[user_id]
    
    # Get items user has already rated
    user_ratings = user_item_matrix[user_idx].toarray().flatten()
    unrated_indices = np.where(user_ratings == 0)[0]
    
    scores = {}
    
    for item_idx in unrated_indices:
        item_ratings = user_item_matrix[:, item_idx].toarray().flatten()
        users_who_rated = np.where(item_ratings > 0)[0]
        
        if len(users_who_rated) == 0:
            continue
        
        similarities = user_similarity[user_idx, users_who_rated]
        valid_mask = similarities > 0
        if not np.any(valid_mask):
            continue
        
        valid_users = users_who_rated[valid_mask]
        valid_sims = similarities[valid_mask]
        neighbor_ratings = item_ratings[valid_users]
        neighbor_means = user_means[valid_users]
        
        numerator = np.sum(valid_sims * (neighbor_ratings - neighbor_means))
        denominator = np.sum(np.abs(valid_sims))
        
        if denominator > 0:
            prediction = user_means[user_idx] + (numerator / denominator)
            prediction = max(1.0, min(5.0, prediction))
            
            item = idx_to_item.get(item_idx)
            if item:
                scores[item] = (prediction - 1) / 4
    
    return scores


def get_user_rating_count(user_id, df_ratings):
    """Get number of ratings for a user."""
    return len(df_ratings[df_ratings['user_id'] == user_id])



# ============================================================================
# PART 3: HYBRID LOGIC (CASCADE)
# ============================================================================

def hybrid_recommend(user_id, models, df_ratings, df_items, 
                    n_recommendations=TOP_N, n_candidates=50):
    """
    Generate hybrid recommendations using CASCADE HYBRID STRATEGY (Option C).
    
    Cascade Strategy:
    1. Content-Based filters ALL items to top-50 candidates (fast, feature-based)
    2. SVD/CF ranks the 50 candidates to produce final top-10 (accurate, user-preference)
    
    This approach is efficient because:
    - CB runs on all items (fast due to pre-computed user profiles)
    - CF/SVD only needs to score 50 candidates instead of all items
    
    Cold-start handling:
    - 0 ratings: Use global popularity
    - <5 ratings: Use content-based only (not enough data for CF)
    - ≥5 ratings: CASCADE (CB → SVD)
    """
    all_items = df_items['streamer_username'].tolist()
    
    # Get user's rating count
    rating_count = get_user_rating_count(user_id, df_ratings)
    
    # === CASCADE HYBRID STRATEGY ===
    
    if rating_count == 0:
        # COLD-START: Use global popularity
        scores = get_global_popularity_scores(df_ratings, all_items)
        method = 'popularity'
        
        # Build and return recommendations
        hybrid_scores = [{'streamer': item, 'hybrid_score': score, 'method': method}
                        for item, score in scores.items()]
        hybrid_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return hybrid_scores[:n_recommendations]
        
    elif rating_count < 5:
        # LOW ACTIVITY: Use content-based only
        scores = get_content_based_scores(user_id, models, df_ratings, all_items)
        method = 'content_based'
        
        if not scores:
            scores = get_global_popularity_scores(df_ratings, all_items)
            method = 'popularity_fallback'
        
        hybrid_scores = [{'streamer': item, 'hybrid_score': score, 'method': method}
                        for item, score in scores.items()]
        hybrid_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return hybrid_scores[:n_recommendations]
        
    else:
        # ESTABLISHED USER: CASCADE HYBRID
        # Step 1: Content-Based filters to top-N candidates
        cb_scores = get_content_based_scores(user_id, models, df_ratings, all_items)
        
        if not cb_scores:
            # Fallback if CB not available
            scores = get_svd_scores(user_id, models, df_ratings, all_items)
            method = 'svd_only'
            hybrid_scores = [{'streamer': item, 'hybrid_score': score, 'method': method}
                            for item, score in scores.items()]
            hybrid_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
            return hybrid_scores[:n_recommendations]
        
        # Get top-50 candidates from Content-Based
        cb_ranked = sorted(cb_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [item for item, score in cb_ranked[:n_candidates]]
        
        # Step 2: SVD/CF ranks the candidates for final selection
        svd_scores = get_svd_scores(user_id, models, df_ratings, all_items)
        
        if not svd_scores:
            # Fallback to CF if SVD not available
            svd_scores = get_collaborative_scores(user_id, models, df_ratings, all_items)
        
        # Score only the candidates using SVD
        final_scores = []
        for item in top_candidates:
            svd_score = svd_scores.get(item, 0)
            cb_score = cb_scores.get(item, 0)
            
            final_scores.append({
                'streamer': item,
                'hybrid_score': svd_score,  # Use SVD score for final ranking
                'cb_score': cb_score,
                'svd_score': svd_score,
                'method': 'cascade_cb_svd'
            })
        
        # Sort by SVD score (the refinement step)
        final_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return final_scores[:n_recommendations]



# ============================================================================
# PART 4: EVALUATION
# ============================================================================

def evaluate_hybrid(df_ratings, df_items, models, n_users=100):
    """Evaluate hybrid recommendations and compare with individual systems."""
    print("\n" + "=" * 60)
    print("EVALUATING HYBRID SYSTEM")
    print("=" * 60)
    
    # Sample users
    user_rating_counts = df_ratings.groupby('user_id').size()
    eligible_users = user_rating_counts[user_rating_counts >= 3].index.tolist()
    
    if len(eligible_users) > n_users:
        np.random.seed(42)
        sample_users = np.random.choice(eligible_users, n_users, replace=False)
    else:
        sample_users = eligible_users
    
    print(f"[EVAL] Testing on {len(sample_users)} users")
    
    # Track hits for each method
    hits = {'hybrid': 0, 'content_based': 0, 'collaborative': 0}
    total = 0
    
    for user_id in sample_users:
        user_ratings = df_ratings[df_ratings['user_id'] == user_id]
        
        if len(user_ratings) < 2:
            continue
        
        # Get user's highest rated item as target
        highest_rated = user_ratings.loc[user_ratings['rating'].idxmax()]
        target_item = highest_rated['streamer_username']
        
        # Get hybrid recommendations
        hybrid_recs = hybrid_recommend(user_id, models, df_ratings, df_items)
        hybrid_items = [r['streamer'] for r in hybrid_recs]
        
        if target_item in hybrid_items:
            hits['hybrid'] += 1
        
        # Get CB recommendations (for comparison)
        cb_scores = get_content_based_scores(user_id, models, df_ratings, 
                                              df_items['streamer_username'].tolist())
        cb_items = sorted(cb_scores.keys(), key=lambda x: cb_scores[x], reverse=True)[:TOP_N]
        if target_item in cb_items:
            hits['content_based'] += 1
        
        # Get CF recommendations (for comparison)
        cf_scores = get_collaborative_scores(user_id, models, df_ratings,
                                              df_items['streamer_username'].tolist())
        cf_items = sorted(cf_scores.keys(), key=lambda x: cf_scores[x], reverse=True)[:TOP_N]
        if target_item in cf_items:
            hits['collaborative'] += 1
        
        total += 1
    
    print(f"\n       Comparison of Hit Rate @ {TOP_N}:")
    print(f"       " + "-" * 40)
    
    for method, hit_count in hits.items():
        hit_rate = hit_count / total if total > 0 else 0
        bar = "█" * int(hit_rate * 50)
        print(f"       {method:20s}: {hit_rate:6.2%} {bar}")
    
    return {method: hits[method] / total if total > 0 else 0 for method in hits}


def generate_sample_recommendations(df_ratings, df_items, models, n_samples=5):
    """Generate and display sample hybrid recommendations."""
    print("\n" + "=" * 60)
    print("SAMPLE HYBRID RECOMMENDATIONS")
    print("=" * 60)
    
    # Get users with enough ratings
    user_rating_counts = df_ratings.groupby('user_id').size()
    eligible_users = user_rating_counts[user_rating_counts >= 3].index.tolist()
    
    np.random.seed(42)
    sample_users = np.random.choice(eligible_users, 
                                     min(n_samples, len(eligible_users)), 
                                     replace=False)
    
    for user_id in sample_users:
        user_ratings = df_ratings[df_ratings['user_id'] == user_id]
        
        print(f"\n--- User {user_id} ---")
        print("Watched streamers:")
        for _, row in user_ratings.head(5).iterrows():
            print(f"  ★{'★' * (int(row['rating'])-1)} {row['streamer_username']}")
        
        if len(user_ratings) > 5:
            print(f"  ... and {len(user_ratings) - 5} more")
        
        recommendations = hybrid_recommend(user_id, models, df_ratings, df_items)
        
        print(f"\nTop {TOP_N} Cascade Hybrid Recommendations (CB→SVD):")
        print(f"  {'Rank':>4} {'Streamer':25} {'CB':>7} {'SVD':>7} {'Method':>15}")
        print("  " + "-" * 65)
        
        for i, rec in enumerate(recommendations, 1):
            cb = rec.get('cb_score', rec['hybrid_score'])
            svd = rec.get('svd_score', rec['hybrid_score'])
            print(f"  {i:>4}. {rec['streamer']:25} "
                  f"{cb:>6.3f} {svd:>6.3f} "
                  f"{rec['method']:>15}")



# ============================================================================
# PART 10-12: FINAL EVALUATION METRICS
# ============================================================================

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
        # We need to ensure we don't modify the dataframe in place permanently or race condition
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

def run_final_evaluation(df_ratings, df_items, models):
    """Run the comprehensive evaluation for the report (Part 10-12)."""
    print("\n" + "="*60)
    print("FINAL EVALUATION METRICS (PART 10-12)")
    print("="*60)
    
    all_items = df_items['streamer_username'].tolist()

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


def save_recommendations(df_ratings, df_items, models, n_users=100):
    """Save hybrid recommendations to CSV."""
    print("\n" + "=" * 60)
    print("SAVING RECOMMENDATIONS")
    print("=" * 60)
    
    # Sample users
    user_rating_counts = df_ratings.groupby('user_id').size()
    eligible_users = user_rating_counts[user_rating_counts >= 3].index.tolist()
    
    np.random.seed(42)
    sample_users = eligible_users[:n_users]
    
    all_recommendations = []
    
    for user_id in sample_users:
        recommendations = hybrid_recommend(user_id, models, df_ratings, df_items)
        
        for rank, rec in enumerate(recommendations, 1):
            all_recommendations.append({
                'user_id': user_id,
                'rank': rank,
                'streamer_username': rec['streamer'],
                'cb_score': rec.get('cb_score', rec['hybrid_score']),
                'svd_score': rec.get('svd_score', rec['hybrid_score']),
                'method': rec['method']
            })
    
    df_recs = pd.DataFrame(all_recommendations)
    output_path = RESULTS_DIR / "hybrid_recommendations.csv"
    df_recs.to_csv(output_path, index=False)
    
    print(f"[SAVED] {output_path.name} ({len(df_recs):,} recommendations)")



# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main hybrid recommendation pipeline."""
    print("\n" + "=" * 60)
    print("HYBRID RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    # Load pre-trained models
    models = load_models()
    
    # Check if both models are available
    if models['content_based'] is None or models['collaborative'] is None:
        print("\n[ERROR] Both models must be trained first!")
        print("        Run content_based.py and collaborative.py first.")
        return
    
    # Load data
    df_ratings, df_items = load_data()
    
    print(f"\n[DATA] Ratings: {len(df_ratings):,}")
    print(f"       Items: {len(df_items):,}")
    
    # Evaluate hybrid system
    metrics = evaluate_hybrid(df_ratings, df_items, models)
    
    # Run Final Report Evaluation (Part 10-12)
    run_final_evaluation(df_ratings, df_items, models)
    
    # Generate sample recommendations
    generate_sample_recommendations(df_ratings, df_items, models)
    
    # Save recommendations
    save_recommendations(df_ratings, df_items, models)
    
    print("\n" + "=" * 60)
    print("[DONE] Hybrid recommendation complete!")
    print("=" * 60 + "\n")
    
    return metrics


if __name__ == "__main__":
    main()
