"""
Collaborative Filtering Recommendation System
=============================================
Team Members:
- [Add your names and IDs here]

Implements collaborative filtering using:
1. Item-Item Collaborative Filtering (Cosine Similarity)
2. Matrix Factorization (SVD)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import pickle

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RATINGS_FILE = DATA_DIR / "final_ratings.csv"
ITEMS_FILE = DATA_DIR / "final_items.csv"

TOP_N = 10


def load_data():
    """Load ratings data."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    df_ratings = pd.read_csv(RATINGS_FILE)
    df_items = pd.read_csv(ITEMS_FILE)
    print(f"[LOADED] Ratings: {len(df_ratings):,}")
    print(f"[LOADED] Items: {len(df_items):,}")
    
    return df_ratings, df_items


def create_interaction_matrix(df_ratings):
    """Create User-Item interaction matrix."""
    print("\n" + "=" * 60)
    print("CREATING INTERACTION MATRIX")
    print("=" * 60)
    
    # Map IDs to indices
    user_ids = df_ratings['user_id'].unique()
    item_ids = df_ratings['streamer_username'].unique()
    
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_to_idx = {iid: i for i, iid in enumerate(item_ids)}
    idx_to_user = {i: uid for uid, i in user_to_idx.items()}
    idx_to_item = {i: iid for iid, i in item_to_idx.items()}
    
    rows = df_ratings['user_id'].map(user_to_idx)
    cols = df_ratings['streamer_username'].map(item_to_idx)
    data = df_ratings['rating']
    
    # Create sparse matrix
    interaction_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    
    print(f"[DONE] Matrix shape: {interaction_matrix.shape}")
    print(f"       Users: {interaction_matrix.shape[0]:,}")
    print(f"       Items: {interaction_matrix.shape[1]:,}")
    print(f"       Sparsity: {1.0 - interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1]):.4%}")
    
    return interaction_matrix, user_to_idx, item_to_idx, idx_to_item, idx_to_user


def train_item_item_cf(interaction_matrix):
    """
    Train Item-Item Collaborative Filtering model.
    Computes cosine similarity between item vectors (columns of interaction matrix).
    """
    print("\n" + "=" * 60)
    print("TRAINING ITEM-ITEM CF")
    print("=" * 60)
    
    # Transpose to get Items x Users matrix
    item_user_matrix = interaction_matrix.T
    
    # Compute similarity (Items x Items)
    print("Computing cosine similarity matrix...")
    item_similarity = cosine_similarity(item_user_matrix)
    
    print(f"[DONE] Item Similarity Matrix: {item_similarity.shape}")
    
    # Zero out diagonal (item is always similar to itself)
    np.fill_diagonal(item_similarity, 0)
    
    return item_similarity


def train_svd_cf(interaction_matrix, n_components=50):
    """
    Train Matrix Factorization (SVD) model.
    Decomposes R ~ U * Sigma * Vt
    """
    print("\n" + "=" * 60)
    print("TRAINING SVD (MATRIX FACTORIZATION)")
    print("=" * 60)
    
    print(f"Decomposing matrix with {n_components} components...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(interaction_matrix)
    item_factors = svd.components_
    
    print(f"[DONE] Explained Variance: {svd.explained_variance_ratio_.sum():.4f}")
    
    return svd, user_factors, item_factors


def recommend_item_item(user_idx, interaction_matrix, item_similarity, idx_to_item, n_recs=TOP_N):
    """Generate recommendations using Item-Item CF."""
    # Get user's ratings (1 x Items)
    user_ratings = interaction_matrix[user_idx].toarray().flatten()
    
    # Weighted sum of item similarities: score = sum(rating * similarity) / sum(similarity)
    # Scores (1 x Items) = UserRatings (1 x Items) * SimilarityMatrix (Items x Items)
    scores = user_ratings.dot(item_similarity)
    
    # Normalize by sum of similarities (optional, but good for rating prediction)
    # For ranking, raw dot product is often sufficient
    
    # Filter out already rated items
    rated_indices = np.where(user_ratings > 0)[0]
    scores[rated_indices] = -1  # Mask already rated
    
    # Get top N indices
    top_indices = scores.argsort()[::-1][:n_recs]
    
    recs = []
    for idx in top_indices:
        if scores[idx] > 0:
            recs.append((idx_to_item[idx], scores[idx]))
            
    return recs


def recommend_svd(user_idx, user_factors, item_factors, interaction_matrix, idx_to_item, n_recs=TOP_N):
    """Generate recommendations using SVD."""
    # Predict all ratings: UserFactor (1 x K) * ItemFactors (K x Items)
    user_vector = user_factors[user_idx].reshape(1, -1)
    scores = user_vector.dot(item_factors).flatten()
    
    # Filter out already rated
    user_ratings = interaction_matrix[user_idx].toarray().flatten()
    rated_indices = np.where(user_ratings > 0)[0]
    scores[rated_indices] = -1
    
    # Get top N indices
    top_indices = scores.argsort()[::-1][:n_recs]
    
    recs = []
    for idx in top_indices:
        if scores[idx] > -1:
            recs.append((idx_to_item[idx], scores[idx]))
            
    return recs


def evaluate_models(interaction_matrix, item_similarity, svd_model, n_test_users=500):
    """Evaluate HIT RATE for both models."""
    print("\n" + "=" * 60)
    print("EVALUATING MODELS (Hit Rate @ 10)")
    print("=" * 60)
    
    # Sample users with enough history
    user_indices = np.where(np.diff(interaction_matrix.indptr) >= 5)[0]  # Users with >= 5 ratings
    if len(user_indices) > n_test_users:
        np.random.seed(42)
        test_users = np.random.choice(user_indices, n_test_users, replace=False)
    else:
        test_users = user_indices
        
    print(f"Testing on {len(test_users)} users...")
    
    hits_item = 0
    hits_svd = 0
    
    # For SVD
    user_factors = svd_model.transform(interaction_matrix)
    item_factors = svd_model.components_
    
    for user_idx in test_users:
        # Get ground truth: hide one high-rated item
        ratings = interaction_matrix[user_idx].toarray().flatten()
        rated_items = np.where(ratings >= 4)[0] # Only consider liked items
        
        if len(rated_items) > 0:
            target_idx = np.random.choice(rated_items)
            
            # Mask the target
            interaction_matrix[user_idx, target_idx] = 0
            
            # Item-Item Recs
            recs_item = recommend_item_item(user_idx, interaction_matrix, item_similarity, {i:i for i in range(len(ratings))}, n_recs=TOP_N)
            rec_indices_item = [r[0] for r in recs_item]
            if target_idx in rec_indices_item:
                hits_item += 1
                
            # SVD Recs
            recs_svd = recommend_svd(user_idx, user_factors, item_factors, interaction_matrix, {i:i for i in range(len(ratings))}, n_recs=TOP_N)
            rec_indices_svd = [r[0] for r in recs_svd]
            if target_idx in rec_indices_svd:
                hits_svd += 1
                
            # Restore mask
            interaction_matrix[user_idx, target_idx] = ratings[target_idx]
    
    print(f"\n[RESULTS]")
    print(f"Item-Item CF Hit Rate: {hits_item / len(test_users):.2%}")
    print(f"SVD (Matrix Factorization) Hit Rate: {hits_svd / len(test_users):.2%}")


def main():
    print("\n" + "=" * 60)
    print("COLLABORATIVE FILTERING SYSTEM")
    print("=" * 60)
    
    # Load and Create Matrix
    df_ratings, df_items = load_data()
    interaction_matrix, user_to_idx, item_to_idx, idx_to_item, idx_to_user = create_interaction_matrix(df_ratings)
    
    # Train Item-Item
    item_similarity = train_item_item_cf(interaction_matrix)
    
    # Train SVD
    svd, user_factors, items_factors = train_svd_cf(interaction_matrix)
    
    # Evaluate
    evaluate_models(interaction_matrix, item_similarity, svd)
    
    # Save Models
    import pickle
    with open(RESULTS_DIR / "collab_models.pkl", 'wb') as f:
        pickle.dump({
            'item_similarity': item_similarity,
            'svd': svd,
            'item_to_idx': item_to_idx,
            'idx_to_item': idx_to_item
        }, f)
        
    print(f"\n[DONE] Models saved to {RESULTS_DIR}/collab_models.pkl")

if __name__ == "__main__":
    main()
