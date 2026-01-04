"""
Collaborative Filtering Recommendation System
==============================================
Team Members:
- [Add your names and IDs here]

Implements User-Based Collaborative Filtering using:
- User-Item rating matrix
- Cosine similarity for user similarity
- Weighted average prediction
- k-Nearest Neighbors for recommendations
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pickle

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RATINGS_FILE = DATA_DIR / "final_ratings.csv"

# Model parameters
TOP_N = 10  # Number of recommendations
K_NEIGHBORS = 50  # Number of similar users to consider


def load_data():
    """Load ratings data."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    df_ratings = pd.read_csv(RATINGS_FILE)
    
    print(f"[LOADED] Ratings: {len(df_ratings):,}")
    print(f"         Users: {df_ratings['user_id'].nunique():,}")
    print(f"         Items: {df_ratings['streamer_username'].nunique():,}")
    
    return df_ratings


def create_user_item_matrix(df_ratings):
    """
    Create User-Item rating matrix.
    Returns sparse matrix and mappings.
    """
    print("\n" + "=" * 60)
    print("CREATING USER-ITEM MATRIX")
    print("=" * 60)
    
    # Create mappings
    users = df_ratings['user_id'].unique()
    items = df_ratings['streamer_username'].unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(users)}
    idx_to_user = {idx: user for idx, user in enumerate(users)}
    item_to_idx = {item: idx for idx, item in enumerate(items)}
    idx_to_item = {idx: item for idx, item in enumerate(items)}
    
    # Create sparse matrix
    n_users = len(users)
    n_items = len(items)
    
    row_indices = df_ratings['user_id'].map(user_to_idx).values
    col_indices = df_ratings['streamer_username'].map(item_to_idx).values
    ratings = df_ratings['rating'].values
    
    user_item_matrix = csr_matrix(
        (ratings, (row_indices, col_indices)),
        shape=(n_users, n_items)
    )
    
    # Calculate sparsity
    sparsity = 1 - (len(df_ratings) / (n_users * n_items))
    
    print(f"[DONE] Matrix shape: {user_item_matrix.shape}")
    print(f"       Non-zero entries: {user_item_matrix.nnz:,}")
    print(f"       Sparsity: {sparsity:.4%}")
    
    return {
        'matrix': user_item_matrix,
        'user_to_idx': user_to_idx,
        'idx_to_user': idx_to_user,
        'item_to_idx': item_to_idx,
        'idx_to_item': idx_to_item,
        'n_users': n_users,
        'n_items': n_items
    }


def compute_user_similarity(matrix_data):
    """
    Compute pairwise cosine similarity between users.
    For large datasets (>20k users), skip full matrix computation
    and use on-demand similarity in prediction.
    """
    print("\n" + "=" * 60)
    print("COMPUTING USER SIMILARITY")
    print("=" * 60)
    
    user_item_matrix = matrix_data['matrix']
    n_users = matrix_data['n_users']
    
    print(f"[INFO] Dataset size: {n_users:,} users")
    
    if n_users <= 20000:
        # Direct computation for smaller datasets
        print("[1] Computing full cosine similarity matrix...")
        similarity_matrix = cosine_similarity(user_item_matrix)
        print(f"\n[DONE] Similarity matrix: {similarity_matrix.shape}")
        
        # Sample similarities
        print("\n       Sample similarities:")
        print(f"       User 0 to User 0: {similarity_matrix[0, 0]:.4f}")
        print(f"       User 0 to User 1: {similarity_matrix[0, 1]:.4f}")
        
        return similarity_matrix
    else:
        # For large datasets, return None and use on-demand computation
        print("[WARNING] Dataset too large for full similarity matrix (>20k users)")
        print("          Using on-demand similarity computation in predictions.")
        print("          This will be slower but avoids memory issues.")
        
        return None


def compute_user_means(matrix_data):
    """Compute mean rating for each user."""
    print("\n" + "-" * 40)
    print("Computing user mean ratings...")
    
    user_item_matrix = matrix_data['matrix']
    
    # For sparse matrix: sum of ratings / count of non-zero entries per user
    row_sums = np.array(user_item_matrix.sum(axis=1)).flatten()
    row_counts = np.array((user_item_matrix != 0).sum(axis=1)).flatten()
    
    # Avoid division by zero
    user_means = np.divide(row_sums, row_counts, 
                           out=np.zeros_like(row_sums, dtype=float), 
                           where=row_counts != 0)
    
    print(f"[DONE] Mean ratings range: {user_means.min():.2f} - {user_means.max():.2f}")
    
    return user_means


def predict_rating(user_idx, item_idx, matrix_data, similarity_matrix, 
                   user_means, k_neighbors=K_NEIGHBORS):
    """
    Predict rating for a user-item pair using weighted average of neighbors.
    
    Formula: r_hat = mean_u + sum(sim(u,v) * (r_vi - mean_v)) / sum(|sim(u,v)|)
    """
    user_item_matrix = matrix_data['matrix']
    
    # Get users who rated this item
    item_ratings = user_item_matrix[:, item_idx].toarray().flatten()
    users_who_rated = np.where(item_ratings > 0)[0]
    
    if len(users_who_rated) == 0:
        return user_means[user_idx]  # No ratings for this item
    
    # Get similarities between target user and users who rated
    if similarity_matrix is not None:
        similarities = similarity_matrix[user_idx, users_who_rated]
    else:
        # On-demand computation for large datasets
        user_vector = user_item_matrix[user_idx]
        neighbor_vectors = user_item_matrix[users_who_rated]
        similarities = cosine_similarity(user_vector, neighbor_vectors)[0]
    
    # Sort by similarity and take top k
    if len(users_who_rated) > k_neighbors:
        top_k_indices = np.argsort(similarities)[::-1][:k_neighbors]
        neighbors = users_who_rated[top_k_indices]
        neighbor_sims = similarities[top_k_indices]
    else:
        neighbors = users_who_rated
        neighbor_sims = similarities
    
    # Filter out zero/negative similarities
    valid_mask = neighbor_sims > 0
    if not np.any(valid_mask):
        return user_means[user_idx]
    
    neighbors = neighbors[valid_mask]
    neighbor_sims = neighbor_sims[valid_mask]
    
    # Compute weighted prediction
    neighbor_ratings = item_ratings[neighbors]
    neighbor_means = user_means[neighbors]
    
    numerator = np.sum(neighbor_sims * (neighbor_ratings - neighbor_means))
    denominator = np.sum(np.abs(neighbor_sims))
    
    if denominator == 0:
        return user_means[user_idx]
    
    prediction = user_means[user_idx] + (numerator / denominator)
    
    # Clip to valid rating range [1, 5]
    prediction = max(1.0, min(5.0, prediction))
    
    return prediction


def recommend_for_user(user_id, matrix_data, similarity_matrix, user_means,
                       df_ratings, n_recommendations=TOP_N):
    """
    Generate recommendations for a specific user.
    Predict ratings for all unrated items and return top-N.
    """
    user_to_idx = matrix_data['user_to_idx']
    idx_to_item = matrix_data['idx_to_item']
    user_item_matrix = matrix_data['matrix']
    
    if user_id not in user_to_idx:
        return []
    
    user_idx = user_to_idx[user_id]
    
    # Get items user has already rated
    user_ratings = user_item_matrix[user_idx].toarray().flatten()
    unrated_items = np.where(user_ratings == 0)[0]
    
    # Predict ratings for unrated items
    predictions = []
    for item_idx in unrated_items:
        pred = predict_rating(user_idx, item_idx, matrix_data, 
                             similarity_matrix, user_means)
        predictions.append({
            'item_idx': item_idx,
            'streamer': idx_to_item[item_idx],
            'predicted_rating': pred
        })
    
    # Sort by predicted rating and return top-N
    predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
    
    return predictions[:n_recommendations]


def evaluate_recommendations(df_ratings, matrix_data, similarity_matrix, 
                            user_means, n_users=100):
    """
    Evaluate recommendation quality using held-out ratings.
    """
    print("\n" + "=" * 60)
    print("EVALUATING RECOMMENDATIONS")
    print("=" * 60)
    
    # Sample users with at least 3 ratings
    user_rating_counts = df_ratings.groupby('user_id').size()
    eligible_users = user_rating_counts[user_rating_counts >= 3].index.tolist()
    
    if len(eligible_users) > n_users:
        np.random.seed(42)
        sample_users = np.random.choice(eligible_users, n_users, replace=False)
    else:
        sample_users = eligible_users
    
    print(f"[EVAL] Testing on {len(sample_users)} users")
    
    hits = 0
    total = 0
    rmse_sum = 0
    rmse_count = 0
    
    for user_id in sample_users:
        user_ratings = df_ratings[df_ratings['user_id'] == user_id]
        
        if len(user_ratings) < 2:
            continue
        
        # Get user's highest rated item as "target"
        highest_rated = user_ratings.loc[user_ratings['rating'].idxmax()]
        target_item = highest_rated['streamer_username']
        actual_rating = highest_rated['rating']
        
        # Get recommendations
        recommendations = recommend_for_user(
            user_id, matrix_data, similarity_matrix, user_means,
            df_ratings, n_recommendations=TOP_N
        )
        
        rec_items = [r['streamer'] for r in recommendations]
        
        # Check if target is in recommendations
        if target_item in rec_items:
            hits += 1
            
            # Get predicted rating for RMSE
            pred_rating = next(r['predicted_rating'] for r in recommendations 
                              if r['streamer'] == target_item)
            rmse_sum += (pred_rating - actual_rating) ** 2
            rmse_count += 1
        
        total += 1
    
    hit_rate = hits / total if total > 0 else 0
    rmse = np.sqrt(rmse_sum / rmse_count) if rmse_count > 0 else 0
    
    print(f"\n       Hit Rate @ {TOP_N}: {hit_rate:.2%} ({hits}/{total})")
    print(f"       RMSE (on hits): {rmse:.4f}")
    
    return {'hit_rate': hit_rate, 'rmse': rmse}


def generate_sample_recommendations(df_ratings, matrix_data, similarity_matrix,
                                     user_means, n_samples=5):
    """Generate and display sample recommendations."""
    print("\n" + "=" * 60)
    print("SAMPLE RECOMMENDATIONS")
    print("=" * 60)
    
    # Get users with enough ratings
    user_rating_counts = df_ratings.groupby('user_id').size()
    eligible_users = user_rating_counts[user_rating_counts >= 3].index.tolist()
    
    np.random.seed(42)
    sample_users = np.random.choice(eligible_users, 
                                     min(n_samples, len(eligible_users)), 
                                     replace=False)
    
    results = []
    
    for user_id in sample_users:
        user_ratings = df_ratings[df_ratings['user_id'] == user_id]
        
        print(f"\n--- User {user_id} ---")
        print("Watched streamers:")
        for _, row in user_ratings.head(5).iterrows():
            print(f"  ★{'★' * (int(row['rating'])-1)} {row['streamer_username']}")
        
        if len(user_ratings) > 5:
            print(f"  ... and {len(user_ratings) - 5} more")
        
        recommendations = recommend_for_user(
            user_id, matrix_data, similarity_matrix, user_means,
            df_ratings, n_recommendations=TOP_N
        )
        
        print(f"\nTop {TOP_N} Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['streamer']} (pred: {rec['predicted_rating']:.2f})")
        
        results.append({
            'user_id': user_id,
            'history': user_ratings['streamer_username'].tolist(),
            'recommendations': [r['streamer'] for r in recommendations]
        })
    
    return results


def save_model(matrix_data, similarity_matrix, user_means):
    """Save trained model for later use."""
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    model_path = RESULTS_DIR / "collaborative_model.pkl"
    
    model = {
        'user_to_idx': matrix_data['user_to_idx'],
        'idx_to_user': matrix_data['idx_to_user'],
        'item_to_idx': matrix_data['item_to_idx'],
        'idx_to_item': matrix_data['idx_to_item'],
        'user_means': user_means,
        'n_users': matrix_data['n_users'],
        'n_items': matrix_data['n_items']
    }
    
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"[SAVED] {model_path.name}")
    
    # Save similarity matrix (separate file due to size) - only if computed
    if similarity_matrix is not None:
        sim_path = RESULTS_DIR / "user_similarity.npy"
        np.save(sim_path, similarity_matrix)
        print(f"[SAVED] {sim_path.name}")
    else:
        print("[SKIP] User similarity matrix (not computed for large dataset)")
    
    # Save user-item matrix
    matrix_path = RESULTS_DIR / "user_item_matrix.npz"
    from scipy.sparse import save_npz
    save_npz(matrix_path, matrix_data['matrix'])
    print(f"[SAVED] {matrix_path.name}")


# =============================================================================
# SVD Matrix Factorization (Assignment Part 3, Section 8.2)
# =============================================================================

def compute_svd(matrix_data, user_means, k=20):
    """
    Compute SVD with k latent factors for matrix factorization.
    
    Steps:
    1. Create normalized matrix: R_normalized = R - user_mean (for each row)
    2. Compute truncated SVD: U, Σ, V^T = svds(R_normalized, k)
    3. Reconstruct predictions: R_pred = U @ diag(Σ) @ V^T + user_mean
    
    This addresses sparsity by learning latent factors that capture 
    user preferences and item characteristics.
    """
    from scipy.sparse.linalg import svds
    from scipy.sparse import lil_matrix
    
    print("\n" + "=" * 60)
    print(f"SVD MATRIX FACTORIZATION (k={k} latent factors)")
    print("=" * 60)
    
    user_item_matrix = matrix_data['matrix']
    n_users = matrix_data['n_users']
    n_items = matrix_data['n_items']
    
    # Step 1: Normalize by subtracting user means (handle sparse matrix)
    print("\n[1] Normalizing matrix by user means...")
    
    # Convert to dense for SVD (required for small/medium datasets)
    R = user_item_matrix.toarray().astype(np.float64)
    
    # Create mask for non-zero entries
    mask = R > 0
    
    # Subtract user mean from non-zero entries only
    for i in range(n_users):
        if user_means[i] > 0:
            R[i, mask[i]] -= user_means[i]
    
    # For missing entries (0), keep them as 0 (mean-centered would be 0)
    print(f"       Normalized matrix shape: {R.shape}")
    
    # Step 2: Compute truncated SVD
    print(f"\n[2] Computing SVD with k={k} factors...")
    
    # Use scipy's svds for sparse-friendly SVD
    # Note: svds returns singular values in ascending order
    U, sigma, Vt = svds(R, k=k)
    
    # Reverse to get descending order (largest first)
    idx = np.argsort(sigma)[::-1]
    U = U[:, idx]
    sigma = sigma[idx]
    Vt = Vt[idx, :]
    
    print(f"       U shape: {U.shape}")
    print(f"       Σ (singular values): {sigma[:5]}...")
    print(f"       V^T shape: {Vt.shape}")
    
    # Step 3: Reconstruct prediction matrix
    print("\n[3] Reconstructing prediction matrix...")
    
    # R_pred = U @ diag(Σ) @ V^T + user_mean
    R_pred = U @ np.diag(sigma) @ Vt
    
    # Add user means back
    for i in range(n_users):
        R_pred[i, :] += user_means[i]
    
    # Clip to valid rating range [1, 5]
    R_pred = np.clip(R_pred, 1.0, 5.0)
    
    print(f"       Prediction matrix shape: {R_pred.shape}")
    print(f"       Prediction range: [{R_pred.min():.2f}, {R_pred.max():.2f}]")
    
    # Calculate explained variance
    total_variance = np.sum(R ** 2)
    explained_variance = np.sum(sigma ** 2) / total_variance * 100
    print(f"\n       Explained variance (k={k}): {explained_variance:.2f}%")
    
    return {
        'U': U,
        'sigma': sigma,
        'Vt': Vt,
        'predictions': R_pred,
        'k': k
    }


def evaluate_svd(matrix_data, svd_data, df_ratings, n_users=100):
    """Evaluate SVD predictions using RMSE on known ratings."""
    print("\n" + "=" * 60)
    print("EVALUATING SVD PREDICTIONS")
    print("=" * 60)
    
    R_pred = svd_data['predictions']
    user_to_idx = matrix_data['user_to_idx']
    item_to_idx = matrix_data['item_to_idx']
    
    # Sample users
    user_rating_counts = df_ratings.groupby('user_id').size()
    eligible_users = user_rating_counts[user_rating_counts >= 3].index.tolist()
    
    np.random.seed(42)
    sample_users = np.random.choice(eligible_users, 
                                     min(n_users, len(eligible_users)), 
                                     replace=False)
    
    # Calculate RMSE on known ratings
    squared_errors = []
    hits = 0
    total = 0
    
    for user_id in sample_users:
        if user_id not in user_to_idx:
            continue
        user_idx = user_to_idx[user_id]
        
        user_ratings = df_ratings[df_ratings['user_id'] == user_id]
        
        for _, row in user_ratings.iterrows():
            item = row['streamer_username']
            if item not in item_to_idx:
                continue
            item_idx = item_to_idx[item]
            
            actual = row['rating']
            predicted = R_pred[user_idx, item_idx]
            
            squared_errors.append((predicted - actual) ** 2)
        
        # Hit rate: check if highest rated item is in top-10 predictions
        if len(user_ratings) >= 2:
            highest_rated = user_ratings.loc[user_ratings['rating'].idxmax()]
            target_item = highest_rated['streamer_username']
            
            if target_item in item_to_idx:
                # Get top-10 predictions for this user
                user_preds = R_pred[user_idx, :]
                # Exclude already rated items
                rated_items = [item_to_idx[i] for i in user_ratings['streamer_username'] 
                               if i in item_to_idx]
                user_preds_copy = user_preds.copy()
                user_preds_copy[rated_items] = -np.inf
                
                top_10_idx = np.argsort(user_preds_copy)[::-1][:10]
                top_10_items = [matrix_data['idx_to_item'][i] for i in top_10_idx]
                
                if target_item in top_10_items:
                    hits += 1
                total += 1
    
    rmse = np.sqrt(np.mean(squared_errors)) if squared_errors else 0
    hit_rate = hits / total if total > 0 else 0
    
    print(f"\n       SVD RMSE: {rmse:.4f}")
    print(f"       SVD Hit Rate @ 10: {hit_rate:.2%} ({hits}/{total})")
    
    return {'rmse': rmse, 'hit_rate': hit_rate}


def save_svd_model(svd_data, matrix_data):
    """Save SVD model and predictions."""
    print("\n" + "=" * 60)
    print("SAVING SVD MODEL")
    print("=" * 60)
    
    # Save predictions matrix
    pred_path = RESULTS_DIR / "svd_predictions.npy"
    np.save(pred_path, svd_data['predictions'])
    print(f"[SAVED] {pred_path.name}")
    
    # Save SVD components
    svd_model_path = RESULTS_DIR / "svd_model.pkl"
    model = {
        'U': svd_data['U'],
        'sigma': svd_data['sigma'],
        'Vt': svd_data['Vt'],
        'k': svd_data['k'],
        'user_to_idx': matrix_data['user_to_idx'],
        'idx_to_item': matrix_data['idx_to_item']
    }
    
    with open(svd_model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[SAVED] {svd_model_path.name}")


def main():
    """Main collaborative filtering pipeline."""
    print("\n" + "=" * 60)
    print("COLLABORATIVE FILTERING RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    # Load data
    df_ratings = load_data()
    
    # Create user-item matrix
    matrix_data = create_user_item_matrix(df_ratings)
    
    # Compute user means
    user_means = compute_user_means(matrix_data)
    
    # Compute user similarity (for K-NN baseline)
    similarity_matrix = compute_user_similarity(matrix_data)
    
    # Evaluate K-NN
    print("\n" + "=" * 60)
    print("K-NN COLLABORATIVE FILTERING EVALUATION")
    print("=" * 60)
    knn_metrics = evaluate_recommendations(
        df_ratings, matrix_data, similarity_matrix, user_means
    )
    
    # =========================================================================
    # SVD Matrix Factorization (Assignment Part 3, Section 8.2)
    # =========================================================================
    svd_data = compute_svd(matrix_data, user_means, k=20)
    svd_metrics = evaluate_svd(matrix_data, svd_data, df_ratings)
    
    # Compare K-NN vs SVD
    print("\n" + "=" * 60)
    print("COMPARISON: K-NN vs SVD")
    print("=" * 60)
    print(f"\n       {'Method':<20} {'Hit Rate @ 10':<15} {'RMSE':<10}")
    print("       " + "-" * 45)
    print(f"       {'K-NN':<20} {knn_metrics['hit_rate']:<15.2%} {knn_metrics.get('rmse', 'N/A')}")
    print(f"       {'SVD (k=20)':<20} {svd_metrics['hit_rate']:<15.2%} {svd_metrics['rmse']:<10.4f}")
    
    # Generate sample recommendations
    results = generate_sample_recommendations(
        df_ratings, matrix_data, similarity_matrix, user_means
    )
    
    # Save models
    save_model(matrix_data, similarity_matrix, user_means)
    save_svd_model(svd_data, matrix_data)
    
    print("\n" + "=" * 60)
    print("[DONE] Collaborative filtering complete!")
    print("=" * 60 + "\n")
    
    return {
        'matrix_data': matrix_data,
        'similarity_matrix': similarity_matrix,
        'user_means': user_means,
        'knn_metrics': knn_metrics,
        'svd_data': svd_data,
        'svd_metrics': svd_metrics
    }


if __name__ == "__main__":
    main()

