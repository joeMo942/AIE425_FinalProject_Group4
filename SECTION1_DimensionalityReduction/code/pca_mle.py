# PCA with Maximum Likelihood Estimation (MLE)
# Part 2: Uses only observed data (no mean-filling), divides by common users count
import os
import pandas as pd
import numpy as np
from numpy.linalg import eigh

# Define results directory path
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CODE_DIR, '..', 'results')

# =============================================================================
# Step 1: Load Data and Calculate Item Means
# =============================================================================

from utils import (get_user_avg_ratings, get_item_avg_ratings, 
                   get_target_users, get_target_items,
                   get_preprocessed_dataset, get_top_peers,
                   project_user, compute_covariance_matrix_mle,
                   predict_rating_reconstruction)

print("=" * 70)
print("Part 2: PCA with Maximum Likelihood Estimation (MLE)")
print("=" * 70)

# Load item average ratings (μ_j for each item)
r_i = get_item_avg_ratings()
print(f"\nItem Average Ratings loaded: {r_i.shape[0]:,} items")

# Load target users and items
target_users = get_target_users()
target_items = get_target_items()
print(f"Target Users: {target_users}")
print(f"Target Items: {target_items}")

# Load the preprocessed dataset
df = get_preprocessed_dataset()
print(f"Dataset loaded: {df.shape[0]:,} ratings")

# Get all unique items
all_items = sorted(r_i['item'].tolist())
print(f"Total items: {len(all_items):,}")

# Create item means dictionary
item_means_dict = r_i.set_index('item')['r_i_bar'].to_dict()

# Show means for target items
print("\n--- Item Means for Target Items ---")
for i, item_id in enumerate(target_items, 1):
    print(f"I{i} (Item {item_id}): μ = {item_means_dict[item_id]:.6f}")

# =============================================================================
# Phase 1: Compute MLE Covariance Matrix
# =============================================================================

print("\n" + "=" * 70)
print("Phase 1: Compute MLE Covariance Matrix")
print("=" * 70)



# Check for cached covariance matrix
cov_cache_path = os.path.join(RESULTS_DIR, 'mle_covariance_matrix.csv')
if os.path.exists(cov_cache_path):
    print("\n[Loading cached MLE covariance matrix...]")
    cov_matrix_mle = pd.read_csv(cov_cache_path, index_col=0)
    cov_matrix_mle.index = cov_matrix_mle.index.astype(int)
    cov_matrix_mle.columns = cov_matrix_mle.columns.astype(int)
    print("Loaded from cache.")
else:
    print("\nComputing MLE covariance matrix...")
    cov_matrix_mle = compute_covariance_matrix_mle(df, all_items, r_i, show_progress=True)
    
    # Save covariance matrix
    print("\n[Saving MLE covariance matrix...]")
    cov_matrix_mle.to_csv(cov_cache_path)
    print("Saved to results folder.")

print(f"\n--- MLE Covariance Matrix Shape ---")
print(f"Shape: {cov_matrix_mle.shape}")

# Show covariance for target items
print(f"\n--- Covariance Values for Target Items ---")
print(f"Var(I1): {cov_matrix_mle.loc[target_items[0], target_items[0]]:.6f}")
print(f"Var(I2): {cov_matrix_mle.loc[target_items[1], target_items[1]]:.6f}")
print(f"Cov(I1, I2): {cov_matrix_mle.loc[target_items[0], target_items[1]]:.6f}")

# =============================================================================
# Phase 2: Eigen Decomposition
# =============================================================================

print("\n" + "=" * 70)
print("Phase 2: Eigen Decomposition")
print("=" * 70)

# Convert to numpy array
cov_array = cov_matrix_mle.values

print("\nComputing eigenvalues and eigenvectors...")
eigenvalues, eigenvectors = eigh(cov_array)

# Sort in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

print(f"\n--- Top 10 Eigenvalues ---")
for i in range(10):
    print(f"  λ_{i+1} = {eigenvalues[i]:.6f}")

# Total variance
total_variance = np.sum(eigenvalues)
print(f"\nTotal variance: {total_variance:.6f}")

# Variance explained
print(f"\n--- Variance Explained ---")
for k in [5, 10]:
    var_explained = np.sum(eigenvalues[:k]) / total_variance * 100
    print(f"  Top-{k} PCs: {var_explained:.2f}%")

# Create projection matrices
W_top5 = eigenvectors[:, :5]
W_top10 = eigenvectors[:, :10]
print(f"\nProjection matrix W_5 shape: {W_top5.shape}")
print(f"Projection matrix W_10 shape: {W_top10.shape}")

# --- Top Peers for Target Items ---
print("\n--- Top Peers for Target Items (based on MLE covariance) ---")

top_peers_data = []
for i, item_id in enumerate(target_items, 1):
    print(f"\n--- Target Item I{i} (Item {item_id}) ---")
    
    # Top 5 peers
    top5 = get_top_peers(cov_matrix_mle, item_id, 5)
    print(f"\nTop 5 Peers:")
    for rank, (peer_id, cov_val) in enumerate(top5.items(), 1):
        print(f"  {rank}. Item {peer_id}: Cov = {cov_val:.6f}")
    
    # Top 10 peers
    top10 = get_top_peers(cov_matrix_mle, item_id, 10)
    print(f"\nTop 10 Peers:")
    for rank, (peer_id, cov_val) in enumerate(top10.items(), 1):
        print(f"  {rank}. Item {peer_id}: Cov = {cov_val:.6f}")
        top_peers_data.append({
            'Target_Item': f'I{i}',
            'Target_Item_ID': item_id,
            'Rank': rank,
            'Peer_Item_ID': peer_id,
            'Covariance': cov_val,
            'Is_Top5': rank <= 5
        })

# Save top peers
top_peers_df = pd.DataFrame(top_peers_data)
top_peers_df.to_csv(os.path.join(RESULTS_DIR, 'mle_target_item_peers.csv'), index=False)
print("\n[Saved] Top peers for I1 and I2.")

# =============================================================================
# Phase 3: Dimensionality Reduction - Project Users
# =============================================================================

print("\n" + "=" * 70)
print("Phase 3: Dimensionality Reduction - Calculate User Scores")
print("=" * 70)



# Get all users
all_users = sorted(df['user'].unique().tolist())
print(f"Total users to project: {len(all_users):,}")

# Build user ratings dictionary
print("Building user-item rating lookup...")
user_ratings = df.groupby('user').apply(
    lambda x: dict(zip(x['item'], x['rating']))
).to_dict()

# Project all users using Top-5 PCs
print("\nProjecting users using Top-5 PCs...")
user_scores_top5 = {}
for i, user_id in enumerate(all_users):
    user_scores_top5[user_id] = project_user(
        user_id, W_top5, all_items, item_means_dict, user_ratings
    )
    if (i + 1) % 50000 == 0:
        print(f"  Projected {i+1:,} users...")
print(f"Projected all {len(all_users):,} users to 5-dimensional space.")

# Project all users using Top-10 PCs
print("\nProjecting users using Top-10 PCs...")
user_scores_top10 = {}
for i, user_id in enumerate(all_users):
    user_scores_top10[user_id] = project_user(
        user_id, W_top10, all_items, item_means_dict, user_ratings
    )
    if (i + 1) % 50000 == 0:
        print(f"  Projected {i+1:,} users...")
print(f"Projected all {len(all_users):,} users to 10-dimensional space.")

# Show sample projections for target users
print("\n--- User Scores (T_u) for Target Users ---")
for i, user_id in enumerate(target_users, 1):
    print(f"\nTarget User U{i} (User {user_id}):")
    print(f"  Top-5 Scores:  {user_scores_top5[user_id]}")
    print(f"  Top-10 Scores: {user_scores_top10[user_id]}")

# =============================================================================
# Phase 4: Prediction via Reconstruction
# =============================================================================

print("\n" + "=" * 70)
print("Phase 4: Prediction via Reconstruction")
print("=" * 70)



prediction_results = []

for k_value, user_scores, W, label in [(5, user_scores_top5, W_top5, "Top-5 PCs"), 
                                         (10, user_scores_top10, W_top10, "Top-10 PCs")]:
    print(f"\n=== Predictions Using {label} ===")
    
    for u_idx, target_user in enumerate(target_users, 1):
        print(f"\n--- Target User U{u_idx} (User {target_user}) ---")
        
        for i_idx, target_item in enumerate(target_items, 1):
            # Check actual rating
            actual_rating = None
            if target_user in user_ratings and target_item in user_ratings[target_user]:
                actual_rating = user_ratings[target_user][target_item]
            
            # Predict using reconstruction
            predicted = predict_rating_reconstruction(
                target_user, target_item, user_scores, W,
                item_means_dict, all_items
            )
            
            item_mean = item_means_dict.get(target_item, 0)
            
            # Round predicted to 2 decimals
            predicted_rounded = round(predicted, 2)
            item_mean_rounded = round(item_mean, 2)
            
            # Calculate error: use actual if available, else use item mean
            if actual_rating is not None:
                actual_for_error = actual_rating
                actual_display = str(round(actual_rating, 2))
            else:
                actual_for_error = item_mean
                actual_display = f"{item_mean_rounded} (item mean)"
            
            error = round(abs(predicted - actual_for_error), 2)
            
            print(f"\n  Item I{i_idx} (Item {target_item}):")
            print(f"    Item Mean: {item_mean_rounded}")
            print(f"    Predicted Rating: {predicted_rounded}")
            print(f"    Actual Rating: {actual_display}")
            print(f"    Error: {error}")
            
            prediction_results.append({
                'Method': 'MLE',
                'PCs': k_value,
                'PC_Label': label,
                'Target_User': f'U{u_idx}',
                'User_ID': target_user,
                'Target_Item': f'I{i_idx}',
                'Item_ID': target_item,
                'Item_Mean': item_mean_rounded,
                'Predicted_Rating': predicted_rounded,
                'Actual_Rating': actual_rating if actual_rating else item_mean_rounded,
                'Actual_Source': 'actual' if actual_rating else 'item_mean',
                'Error': error
            })

# Save prediction results
print("\n[Saving predictions...]")
predictions_df = pd.DataFrame(prediction_results)
predictions_df.to_csv(os.path.join(RESULTS_DIR, 'mle_predictions.csv'), index=False)
print("[Saved] Prediction results.")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("PREDICTION SUMMARY (MLE Method)")
print("=" * 70)

print(f"\n{'PCs':<12} {'User':<8} {'Item':<8} {'Predicted':<12} {'Actual':<15} {'Source':<12} {'Error':<10}")
print("-" * 75)
for result in prediction_results:
    actual_str = str(result['Actual_Rating'])
    source_str = result['Actual_Source']
    print(f"{result['PC_Label']:<12} {result['Target_User']:<8} {result['Target_Item']:<8} "
          f"{result['Predicted_Rating']:<12} {actual_str:<15} {source_str:<12} {result['Error']:<10}")

print("\n" + "=" * 70)
print("[SUCCESS] PCA MLE completed! (All phases)")
print("=" * 70)

# =============================================================================
# Output Files Summary
# =============================================================================
print("\n--- Saved Files ---")
print("1. mle_covariance_matrix.csv - MLE covariance matrix")
print("2. mle_target_item_peers.csv - Top 5 & Top 10 peers for each target item")
print("3. mle_predictions.csv - Predictions for target items (Top-5 and Top-10)")
