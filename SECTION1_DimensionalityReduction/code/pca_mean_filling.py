# PCA with Mean Filling
import os
import pandas as pd

# Define results directory path
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CODE_DIR, '..', 'results')
PLOTS_DIR = os.path.join(CODE_DIR, '..', 'plots')

# =============================================================================
# Step 1: Load r_u (user average ratings) and r_i (item average ratings)
# =============================================================================

from utils import (get_user_avg_ratings, get_item_avg_ratings, 
                   get_target_users, get_target_items,
                   get_preprocessed_dataset, create_user_item_matrix,
                   mean_fill_matrix, get_top_peers, project_user,
                   cosine_similarity, predict_rating)

# Load user average ratings (r_u)
r_u = get_user_avg_ratings()
print("User Average Ratings (r_u):")
print(r_u.head())
print(f"Shape: {r_u.shape}\n")

# Load item average ratings (r_i)
r_i = get_item_avg_ratings()
print("Item Average Ratings (r_i):")
print(r_i.head())
print(f"Shape: {r_i.shape}")



# =============================================================================
# Step 2: Load target users and target items
# =============================================================================

# Load target users
target_users = get_target_users()
print("\nTarget Users:")
print(target_users)

# Load target items
target_items = get_target_items()
print("\nTarget Items:")
print(target_items)



# =============================================================================
# Step 3: Mean-Filling Method - Replace missing ratings with column means
# =============================================================================

# Load the preprocessed dataset
df = get_preprocessed_dataset()
print(f"\nPreprocessed Dataset Shape: {df.shape}")

# Filter dataset to only include target items (I1 and I2) for memory efficiency
print("\nFiltering dataset to target items only...")
df_target_items = df[df['item'].isin(target_items)]
print(f"Filtered Dataset Shape: {df_target_items.shape}")

# Create user-item rating matrix for target items only
print("\nCreating User-Item Rating Matrix for Target Items...")
rating_matrix_target = create_user_item_matrix(df_target_items)
print(f"Rating Matrix Shape: {rating_matrix_target.shape}")
print(f"(Rows = Users who rated target items, Columns = Target Items)")

# Show missing values before filling for target items
total_entries = rating_matrix_target.shape[0] * rating_matrix_target.shape[1]
missing_before = rating_matrix_target.isna().sum().sum()
print(f"\nMissing values before filling: {missing_before:,} ({missing_before/total_entries*100:.2f}%)")

# Calculate mean for each target item (column)
print("\n--- Mean Values for Target Items (I1 and I2) ---")
item_means = {}
for i, item_id in enumerate(target_items, 1):
    item_mean = rating_matrix_target[item_id].mean()
    item_means[item_id] = item_mean
    print(f"I{i} (Item {item_id}): Mean = {item_mean:.6f}")

# Apply mean-filling: Replace missing entries with column means
print("\nApplying Mean-Filling Method...")
rating_matrix_filled = mean_fill_matrix(rating_matrix_target)

# Verify no missing values after filling
missing_after = rating_matrix_filled.isna().sum().sum()
print(f"Missing values after filling: {missing_after}")

# Show sample of original vs filled matrix for comparison
print("\n--- Original Matrix (with missing values) ---")
print(rating_matrix_target.head(10))

print("\n--- Filled Matrix (missing values replaced with mean) ---")
print(rating_matrix_filled.head(10))



# =============================================================================
# Step 4: Calculate average rating for each item (using r_i)
# =============================================================================

print("\n" + "="*70)
print("Step 4: Calculate Average Rating for Each Item")
print("="*70)

# We already have r_i loaded, which contains item means
# Display means for target items
print("\nItem means for target items (from r_i):")
for i, item_id in enumerate(target_items, 1):
    item_mean = r_i[r_i['item'] == item_id]['r_i_bar'].values[0]
    print(f"I{i} (Item {item_id}): Mean = {item_mean:.6f}")



# =============================================================================
# Step 5: Calculate centered ratings (actual - mean) for ALL items
# =============================================================================

print("\n" + "="*70)
print("Step 5: Calculate Centered Ratings (actual - mean) for ALL Items")
print("="*70)

from utils import compute_centered_ratings

# Compute centered ratings for ALL items (entire dataset)
df_centered = compute_centered_ratings(df, r_i)
print(f"\nCentered Ratings DataFrame Shape: {df_centered.shape}")
print("\nSample of centered ratings:")
print(df_centered.head(10))

# Show centered ratings for target items
print(f"\n--- Centered ratings sample for target items (I1={target_items[0]}, I2={target_items[1]}) ---")
df_centered_targets = df_centered[df_centered['item'].isin(target_items)]
print(df_centered_targets.head(10))



# Step 6: Compute Covariance Matrix for ALL Items (Memory-Efficient)
# =============================================================================

print("\n" + "="*70)
print("Step 6: Compute Covariance Matrix for ALL Items")
print("="*70)

from utils import compute_covariance_matrix_efficient

# Get total number of users (N)
total_users = r_u.shape[0]
print(f"\nTotal number of users (N): {total_users:,}")

# Get all unique items from the dataset
all_items = sorted(r_i['item'].tolist())
print(f"Total number of items: {len(all_items):,}")

# Compute covariance matrix for ALL items
print("\nComputing covariance matrix for ALL items...")
print("(Using memory-efficient algorithm: only users who rated BOTH items contribute)")
print("(Dividing by N-1 for sample covariance)")
print("This may take a few minutes...")

cov_matrix = compute_covariance_matrix_efficient(
    df, all_items, r_i, total_users, show_progress=True
)

print(f"\n--- Covariance Matrix Shape ---")
print(f"Shape: {cov_matrix.shape} ({len(all_items)} x {len(all_items)})")

# Show covariance for target items specifically
print(f"\n--- Covariance Values for Target Items (I1={target_items[0]}, I2={target_items[1]}) ---")
print(f"Var(I1): {cov_matrix.loc[target_items[0], target_items[0]]:.6f}")
print(f"Var(I2): {cov_matrix.loc[target_items[1], target_items[1]]:.6f}")
print(f"Cov(I1, I2): {cov_matrix.loc[target_items[0], target_items[1]]:.6f}")

# Show a sample of the covariance matrix (first 5x5)
print("\n--- Sample of Covariance Matrix (first 5x5) ---")
print(cov_matrix.iloc[:5, :5])

# Save Step 6 results
print("\n[Saving Step 6 results...]")
cov_matrix.to_csv(os.path.join(RESULTS_DIR, 'step6_covariance_matrix.csv'))
print(f"Saved full covariance matrix ({len(all_items)}x{len(all_items)}) to results folder.")

# =============================================================================
# Step 7: Eigenvalue Decomposition - Determine Top 5 and Top 10 Principal Components
# =============================================================================

print("\n" + "="*70)
print("Step 7: PCA Eigenvalue Decomposition")
print("="*70)

import numpy as np
from numpy.linalg import eigh

# Convert covariance matrix to numpy array for eigendecomposition
cov_array = cov_matrix.values

print("\nComputing eigenvalues and eigenvectors of the covariance matrix...")
print(f"Covariance matrix shape: {cov_array.shape}")

# Compute eigenvalues and eigenvectors
# eigh is for symmetric matrices (covariance is symmetric)
eigenvalues, eigenvectors = eigh(cov_array)

# Sort eigenvalues and eigenvectors in descending order
# (eigh returns in ascending order)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

print(f"\n--- Eigenvalues (Top 10) ---")
for i in range(10):
    print(f"  lambda_{i+1} = {eigenvalues[i]:.6f}")

# Total variance
total_variance = np.sum(eigenvalues)
print(f"\nTotal variance (sum of all eigenvalues): {total_variance:.6f}")

# Variance explained by top-k PCs
print(f"\n--- Variance Explained ---")
for k in [5, 10]:
    var_explained = np.sum(eigenvalues[:k]) / total_variance * 100
    print(f"  Top-{k} PCs: {var_explained:.2f}%")

# Top 5 Principal Components (eigenvectors)
print(f"\n--- Top 5 Principal Components (Eigenvectors) ---")
top5_eigenvectors = eigenvectors[:, :5]
print(f"Shape: {top5_eigenvectors.shape} (n_items x 5)")

# Top 10 Principal Components (eigenvectors)
print(f"\n--- Top 10 Principal Components (Eigenvectors) ---")
top10_eigenvectors = eigenvectors[:, :10]
print(f"Shape: {top10_eigenvectors.shape} (n_items x 10)")

# Create projection matrix W for top-5 (this will be used in Step 8)
W_top5 = top5_eigenvectors
W_top10 = top10_eigenvectors

# =============================================================================
# Covariance Matrix: Before vs After Reduction
# =============================================================================

print("\n" + "="*70)
print("Covariance Matrix: Before vs After Reduction")
print("="*70)

# Before reduction: Original covariance matrix (for target items)
print("\n--- BEFORE Reduction (Original Covariance for Target Items) ---")
cov_before = cov_matrix.loc[target_items, target_items]
print(cov_before.round(6))

# After reduction: Reconstructed covariance using Top-5 and Top-10 PCs
# Reconstructed Σ ≈ W @ Λ @ W.T where Λ = diag(eigenvalues[:k])

# Top-5 reconstruction
Lambda_5 = np.diag(eigenvalues[:5])
cov_reconstructed_5 = W_top5 @ Lambda_5 @ W_top5.T
cov_reconstructed_5_df = pd.DataFrame(cov_reconstructed_5, index=all_items, columns=all_items)

print("\n--- AFTER Reduction: Reconstructed Covariance (Top-5 PCs) for Target Items ---")
cov_after_5 = cov_reconstructed_5_df.loc[target_items, target_items]
print(cov_after_5.round(6))

# Top-10 reconstruction
Lambda_10 = np.diag(eigenvalues[:10])
cov_reconstructed_10 = W_top10 @ Lambda_10 @ W_top10.T
cov_reconstructed_10_df = pd.DataFrame(cov_reconstructed_10, index=all_items, columns=all_items)

print("\n--- AFTER Reduction: Reconstructed Covariance (Top-10 PCs) for Target Items ---")
cov_after_10 = cov_reconstructed_10_df.loc[target_items, target_items]
print(cov_after_10.round(6))

# Save covariance files separately
print("\n[Saving covariance matrices...]")

# Before reduction
cov_before.to_csv(os.path.join(RESULTS_DIR, 'meanfill_covariance_before_reduction.csv'))
print("[Saved] meanfill_covariance_before_reduction.csv")

# After Top-5 reduction
cov_after_5.to_csv(os.path.join(RESULTS_DIR, 'meanfill_covariance_after_top5.csv'))
print("[Saved] meanfill_covariance_after_top5.csv")

# After Top-10 reduction
cov_after_10.to_csv(os.path.join(RESULTS_DIR, 'meanfill_covariance_after_top10.csv'))
print("[Saved] meanfill_covariance_after_top10.csv")

# --- Determine Top 5 and Top 10 Peers for Each Target Item ---
print("\n--- Top Peers for Target Items (based on covariance) ---")

# Find top peers for each target item
top_peers_results = {}
for i, item_id in enumerate(target_items, 1):
    print(f"\n--- Target Item I{i} (Item {item_id}) ---")
    
    # Top 5 peers
    top5 = get_top_peers(cov_matrix, item_id, 5)
    print(f"\nTop 5 Peers:")
    for rank, (peer_id, cov_val) in enumerate(top5.items(), 1):
        print(f"  {rank}. Item {peer_id}: Cov = {cov_val:.6f}")
    
    # Top 10 peers
    top10 = get_top_peers(cov_matrix, item_id, 10)
    print(f"\nTop 10 Peers:")
    for rank, (peer_id, cov_val) in enumerate(top10.items(), 1):
        print(f"  {rank}. Item {peer_id}: Cov = {cov_val:.6f}")
    
    top_peers_results[f'I{i}'] = {
        'item_id': item_id,
        'top5_peers': top5.index.tolist(),
        'top10_peers': top10.index.tolist()
    }

# Save top peers for target items
top_peers_data = []
for i, item_id in enumerate(target_items, 1):
    top10 = get_top_peers(cov_matrix, item_id, 10)
    for rank, (peer_id, cov_val) in enumerate(top10.items(), 1):
        top_peers_data.append({
            'Target_Item': f'I{i}',
            'Target_Item_ID': item_id,
            'Rank': rank,
            'Peer_Item_ID': peer_id,
            'Covariance': cov_val,
            'Is_Top5': rank <= 5
        })
top_peers_df = pd.DataFrame(top_peers_data)
top_peers_df.to_csv(os.path.join(RESULTS_DIR, 'step7_target_item_peers.csv'), index=False)
print("\nSaved top peers for I1 and I2 to results folder.")

# =============================================================================
# Step 8 & 10: Project Users into Reduced Latent Space (Top-5 and Top-10)
# =============================================================================

print("\n" + "="*70)
print("Step 8 & 10: Project Users into Reduced Latent Space")
print("="*70)

# Formula: UserVector_u = R'_u × W_k
# Optimization: Only iterate over items the user actually rated (missing = 0)
# UserVector_u[dim] = Sum_{j in Observed} (R_{u,j} - μ_j) × W_{j, dim}

# Get item means as a dictionary for fast lookup
item_means_dict = r_i.set_index('item')['r_i_bar'].to_dict()

# Get all users from the dataset
all_users = sorted(df['user'].unique().tolist())
print(f"\nTotal users to project: {len(all_users):,}")

# Create a mapping: user -> {item: rating}
print("Building user-item rating lookup...")
user_ratings = df.groupby('user').apply(
    lambda x: dict(zip(x['item'], x['rating']))
).to_dict()

# project_user function imported from utils

# Project all users using Top-5 PCs
print("\nProjecting all users using Top-5 PCs...")
user_vectors_top5 = {}
for i, user_id in enumerate(all_users):
    user_vectors_top5[user_id] = project_user(
        user_id, W_top5, all_items, item_means_dict, user_ratings
    )
    if (i + 1) % 50000 == 0:
        print(f"  Projected {i+1:,} users...")
print(f"Projected all {len(all_users):,} users to 5-dimensional space.")

# Project all users using Top-10 PCs
print("\nProjecting all users using Top-10 PCs...")
user_vectors_top10 = {}
for i, user_id in enumerate(all_users):
    user_vectors_top10[user_id] = project_user(
        user_id, W_top10, all_items, item_means_dict, user_ratings
    )
    if (i + 1) % 50000 == 0:
        print(f"  Projected {i+1:,} users...")
print(f"Projected all {len(all_users):,} users to 10-dimensional space.")

# Show sample projections for target users
print("\n--- Sample User Projections (Target Users) ---")
for i, user_id in enumerate(target_users, 1):
    print(f"\nTarget User U{i} (User {user_id}):")
    print(f"  Top-5 Vector:  {user_vectors_top5[user_id]}")
    print(f"  Top-10 Vector: {user_vectors_top10[user_id]}")



# =============================================================================
# Step 9 & 11: User Similarity and Rating Prediction
# =============================================================================

print("\n" + "="*70)
print("Step 9 & 11: User Similarity and Rating Prediction")
print("="*70)

# cosine_similarity and predict_rating functions imported from utils

# Make predictions for target users on target items
print("\n--- Rating Predictions for Target Users on Target Items ---")
n_neighbors = 20  # Number of nearest neighbors

prediction_results = []

for k_value, user_vectors, label in [(5, user_vectors_top5, "Top-5 PCs"), 
                                       (10, user_vectors_top10, "Top-10 PCs")]:
    print(f"\n=== Using {label} ===")
    
    for u_idx, target_user in enumerate(target_users, 1):
        print(f"\n--- Target User U{u_idx} (User {target_user}) ---")
        
        for i_idx, target_item in enumerate(target_items, 1):
            # Check if user already rated this item
            actual_rating = None
            if target_user in user_ratings and target_item in user_ratings[target_user]:
                actual_rating = user_ratings[target_user][target_item]
            
            predicted, n_used, neighbors = predict_rating(
                target_user, target_item, user_vectors, 
                item_means_dict, user_ratings, n_neighbors
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
            print(f"    Neighbors Used: {n_used}")
            print(f"    Predicted Rating: {predicted_rounded}")
            print(f"    Actual Rating: {actual_display}")
            print(f"    Error: {error}")
            
            # Show top 5 neighbors for detail
            if neighbors:
                print(f"    Top 5 Neighbors:")
                for n_info in neighbors[:5]:
                    print(f"      User {n_info['neighbor_user']}: "
                          f"Sim={round(n_info['similarity'], 2)}, "
                          f"Rating={round(n_info['actual_rating'], 2)}, "
                          f"Centered={round(n_info['centered_rating'], 2)}")
            
            prediction_results.append({
                'PCs': k_value,
                'PC_Label': label,
                'Target_User': f'U{u_idx}',
                'User_ID': target_user,
                'Target_Item': f'I{i_idx}',
                'Item_ID': target_item,
                'Item_Mean': item_mean_rounded,
                'Neighbors_Used': n_used,
                'Predicted_Rating': predicted_rounded,
                'Actual_Rating': actual_rating if actual_rating else item_mean_rounded,
                'Actual_Source': 'actual' if actual_rating else 'item_mean',
                'Error': error
            })

# Save prediction results
print("\n[Saving Step 9 & 11 results...]")
predictions_df = pd.DataFrame(prediction_results)
predictions_df.to_csv(os.path.join(RESULTS_DIR, 'step9_11_predictions.csv'), index=False)
print("Saved prediction results to results folder.")

# Summary table
print("\n" + "="*70)
print("PREDICTION SUMMARY")
print("="*70)

print("\n--- Rating Predictions Summary ---")
print(f"{'PCs':<10} {'User':<8} {'Item':<8} {'Predicted':<12} {'Actual':<15} {'Source':<12} {'Error':<10}")
print("-" * 75)
for result in prediction_results:
    actual_str = str(result['Actual_Rating'])
    source_str = result['Actual_Source']
    print(f"{result['PC_Label']:<10} {result['Target_User']:<8} {result['Target_Item']:<8} "
          f"{result['Predicted_Rating']:<12} {actual_str:<15} {source_str:<12} {result['Error']:<10}")

print("\n" + "="*70)
print("[SUCCESS] All steps completed! (Steps 1-11)")
print("="*70)
