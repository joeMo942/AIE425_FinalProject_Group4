# PCA with Mean Filling
import os
import pandas as pd

# Define results directory path
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CODE_DIR, '..', 'results')
PLOTS_DIR = os.path.join(CODE_DIR, '..', 'plots')

# =============================================================================
# Step 0: Data Loading (Preparation)
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
# Step 0 (continued): Load target users and target items
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
# Step 1 & 2: Calculate average rating for I1/I2 + Mean-Filling Method
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
# Step 3: Calculate average rating for each item (using r_i)
# =============================================================================

print("\n" + "="*70)
print("Step 3: Calculate Average Rating for Each Item")
print("="*70)

# We already have r_i loaded, which contains item means
# Display means for target items
print("\nItem means for target items (from r_i):")
for i, item_id in enumerate(target_items, 1):
    item_mean = r_i[r_i['item'] == item_id]['r_i_bar'].values[0]
    print(f"I{i} (Item {item_id}): Mean = {item_mean:.6f}")



# =============================================================================
# Step 4: Calculate centered ratings (actual - mean) for ALL items
# =============================================================================

print("\n" + "="*70)
print("Step 4: Calculate Centered Ratings (actual - mean) for ALL Items")
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



# Step 5 & 6: Compute Covariance Matrix for ALL Items (Memory-Efficient)
# =============================================================================

print("\n" + "="*70)
print("Step 5 & 6: Compute Covariance Matrix for ALL Items")
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
print("\n[Saving Step 5 & 6 results...]")
cov_matrix.to_csv(os.path.join(RESULTS_DIR, 'meanfill_covariance_matrix.csv'))
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

# Save top 5 and top 10 eigenvalues
print("\n[Saving eigenvalues...]")
top5_eigenvalues_df = pd.DataFrame({
    'PC': [f'PC{i+1}' for i in range(5)],
    'Eigenvalue': eigenvalues[:5],
    'Variance_Explained_Pct': (eigenvalues[:5] / total_variance) * 100
})
top5_eigenvalues_df.to_csv(os.path.join(RESULTS_DIR, 'meanfill_top5_eigenvalues.csv'), index=False)
print("[Saved] meanfill_top5_eigenvalues.csv")

top10_eigenvalues_df = pd.DataFrame({
    'PC': [f'PC{i+1}' for i in range(10)],
    'Eigenvalue': eigenvalues[:10],
    'Variance_Explained_Pct': (eigenvalues[:10] / total_variance) * 100
})
top10_eigenvalues_df.to_csv(os.path.join(RESULTS_DIR, 'meanfill_top10_eigenvalues.csv'), index=False)
print("[Saved] meanfill_top10_eigenvalues.csv")

# =============================================================================
# Visualization 1: Top-10 Eigenvalues Bar Chart
# =============================================================================
import matplotlib.pyplot as plt

print("\n[Creating Eigenvalues Bar Chart...]")
fig, ax = plt.subplots(figsize=(8, 5))

pcs = [f'PC{i+1}' for i in range(10)]
top10_eig = eigenvalues[:10]

ax.bar(pcs, top10_eig, color='steelblue', alpha=0.8)
ax.set_xlabel('Principal Component', fontsize=12)
ax.set_ylabel('Eigenvalue', fontsize=12)
ax.set_title('Mean-Fill PCA: Top 10 Eigenvalues', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'meanfill_eigenvalues.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] meanfill_eigenvalues.png")


# =============================================================================
# Covariance Matrix: Before vs After Reduction
# =============================================================================

print("\n" + "=" * 70)
print("Covariance Matrix: Before vs After Reduction")
print("=" * 70)

# Before reduction: Original covariance matrix (FULL)
print("\n--- BEFORE Reduction (Original Covariance Matrix) ---")
# cov_matrix is already the full matrix
print(f"Shape: {cov_matrix.shape}")

# After reduction: Reconstructed covariance using Top-5 and Top-10 PCs
# Reconstructed Σ ≈ W @ Λ @ W.T where Λ = diag(eigenvalues[:k])

# Top-5 reconstruction (FULL)
print("\nComputing Reconstructed Covariance Matrix (Top-5 PCs)...")
Lambda_5 = np.diag(eigenvalues[:5])
cov_reconstructed_5 = W_top5 @ Lambda_5 @ W_top5.T
cov_reconstructed_5_df = pd.DataFrame(cov_reconstructed_5, index=all_items, columns=all_items)
print(f"Shape: {cov_reconstructed_5_df.shape}")

# Top-10 reconstruction (FULL)
print("\nComputing Reconstructed Covariance Matrix (Top-10 PCs)...")
Lambda_10 = np.diag(eigenvalues[:10])
cov_reconstructed_10 = W_top10 @ Lambda_10 @ W_top10.T
cov_reconstructed_10_df = pd.DataFrame(cov_reconstructed_10, index=all_items, columns=all_items)
print(f"Shape: {cov_reconstructed_10_df.shape}")

# Save covariance files separately (FULL MATRICES)
print("\n[Saving FULL covariance matrices...]")

# Before reduction
cov_matrix.to_csv(os.path.join(RESULTS_DIR, 'meanfill_covariance_before_reduction.csv'))
print("[Saved] meanfill_covariance_before_reduction.csv")

# After Top-5 reduction
cov_reconstructed_5_df.to_csv(os.path.join(RESULTS_DIR, 'meanfill_covariance_after_top5.csv'))
print("[Saved] meanfill_covariance_after_top5.csv")

# After Top-10 reduction
cov_reconstructed_10_df.to_csv(os.path.join(RESULTS_DIR, 'meanfill_covariance_after_top10.csv'))
print("[Saved] meanfill_covariance_after_top10.csv")

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
predictions_df.to_csv(os.path.join(RESULTS_DIR, 'meanfill_predictions.csv'), index=False)
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

# =============================================================================
# Visualization 2: Prediction Error Comparison Bar Chart
# =============================================================================
print("\n[Creating Prediction Error Comparison Chart...]")

# Prepare data for visualization
top5_results = [r for r in prediction_results if r['PCs'] == 5]
top10_results = [r for r in prediction_results if r['PCs'] == 10]

labels = [f"{r['Target_User']}-{r['Target_Item']}" for r in top5_results]
top5_errors = [r['Error'] for r in top5_results]
top10_errors = [r['Error'] for r in top10_results]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, top5_errors, width, label='Top-5 PCs', color='coral', alpha=0.8)
bars2 = ax.bar(x + width/2, top10_errors, width, label='Top-10 PCs', color='teal', alpha=0.8)

ax.set_xlabel('User-Item Pair', fontsize=12)
ax.set_ylabel('Absolute Error', fontsize=12)
ax.set_title('Mean-Fill PCA: Prediction Error Comparison (Top-5 vs Top-10 PCs)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

fig.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'meanfill_error_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] meanfill_error_comparison.png")

# =============================================================================
# Visualization 3: Average Error Bar Chart (Top-5 vs Top-10)
# =============================================================================
print("\n[Creating Average Error Bar Chart...]")

fig, ax = plt.subplots(figsize=(6, 5))

# Calculate average errors
avg_top5 = sum(top5_errors) / len(top5_errors)
avg_top10 = sum(top10_errors) / len(top10_errors)

methods = ['Top-5 PCs', 'Top-10 PCs']
avg_errors = [avg_top5, avg_top10]
colors = ['coral', 'teal']

bars = ax.bar(methods, avg_errors, color=colors, alpha=0.8)
ax.set_ylabel('Average Error', fontsize=12)
ax.set_title('Mean-Fill PCA: Average Prediction Error', fontsize=14)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'meanfill_avg_error.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] meanfill_avg_error.png")

print("\n[All visualizations saved to plots folder!]")

