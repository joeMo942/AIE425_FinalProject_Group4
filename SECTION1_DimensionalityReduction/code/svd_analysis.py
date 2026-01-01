# SVD Analysis
# Full SVD Decomposition with Data Preparation and Visualization
# Memory-optimized version for large sparse matrices

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import gc

# Set UTF-8 encoding for console output
sys.stdout.reconfigure(encoding='utf-8')

from utils import (
    get_preprocessed_dataset,
    get_item_avg_ratings,
    get_target_users,
    get_target_items,
    CODE_DIR
)

# Output directories
RESULTS_DIR = os.path.join(CODE_DIR, '..', 'results')
PLOTS_DIR = os.path.join(CODE_DIR, '..', 'plots')
TABLES_DIR = os.path.join(CODE_DIR, '..', 'tables')

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


# =============================================================================
# 1. DATA PREPARATION (Memory Optimized)
# =============================================================================

def load_ratings_matrix(max_users=10000, max_items=2000):
    """
    1.1 Load ratings matrix from preprocessed dataset.
    For memory efficiency, select top users and items by rating count.
    
    Args:
        max_users: Maximum number of users to include
        max_items: Maximum number of items to include
    
    Returns:
        pd.DataFrame: Pivot table with users as rows and items as columns.
        np.ndarray: User IDs
        np.ndarray: Item IDs
    """
    print("=" * 60)
    print("1. DATA PREPARATION")
    print("=" * 60)
    
    print("\n[LOAD] Loading preprocessed dataset...")
    df = get_preprocessed_dataset()
    print(f"        Full dataset shape: {df.shape}")
    print(f"        Columns: {df.columns.tolist()}")
    
    # Get unique counts
    n_users = df['user'].nunique()
    n_items = df['item'].nunique()
    print(f"        Total unique users: {n_users:,}")
    print(f"        Total unique items: {n_items:,}")
    
    # Select top users by number of ratings
    print(f"\n[FILTER] Selecting top {max_users} users and {max_items} items...")
    user_counts = df['user'].value_counts()
    top_users = user_counts.head(max_users).index.tolist()
    
    item_counts = df['item'].value_counts()
    top_items = item_counts.head(max_items).index.tolist()
    
    # Filter dataset
    df_filtered = df[(df['user'].isin(top_users)) & (df['item'].isin(top_items))]
    print(f"        Filtered dataset shape: {df_filtered.shape}")
    
    # Create pivot table (user-item matrix)
    print("\n[BUILD] Creating user-item ratings matrix...")
    ratings_matrix = df_filtered.pivot_table(
        index='user', 
        columns='item', 
        values='rating',
        aggfunc='mean'  # In case of duplicates, take mean
    )
    
    user_ids = ratings_matrix.index.values
    item_ids = ratings_matrix.columns.values
    
    print(f"        Matrix dimensions: {ratings_matrix.shape[0]} users x {ratings_matrix.shape[1]} items")
    print(f"        Total cells: {ratings_matrix.shape[0] * ratings_matrix.shape[1]:,}")
    
    # Analyze user activity levels (based on % of items rated)
    n_items_in_matrix = len(item_ids)
    user_rating_counts = (~ratings_matrix.isna()).sum(axis=1)
    user_rating_pct = user_rating_counts / n_items_in_matrix * 100
    
    cold_users = (user_rating_pct <= 2).sum()
    medium_users = ((user_rating_pct > 2) & (user_rating_pct <= 5)).sum()
    rich_users = (user_rating_pct > 10).sum()
    
    print(f"\n[USER ANALYSIS] User Activity Distribution:")
    print(f"        Cold users (≤2% ratings): {cold_users:,}")
    print(f"        Medium users (2-5% ratings): {medium_users:,}")
    print(f"        Rich users (>10% ratings): {rich_users:,}")
    
    # Analyze item popularity (based on % of users who rated)
    n_users_in_matrix = len(user_ids)
    item_rating_counts = (~ratings_matrix.isna()).sum(axis=0)
    item_rating_pct = item_rating_counts / n_users_in_matrix * 100
    
    low_pop_items = (item_rating_pct <= 2).sum()
    medium_pop_items = ((item_rating_pct > 2) & (item_rating_pct <= 10)).sum()
    high_pop_items = (item_rating_pct > 10).sum()
    
    print(f"\n[ITEM ANALYSIS] Item Popularity Distribution:")
    print(f"        Low popularity (≤2% users): {low_pop_items:,}")
    print(f"        Medium popularity (2-10% users): {medium_pop_items:,}")
    print(f"        High popularity (>10% users): {high_pop_items:,}")
    
    # Verify dataset requirements
    print(f"\n[VERIFY] Dataset Requirements Check:")
    print(f"        ≥10,000 users: {len(user_ids):,} {'✓' if len(user_ids) >= 10000 else '✗'}")
    print(f"        ≥500 items: {len(item_ids):,} {'✓' if len(item_ids) >= 500 else '✗'}")
    n_ratings = (~ratings_matrix.isna()).sum().sum()
    print(f"        ≥100,000 ratings: {n_ratings:,} {'✓' if n_ratings >= 100000 else '✗'}")
    
    # Clean up
    del df, df_filtered
    gc.collect()
    
    return ratings_matrix, user_ids, item_ids


def calculate_item_averages(ratings_matrix):
    """
    1.2 Calculate the average rating for each item (r_i).
    
    Args:
        ratings_matrix: DataFrame with users as rows and items as columns.
        
    Returns:
        pd.Series: Average rating for each item.
    """
    print("\n[CALC] Calculating average rating for each item (r_i)...")
    item_averages = ratings_matrix.mean(axis=0, skipna=True)
    
    print(f"        Number of items: {len(item_averages)}")
    print(f"        Average rating range: [{item_averages.min():.4f}, {item_averages.max():.4f}]")
    print(f"        Global average: {item_averages.mean():.4f}")
    
    return item_averages


def apply_mean_filling(ratings_matrix, item_averages):
    """
    1.3 Apply mean-filling: replace missing ratings with item's average rating.
    Memory-optimized using numpy operations.
    
    Args:
        ratings_matrix: DataFrame with users as rows and items as columns.
        item_averages: Series with average rating for each item.
        
    Returns:
        np.ndarray: Filled ratings matrix with no missing values.
    """
    print("\n[FILL] Applying mean-filling for missing values...")
    
    # Convert to numpy for memory efficiency
    matrix = ratings_matrix.values.astype(np.float32)  # Use float32 to save memory
    
    missing_before = np.isnan(matrix).sum()
    total_cells = matrix.shape[0] * matrix.shape[1]
    sparsity = missing_before / total_cells * 100
    
    print(f"        Missing values before: {missing_before:,} ({sparsity:.2f}%)")
    
    # Fill missing values with item averages (column-wise)
    item_avg_array = item_averages.values.astype(np.float32)
    
    # Create filled matrix
    for j in range(matrix.shape[1]):
        col_mask = np.isnan(matrix[:, j])
        matrix[col_mask, j] = item_avg_array[j]
    
    # Handle any remaining NaN (items with no ratings at all)
    global_avg = np.nanmean(item_avg_array)
    matrix = np.nan_to_num(matrix, nan=global_avg)
    
    missing_after = np.isnan(matrix).sum()
    print(f"        Missing values after: {missing_after}")
    
    return matrix


def verify_completeness(filled_matrix):
    """
    1.4 Verify matrix completeness (no missing values).
    
    Args:
        filled_matrix: Numpy array that should have no missing values.
        
    Returns:
        bool: True if matrix is complete, False otherwise.
    """
    print("\n[CHECK] Verifying matrix completeness...")
    
    missing_after = np.isnan(filled_matrix).sum()
    
    if missing_after == 0:
        print("        [OK] Matrix is complete - no missing values!")
        return True
    else:
        print(f"        [WARNING] {missing_after} missing values remain!")
        return False


# =============================================================================
# 2. FULL SVD DECOMPOSITION
# =============================================================================

def compute_full_svd(filled_matrix):
    """
    2.1 Compute the full SVD: R = U * Sigma * Vt
    2.2 Calculate eigenpairs, singular values, and matrices.
    
    Args:
        filled_matrix: Complete ratings matrix (no missing values).
        
    Returns:
        tuple: (U, sigma, Vt, eigenvalues, eigenvectors)
    """
    print("\n" + "=" * 60)
    print("2. FULL SVD DECOMPOSITION")
    print("=" * 60)
    
    # Use the numpy array directly
    R = filled_matrix.astype(np.float64)  # Use float64 for SVD precision
    m, n = R.shape
    print(f"\n[SVD] Computing full SVD for matrix R ({m} x {n})...")
    print("        This may take a few minutes...")
    
    # Compute full SVD: R = U @ Sigma @ Vt
    # full_matrices=True gives:
    #   U: m x m (5000 x 5000)
    #   sigma: min(m,n) singular values (2000)
    #   Vt: n x n (2000 x 2000)
    U, sigma, Vt = np.linalg.svd(R, full_matrices=True)
    
    # Create full Sigma matrix (m x n) with singular values on diagonal
    Sigma_full = np.zeros((m, n))
    np.fill_diagonal(Sigma_full, sigma)
    
    print(f"        U shape: {U.shape} (m x m)")
    print(f"        Σ shape: {Sigma_full.shape} (m x n)")
    print(f"        Vt shape: {Vt.shape} (n x n)")
    
    # 2.2 Calculate eigenpairs
    print("\n[EIGEN] Computing eigenpairs from singular values...")
    
    # Eigenvalues are sigma^2 (singular values squared)
    eigenvalues = sigma ** 2
    
    # Eigenvectors are the columns of V (rows of Vt)
    V = Vt.T  # V matrix (item latent factors)
    eigenvectors = V
    
    print(f"        Number of eigenvalues: {len(eigenvalues)}")
    print(f"        Largest eigenvalue: {eigenvalues[0]:.4f}")
    print(f"        Smallest eigenvalue: {eigenvalues[-1]:.6f}")
    
    # Verify normalization of V columns (should already be orthonormal from SVD)
    print("\n[NORM] Verifying V columns are normalized (||v_i|| = 1)...")
    norms = np.linalg.norm(V, axis=0)
    print(f"        Column norms range: [{norms.min():.6f}, {norms.max():.6f}]")
    
    # Verify U = (R @ V) / sigma relationship (sample check for memory)
    print("\n[VERIFY] Verifying u_i = (R*e_i)/sigma_i relationship (first 10 components)...")
    n_check = min(10, len(sigma))
    U_reconstructed = np.zeros((R.shape[0], n_check))
    for i in range(n_check):
        if sigma[i] > 1e-10:  # Avoid division by zero
            U_reconstructed[:, i] = (R @ V[:, i]) / sigma[i]
    
    reconstruction_error = np.linalg.norm(U[:, :n_check] - U_reconstructed) / np.linalg.norm(U[:, :n_check])
    print(f"        Relative reconstruction error: {reconstruction_error:.2e}")
    
    return U, sigma, Vt, eigenvalues, eigenvectors


def verify_orthogonality(U, Vt):
    """
    2.3 Verify orthogonality: U^T*U = 1 and V^T*V = 1 (identity matrix)
    
    Args:
        U: Left singular vectors matrix.
        Vt: Right singular vectors matrix (transposed).
        
    Returns:
        dict: Orthogonality verification results.
    """
    print("\n[ORTHO] Verifying orthogonality properties...")
    
    results = {}
    
    # Check U^T*U = 1 (identity matrix)
    UtU = U.T @ U
    I_U = np.eye(UtU.shape[0])
    deviation_U = np.linalg.norm(UtU - I_U, 'fro')
    max_deviation_U = np.max(np.abs(UtU - I_U))
    
    print(f"\n        U^T*U = 1 (identity matrix) verification:")
    print(f"        Frobenius norm deviation: {deviation_U:.2e}")
    print(f"        Max element deviation: {max_deviation_U:.2e}")
    
    results['UtU_frobenius_deviation'] = deviation_U
    results['UtU_max_deviation'] = max_deviation_U
    results['UtU_is_identity'] = deviation_U < 1e-10
    
    # Check V^T*V = 1 (identity matrix)
    V = Vt.T
    VtV = V.T @ V
    I_V = np.eye(VtV.shape[0])
    deviation_V = np.linalg.norm(VtV - I_V, 'fro')
    max_deviation_V = np.max(np.abs(VtV - I_V))
    
    print(f"\n        V^T*V = 1 (identity matrix) verification:")
    print(f"        Frobenius norm deviation: {deviation_V:.2e}")
    print(f"        Max element deviation: {max_deviation_V:.2e}")
    
    results['VtV_frobenius_deviation'] = deviation_V
    results['VtV_max_deviation'] = max_deviation_V
    results['VtV_is_identity'] = deviation_V < 1e-10
    
    # Summary
    if results['UtU_is_identity'] and results['VtV_is_identity']:
        print("\n        [OK] Orthogonality verified: U^T*U = 1 and V^T*V = 1")
    else:
        print("\n        [NOTE] Orthogonality has small numerical deviations (expected for floating point)")
    
    return results


def save_svd_results(U, sigma, Vt, eigenvalues, eigenvectors, user_ids, item_ids, ortho_results):
    """
    Save all SVD results to files.
    """
    print("\n[SAVE] Saving SVD results to files...")
    
    # Save singular values
    sigma_df = pd.DataFrame({
        'index': range(1, len(sigma) + 1),
        'singular_value': sigma,
        'eigenvalue': eigenvalues,
        'variance_explained': eigenvalues / eigenvalues.sum() * 100,
        'cumulative_variance': np.cumsum(eigenvalues) / eigenvalues.sum() * 100
    })
    sigma_path = os.path.join(RESULTS_DIR, 'singular_values.csv')
    sigma_df.to_csv(sigma_path, index=False)
    print(f"        Saved: {sigma_path}")
    
    # Save orthogonality results
    ortho_path = os.path.join(RESULTS_DIR, 'orthogonality_verification.txt')
    with open(ortho_path, 'w') as f:
        f.write("SVD Orthogonality Verification Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("U^T*U = 1 (identity matrix) Verification:\n")
        f.write(f"  Frobenius norm deviation: {ortho_results['UtU_frobenius_deviation']:.2e}\n")
        f.write(f"  Max element deviation: {ortho_results['UtU_max_deviation']:.2e}\n")
        f.write(f"  Is identity: {ortho_results['UtU_is_identity']}\n\n")
        f.write("V^T*V = 1 (identity matrix) Verification:\n")
        f.write(f"  Frobenius norm deviation: {ortho_results['VtV_frobenius_deviation']:.2e}\n")
        f.write(f"  Max element deviation: {ortho_results['VtV_max_deviation']:.2e}\n")
        f.write(f"  Is identity: {ortho_results['VtV_is_identity']}\n")
    print(f"        Saved: {ortho_path}")
    
    # Save U matrix (user latent factors) - first 50 components to save space
    n_components_to_save = min(50, U.shape[1])
    U_df = pd.DataFrame(
        U[:, :n_components_to_save],
        index=user_ids,
        columns=[f'component_{i+1}' for i in range(n_components_to_save)]
    )
    U_path = os.path.join(RESULTS_DIR, 'U_matrix.csv')
    U_df.to_csv(U_path)
    print(f"        Saved: {U_path} (first {n_components_to_save} components)")
    
    # Save V matrix (item latent factors) - first 50 components
    V = Vt.T
    V_df = pd.DataFrame(
        V[:, :n_components_to_save],
        index=item_ids,
        columns=[f'component_{i+1}' for i in range(n_components_to_save)]
    )
    V_path = os.path.join(RESULTS_DIR, 'V_matrix.csv')
    V_df.to_csv(V_path)
    print(f"        Saved: {V_path} (first {n_components_to_save} components)")
    
    return sigma_df


# =============================================================================
# 2.4 VISUALIZATION
# =============================================================================

def visualize_singular_values(sigma, eigenvalues):
    """
    2.4 Visualize singular values and variance explained.
    
    Args:
        sigma: Array of singular values.
        eigenvalues: Array of eigenvalues (sigma^2).
    """
    print("\n[PLOT] Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Singular values in descending order
    ax1 = axes[0, 0]
    ax1.plot(range(1, len(sigma) + 1), sigma, 'b-', linewidth=1.5)
    ax1.set_xlabel('Index', fontsize=11)
    ax1.set_ylabel('Singular Value (sigma)', fontsize=11)
    ax1.set_title('Singular Values in Descending Order', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1, len(sigma)])
    
    # Plot 2: Singular values (log scale)
    ax2 = axes[0, 1]
    ax2.semilogy(range(1, len(sigma) + 1), sigma, 'b-', linewidth=1.5)
    ax2.set_xlabel('Index', fontsize=11)
    ax2.set_ylabel('Singular Value (sigma) - Log Scale', fontsize=11)
    ax2.set_title('Singular Values (Logarithmic Scale)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1, len(sigma)])
    
    # Plot 3: Scree plot - variance explained by each singular value
    variance_explained = eigenvalues / eigenvalues.sum() * 100
    ax3 = axes[1, 0]
    ax3.bar(range(1, min(51, len(variance_explained) + 1)), 
            variance_explained[:50], 
            color='steelblue', alpha=0.7)
    ax3.set_xlabel('Component', fontsize=11)
    ax3.set_ylabel('Variance Explained (%)', fontsize=11)
    ax3.set_title('Scree Plot - Variance Explained by Each Component', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Cumulative variance explained
    cumulative_variance = np.cumsum(variance_explained)
    ax4 = axes[1, 1]
    ax4.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'g-', linewidth=2)
    ax4.axhline(y=90, color='r', linestyle='--', label='90% variance')
    ax4.axhline(y=95, color='orange', linestyle='--', label='95% variance')
    ax4.axhline(y=99, color='purple', linestyle='--', label='99% variance')
    
    # Find components needed for different variance thresholds
    n_90 = np.argmax(cumulative_variance >= 90) + 1
    n_95 = np.argmax(cumulative_variance >= 95) + 1
    n_99 = np.argmax(cumulative_variance >= 99) + 1
    
    ax4.axvline(x=n_90, color='r', linestyle=':', alpha=0.5)
    ax4.axvline(x=n_95, color='orange', linestyle=':', alpha=0.5)
    ax4.axvline(x=n_99, color='purple', linestyle=':', alpha=0.5)
    
    ax4.set_xlabel('Number of Components', fontsize=11)
    ax4.set_ylabel('Cumulative Variance Explained (%)', fontsize=11)
    ax4.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([1, len(cumulative_variance)])
    ax4.set_ylim([0, 105])
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(PLOTS_DIR, 'svd_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"        Saved: {plot_path}")
    plt.close()
    
    # Print variance thresholds
    print(f"\n        Components for 90% variance: {n_90}")
    print(f"        Components for 95% variance: {n_95}")
    print(f"        Components for 99% variance: {n_99}")
    
    return n_90, n_95, n_99


def print_summary(sigma, eigenvalues, ortho_results, n_90, n_95, n_99):
    """
    Print final summary of SVD analysis.
    """
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    variance_explained = eigenvalues / eigenvalues.sum() * 100
    
    print(f"\n  Total singular values: {len(sigma)}")
    print(f"  Largest singular value: {sigma[0]:.4f}")
    print(f"  Smallest singular value: {sigma[-1]:.6f}")
    print(f"  Condition number: {sigma[0] / sigma[-1]:.2e}")
    
    print(f"\n  Top 5 singular values:")
    for i in range(min(5, len(sigma))):
        print(f"    sigma_{i+1} = {sigma[i]:.4f} (variance: {variance_explained[i]:.2f}%)")
    
    print(f"\n  Dimensionality reduction:")
    print(f"    90% variance with {n_90} components ({n_90/len(sigma)*100:.1f}% of original)")
    print(f"    95% variance with {n_95} components ({n_95/len(sigma)*100:.1f}% of original)")
    print(f"    99% variance with {n_99} components ({n_99/len(sigma)*100:.1f}% of original)")
    
    print(f"\n  Orthogonality verification:")
    print(f"    U^T*U = 1: {'[OK] Verified' if ortho_results['UtU_is_identity'] else '[NOTE] Small deviation'}")
    print(f"    V^T*V = 1: {'[OK] Verified' if ortho_results['VtV_is_identity'] else '[NOTE] Small deviation'}")
    
    print("\n" + "=" * 60)
    print("[DONE] Full SVD Analysis Complete!")
    print("=" * 60)


# =============================================================================
# 3. TRUNCATED SVD (LOW-RANK APPROXIMATION)
# =============================================================================

def compute_truncated_svd(U, sigma, Vt, k_values=[5, 20, 50, 100]):
    """
    3.1-3.2 Compute truncated SVD for different k values.
    
    For each k:
    - Construct U_k (first k columns of U)
    - Construct Sigma_k (top-left k x k submatrix)
    - Construct V_k (first k columns of V)
    - Compute approximation: R_hat_k = U_k @ Sigma_k @ V_k^T
    
    Args:
        U: Left singular vectors matrix (m x r)
        sigma: Singular values array (r,)
        Vt: Right singular vectors matrix transposed (r x n)
        k_values: List of k values for truncation
        
    Returns:
        dict: Dictionary with k as keys and approximated matrices as values
    """
    print("\n" + "=" * 60)
    print("3. TRUNCATED SVD (LOW-RANK APPROXIMATION)")
    print("=" * 60)
    
    V = Vt.T  # V matrix (n x r)
    approximations = {}
    
    print(f"\n[TRUNCATE] Computing truncated SVD for k = {k_values}...")
    
    for k in k_values:
        if k > len(sigma):
            print(f"        [WARNING] k={k} exceeds available singular values ({len(sigma)}), skipping...")
            continue
            
        print(f"\n        k = {k}:")
        
        # 3.2 Construct truncated matrices
        U_k = U[:, :k]           # First k columns of U (m x k)
        sigma_k = sigma[:k]       # First k singular values
        V_k = V[:, :k]           # First k columns of V (n x k)
        
        print(f"          U_k shape: {U_k.shape}")
        print(f"          Sigma_k shape: {sigma_k.shape}")
        print(f"          V_k shape: {V_k.shape}")
        
        # Compute approximation: R_hat_k = U_k @ diag(sigma_k) @ V_k^T
        # More memory efficient: (U_k * sigma_k) @ V_k^T
        R_hat_k = (U_k * sigma_k) @ V_k.T
        
        print(f"          R_hat_k shape: {R_hat_k.shape}")
        
        approximations[k] = {
            'R_hat': R_hat_k,
            'U_k': U_k,
            'sigma_k': sigma_k,
            'V_k': V_k
        }
    
    print(f"\n        [OK] Computed {len(approximations)} approximations")
    
    return approximations


def calculate_reconstruction_error(filled_matrix, approximations, eigenvalues):
    """
    3.3 Calculate reconstruction error for each k value.
    
    Metrics:
    - MAE: Mean Absolute Error on all ratings
    - RMSE: Root Mean Square Error on all ratings
    - Variance retained: Cumulative variance explained by k components
    
    Args:
        filled_matrix: Original filled ratings matrix (m x n)
        approximations: Dictionary of approximated matrices from compute_truncated_svd
        eigenvalues: Array of eigenvalues for variance calculation
        
    Returns:
        pd.DataFrame: Error metrics for each k value
    """
    print("\n[ERROR] Calculating reconstruction errors...")
    
    results = []
    total_variance = eigenvalues.sum()
    
    for k, approx_data in sorted(approximations.items()):
        R_hat_k = approx_data['R_hat']
        
        # Calculate error metrics
        error = filled_matrix - R_hat_k
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error ** 2))
        
        # Calculate variance retained
        variance_retained = eigenvalues[:k].sum() / total_variance * 100
        
        # Frobenius norm of error (for reference)
        frobenius_error = np.linalg.norm(error, 'fro')
        frobenius_original = np.linalg.norm(filled_matrix, 'fro')
        relative_error = frobenius_error / frobenius_original * 100
        
        results.append({
            'k': k,
            'MAE': mae,
            'RMSE': rmse,
            'Variance_Retained_%': variance_retained,
            'Frobenius_Error': frobenius_error,
            'Relative_Error_%': relative_error
        })
        
        print(f"        k={k:3d}: MAE={mae:.4f}, RMSE={rmse:.4f}, Variance={variance_retained:.2f}%")
    
    error_df = pd.DataFrame(results)
    
    # Save to CSV
    error_path = os.path.join(RESULTS_DIR, 'reconstruction_errors.csv')
    error_df.to_csv(error_path, index=False)
    print(f"\n        [SAVED] {error_path}")
    
    return error_df


def identify_optimal_k(error_df, eigenvalues):
    """
    3.4 Identify optimal k using elbow method.
    
    The elbow point is where the rate of error decrease slows significantly.
    
    Args:
        error_df: DataFrame with reconstruction errors for each k
        eigenvalues: Array of eigenvalues
        
    Returns:
        int: Optimal k value
    """
    print("\n[ELBOW] Identifying optimal k using elbow method...")
    
    k_values = error_df['k'].values
    rmse_values = error_df['RMSE'].values
    
    # Calculate rate of change (derivative)
    if len(k_values) > 1:
        # Calculate improvement per unit k
        improvements = []
        for i in range(1, len(k_values)):
            delta_rmse = rmse_values[i-1] - rmse_values[i]
            delta_k = k_values[i] - k_values[i-1]
            improvement_rate = delta_rmse / delta_k
            improvements.append({
                'from_k': k_values[i-1],
                'to_k': k_values[i],
                'RMSE_improvement': delta_rmse,
                'improvement_rate': improvement_rate
            })
        
        improvements_df = pd.DataFrame(improvements)
        print("\n        Error improvement analysis:")
        for _, row in improvements_df.iterrows():
            print(f"          k: {int(row['from_k'])} -> {int(row['to_k'])}: "
                  f"RMSE improvement = {row['RMSE_improvement']:.4f} "
                  f"(rate: {row['improvement_rate']:.6f}/k)")
    
    # Use variance threshold method: find k where variance >= 90%
    variance_90_idx = error_df[error_df['Variance_Retained_%'] >= 90].index
    if len(variance_90_idx) > 0:
        optimal_k_variance = error_df.loc[variance_90_idx[0], 'k']
    else:
        optimal_k_variance = k_values[-1]
    
    # Simple elbow detection: largest relative improvement
    if len(k_values) > 1:
        relative_improvements = []
        for i in range(1, len(k_values)):
            rel_imp = (rmse_values[i-1] - rmse_values[i]) / rmse_values[i-1]
            relative_improvements.append(rel_imp)
        
        # The elbow is typically at the point after which improvements are small
        # Choose the k where we still have significant improvement
        threshold = 0.05  # 5% relative improvement threshold
        optimal_k_elbow = k_values[0]
        for i, rel_imp in enumerate(relative_improvements):
            if rel_imp > threshold:
                optimal_k_elbow = k_values[i + 1]
    else:
        optimal_k_elbow = k_values[0]
    
    # Use the larger of the two methods (more conservative)
    optimal_k = max(optimal_k_variance, optimal_k_elbow)
    
    print(f"\n        Optimal k (90% variance threshold): {optimal_k_variance}")
    print(f"        Optimal k (elbow method): {optimal_k_elbow}")
    print(f"        [RESULT] Selected optimal k: {optimal_k}")
    
    return int(optimal_k)


def visualize_truncated_svd(error_df, eigenvalues, optimal_k):
    """
    3.4 Create visualizations for truncated SVD analysis.
    
    Plots:
    1. Reconstruction error vs k (elbow curve)
    2. Variance retained vs k
    3. Error reduction between k values
    4. Combined MAE/RMSE comparison
    
    Args:
        error_df: DataFrame with reconstruction errors
        eigenvalues: Array of eigenvalues
        optimal_k: Identified optimal k value
    """
    print("\n[PLOT] Creating truncated SVD visualizations...")
    
    k_values = error_df['k'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Reconstruction Error (MAE & RMSE) vs k - Elbow Curve
    ax1 = axes[0, 0]
    ax1.plot(k_values, error_df['MAE'], 'b-o', linewidth=2, markersize=8, label='MAE')
    ax1.plot(k_values, error_df['RMSE'], 'r-s', linewidth=2, markersize=8, label='RMSE')
    ax1.axvline(x=optimal_k, color='green', linestyle='--', linewidth=2, 
                label=f'Optimal k={optimal_k}')
    ax1.set_xlabel('Number of Latent Factors (k)', fontsize=11)
    ax1.set_ylabel('Reconstruction Error', fontsize=11)
    ax1.set_title('Reconstruction Error vs k (Elbow Curve)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    
    # Plot 2: Variance Retained vs k
    ax2 = axes[0, 1]
    ax2.plot(k_values, error_df['Variance_Retained_%'], 'g-o', linewidth=2, markersize=8)
    ax2.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='90% threshold')
    ax2.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
    ax2.axvline(x=optimal_k, color='green', linestyle='--', linewidth=2, 
                label=f'Optimal k={optimal_k}')
    ax2.fill_between(k_values, 0, error_df['Variance_Retained_%'], alpha=0.3, color='green')
    ax2.set_xlabel('Number of Latent Factors (k)', fontsize=11)
    ax2.set_ylabel('Variance Retained (%)', fontsize=11)
    ax2.set_title('Cumulative Variance Retained vs k', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)
    ax2.set_ylim([0, 105])
    
    # Plot 3: Bar chart of error for each k
    ax3 = axes[1, 0]
    x_pos = np.arange(len(k_values))
    width = 0.35
    bars1 = ax3.bar(x_pos - width/2, error_df['MAE'], width, label='MAE', color='steelblue', alpha=0.8)
    bars2 = ax3.bar(x_pos + width/2, error_df['RMSE'], width, label='RMSE', color='coral', alpha=0.8)
    
    # Highlight optimal k
    opt_idx = np.where(k_values == optimal_k)[0]
    if len(opt_idx) > 0:
        bars1[opt_idx[0]].set_edgecolor('green')
        bars1[opt_idx[0]].set_linewidth(3)
        bars2[opt_idx[0]].set_edgecolor('green')
        bars2[opt_idx[0]].set_linewidth(3)
    
    ax3.set_xlabel('Number of Latent Factors (k)', fontsize=11)
    ax3.set_ylabel('Error Value', fontsize=11)
    ax3.set_title('MAE and RMSE Comparison by k', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(k_values)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Relative Error (%) vs k
    ax4 = axes[1, 1]
    ax4.plot(k_values, error_df['Relative_Error_%'], 'purple', linewidth=2, marker='D', markersize=8)
    ax4.axvline(x=optimal_k, color='green', linestyle='--', linewidth=2, 
                label=f'Optimal k={optimal_k}')
    ax4.set_xlabel('Number of Latent Factors (k)', fontsize=11)
    ax4.set_ylabel('Relative Frobenius Error (%)', fontsize=11)
    ax4.set_title('Relative Reconstruction Error vs k', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(k_values)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(PLOTS_DIR, 'truncated_svd_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"        [SAVED] {plot_path}")
    plt.close()
    
    return plot_path


# =============================================================================
# 4. RATING PREDICTION WITH TRUNCATED SVD
# =============================================================================

def predict_rating_svd(U_k, sigma_k, V_k, user_idx, item_idx):
    """
    4.1 Predict a single rating using truncated SVD.
    
    Formula: r_hat_ui = u_k^T @ diag(sigma_k) @ v_k
    
    Args:
        U_k: Truncated user latent factors (m x k)
        sigma_k: Truncated singular values (k,)
        V_k: Truncated item latent factors (n x k)
        user_idx: Index of user in matrix
        item_idx: Index of item in matrix
        
    Returns:
        float: Predicted rating
    """
    # Extract user's latent factor representation
    u = U_k[user_idx, :]  # (k,)
    
    # Extract item's latent factor representation
    v = V_k[item_idx, :]  # (k,)
    
    # Compute predicted rating: r_hat = u^T @ diag(sigma) @ v = sum(u * sigma * v)
    r_hat = np.dot(u * sigma_k, v)
    
    return r_hat


def predict_missing_ratings(U, sigma, Vt, k, user_ids, item_ids, original_ratings_matrix=None):
    """
    4.2 Predict missing ratings for target users and items.
    
    Args:
        U: Full U matrix from SVD
        sigma: Full singular values array
        Vt: Full Vt matrix from SVD
        k: Number of latent factors to use
        user_ids: Array of user IDs in the matrix
        item_ids: Array of item IDs in the matrix
        original_ratings_matrix: Original sparse ratings matrix (before filling) for ground truth
        
    Returns:
        pd.DataFrame: Predictions with columns [User, Item, Predicted_Rating, Ground_Truth, Is_Missing]
    """
    print("\n" + "=" * 60)
    print("4. RATING PREDICTION WITH TRUNCATED SVD")
    print("=" * 60)
    
    # Load target users and items
    try:
        target_users = get_target_users()
        target_items = get_target_items()
        print(f"\n[LOAD] Target users: {target_users}")
        print(f"        Target items: {target_items}")
    except FileNotFoundError as e:
        print(f"\n[WARNING] {e}")
        print("        Using first 3 users and first 2 items as targets...")
        target_users = list(user_ids[:3])
        target_items = list(item_ids[:2])
    
    # Prepare truncated matrices
    V = Vt.T
    U_k = U[:, :k]
    sigma_k = sigma[:k]
    V_k = V[:, :k]
    
    print(f"\n[PREDICT] Using k={k} latent factors for prediction...")
    print(f"          U_k shape: {U_k.shape}")
    print(f"          V_k shape: {V_k.shape}")
    
    # Create mappings
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    
    predictions = []
    
    print("\n        Predictions:")
    print("        " + "-" * 60)
    
    for user_id in target_users:
        for item_id in target_items:
            # Check if user and item exist in our matrix
            if user_id not in user_to_idx:
                print(f"          [SKIP] User {user_id} not in matrix")
                continue
            if item_id not in item_to_idx:
                print(f"          [SKIP] Item {item_id} not in matrix")
                continue
            
            user_idx = user_to_idx[user_id]
            item_idx = item_to_idx[item_id]
            
            # Predict rating
            predicted_rating = predict_rating_svd(U_k, sigma_k, V_k, user_idx, item_idx)
            
            # Clip to valid rating range [1, 5]
            predicted_rating_clipped = np.clip(predicted_rating, 1.0, 5.0)
            
            # Check ground truth if original matrix available
            ground_truth = None
            is_missing = True
            if original_ratings_matrix is not None:
                gt_value = original_ratings_matrix.iloc[user_idx, item_idx]
                if not pd.isna(gt_value):
                    ground_truth = gt_value
                    is_missing = False
            
            predictions.append({
                'User_ID': user_id,
                'Item_ID': item_id,
                'Predicted_Rating': predicted_rating_clipped,
                'Raw_Prediction': predicted_rating,
                'Ground_Truth': ground_truth,
                'Is_Missing': is_missing
            })
            
            gt_str = f"{ground_truth:.2f}" if ground_truth is not None else "N/A"
            status = "MISSING" if is_missing else "EXISTS"
            print(f"          User {user_id:6d} x Item {item_id:5d}: "
                  f"Predicted={predicted_rating_clipped:.3f} "
                  f"(raw={predicted_rating:.3f}), GT={gt_str} [{status}]")
    
    print("        " + "-" * 60)
    
    predictions_df = pd.DataFrame(predictions)
    
    return predictions_df


def calculate_prediction_accuracy(predictions_df):
    """
    4.4 Calculate prediction accuracy for predictions with ground truth.
    
    Args:
        predictions_df: DataFrame with predictions and ground truth
        
    Returns:
        dict: Accuracy metrics (MAE, RMSE) or None if no ground truth available
    """
    print("\n[ACCURACY] Calculating prediction accuracy...")
    
    # Filter to predictions with ground truth
    valid_predictions = predictions_df[predictions_df['Ground_Truth'].notna()]
    
    if len(valid_predictions) == 0:
        print("        [NOTE] No ground truth available for accuracy calculation")
        return None
    
    y_true = valid_predictions['Ground_Truth'].values
    y_pred = valid_predictions['Predicted_Rating'].values
    
    # Calculate metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Calculate baseline (item average prediction)
    baseline_mae = np.mean(np.abs(y_true - y_true.mean()))
    
    accuracy = {
        'n_predictions': len(valid_predictions),
        'MAE': mae,
        'RMSE': rmse,
        'baseline_MAE': baseline_mae,
        'improvement_over_baseline': baseline_mae - mae
    }
    
    print(f"        Predictions with ground truth: {accuracy['n_predictions']}")
    print(f"        MAE: {mae:.4f}")
    print(f"        RMSE: {rmse:.4f}")
    print(f"        Baseline MAE (mean): {baseline_mae:.4f}")
    print(f"        Improvement over baseline: {accuracy['improvement_over_baseline']:.4f}")
    
    return accuracy


def save_prediction_results(predictions_df, accuracy, optimal_k):
    """
    4.3 & 4.5 Save prediction results to files.
    
    Args:
        predictions_df: DataFrame with all predictions
        accuracy: Dictionary with accuracy metrics (or None)
        optimal_k: The k value used for predictions
    """
    print("\n[SAVE] Saving prediction results...")
    
    # Save predictions table
    predictions_path = os.path.join(TABLES_DIR, 'svd_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"        [SAVED] {predictions_path}")
    
    # Save accuracy summary
    accuracy_path = os.path.join(RESULTS_DIR, 'prediction_accuracy.txt')
    with open(accuracy_path, 'w') as f:
        f.write("SVD Rating Prediction Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Optimal k (latent factors): {optimal_k}\n\n")
        
        f.write("Predictions Summary:\n")
        f.write(f"  Total predictions: {len(predictions_df)}\n")
        f.write(f"  Missing ratings predicted: {predictions_df['Is_Missing'].sum()}\n")
        f.write(f"  Existing ratings (ground truth): {(~predictions_df['Is_Missing']).sum()}\n\n")
        
        if accuracy is not None:
            f.write("Accuracy Metrics (on existing ratings):\n")
            f.write(f"  MAE: {accuracy['MAE']:.4f}\n")
            f.write(f"  RMSE: {accuracy['RMSE']:.4f}\n")
            f.write(f"  Baseline MAE: {accuracy['baseline_MAE']:.4f}\n")
            f.write(f"  Improvement: {accuracy['improvement_over_baseline']:.4f}\n")
        else:
            f.write("Accuracy Metrics:\n")
            f.write("  No ground truth available for accuracy calculation.\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Prediction Details:\n\n")
        
        for _, row in predictions_df.iterrows():
            gt_str = f"{row['Ground_Truth']:.2f}" if pd.notna(row['Ground_Truth']) else "N/A"
            f.write(f"  User {int(row['User_ID']):6d} x Item {int(row['Item_ID']):5d}: "
                   f"Predicted={row['Predicted_Rating']:.3f}, Ground Truth={gt_str}\n")
    
    print(f"        [SAVED] {accuracy_path}")
    
    # Create a formatted markdown table for easy viewing
    table_md_path = os.path.join(TABLES_DIR, 'svd_predictions.md')
    with open(table_md_path, 'w') as f:
        f.write("# SVD Rating Predictions\n\n")
        f.write(f"**Optimal k (latent factors):** {optimal_k}\n\n")
        f.write("## Prediction Results\n\n")
        f.write("| User ID | Item ID | Predicted Rating | Ground Truth | Status |\n")
        f.write("|---------|---------|-----------------|--------------|--------|\n")
        
        for _, row in predictions_df.iterrows():
            gt_str = f"{row['Ground_Truth']:.2f}" if pd.notna(row['Ground_Truth']) else "N/A"
            status = "Missing" if row['Is_Missing'] else "Exists"
            f.write(f"| {int(row['User_ID'])} | {int(row['Item_ID'])} | "
                   f"{row['Predicted_Rating']:.3f} | {gt_str} | {status} |\n")
        
        if accuracy is not None:
            f.write(f"\n## Accuracy Metrics\n\n")
            f.write(f"- **MAE:** {accuracy['MAE']:.4f}\n")
            f.write(f"- **RMSE:** {accuracy['RMSE']:.4f}\n")
    
    print(f"        [SAVED] {table_md_path}")


def print_truncated_svd_summary(error_df, optimal_k, predictions_df, accuracy):
    """
    Print summary for Truncated SVD and Rating Prediction sections.
    """
    print("\n" + "=" * 60)
    print("TRUNCATED SVD & PREDICTION SUMMARY")
    print("=" * 60)
    
    print(f"\n  Truncated SVD Analysis:")
    print(f"    k values tested: {error_df['k'].tolist()}")
    print(f"    Optimal k selected: {optimal_k}")
    
    opt_row = error_df[error_df['k'] == optimal_k].iloc[0]
    print(f"    At k={optimal_k}:")
    print(f"      - MAE: {opt_row['MAE']:.4f}")
    print(f"      - RMSE: {opt_row['RMSE']:.4f}")
    print(f"      - Variance retained: {opt_row['Variance_Retained_%']:.2f}%")
    
    print(f"\n  Rating Predictions:")
    print(f"    Total predictions: {len(predictions_df)}")
    print(f"    Missing ratings predicted: {predictions_df['Is_Missing'].sum()}")
    
    if accuracy is not None:
        print(f"\n  Prediction Accuracy:")
        print(f"    MAE: {accuracy['MAE']:.4f}")
        print(f"    RMSE: {accuracy['RMSE']:.4f}")
    
    print("\n" + "=" * 60)
    print("[DONE] Truncated SVD & Rating Prediction Complete!")
    print("=" * 60)


# =============================================================================
# 6. LATENT FACTOR INTERPRETATION
# =============================================================================

def analyze_latent_factors(U, V, sigma, user_ids, item_ids, top_n_factors=3, top_k=10):
    """
    6.1-6.2 Analyze the top latent factors (largest singular values).
    
    For each factor:
    - Identify items with highest absolute values in V
    - Identify users with highest absolute values in U
    - Attempt to interpret semantic meaning
    
    Args:
        U: User latent factors matrix (m x r)
        V: Item latent factors matrix (n x r) - Note: V = Vt.T
        sigma: Singular values array
        user_ids: Array of user IDs
        item_ids: Array of item IDs
        top_n_factors: Number of top factors to analyze
        top_k: Number of top users/items to show per factor
        
    Returns:
        dict: Analysis results for each factor
    """
    print("\n" + "=" * 60)
    print("6. LATENT FACTOR INTERPRETATION")
    print("=" * 60)
    
    results = {}
    
    print(f"\n[ANALYZE] Analyzing top-{top_n_factors} latent factors...")
    print(f"          Each factor will show top-{top_k} users and items")
    
    for factor_idx in range(min(top_n_factors, len(sigma))):
        print(f"\n" + "-" * 60)
        print(f"  FACTOR {factor_idx + 1} (σ = {sigma[factor_idx]:.4f}, variance = {(sigma[factor_idx]**2 / (sigma**2).sum() * 100):.4f}%)")
        print("-" * 60)
        
        factor_results = {
            'singular_value': sigma[factor_idx],
            'variance_pct': sigma[factor_idx]**2 / (sigma**2).sum() * 100
        }
        
        # Get item loadings for this factor
        item_loadings = V[:, factor_idx]
        top_item_indices = np.argsort(np.abs(item_loadings))[::-1][:top_k]
        
        print(f"\n  Top-{top_k} Items (highest |V| values):")
        factor_results['top_items'] = []
        for rank, idx in enumerate(top_item_indices):
            item_id = item_ids[idx]
            loading = item_loadings[idx]
            sign = "+" if loading > 0 else "-"
            print(f"    {rank+1:2d}. Item {item_id:6d}: {sign}{abs(loading):.4f}")
            factor_results['top_items'].append({
                'item_id': item_id,
                'loading': loading
            })
        
        # Get user loadings for this factor
        # For full SVD, U is m x m, so we need first r columns
        user_loadings = U[:, factor_idx] if factor_idx < U.shape[1] else np.zeros(U.shape[0])
        top_user_indices = np.argsort(np.abs(user_loadings))[::-1][:top_k]
        
        print(f"\n  Top-{top_k} Users (highest |U| values):")
        factor_results['top_users'] = []
        for rank, idx in enumerate(top_user_indices):
            user_id = user_ids[idx]
            loading = user_loadings[idx]
            sign = "+" if loading > 0 else "-"
            print(f"    {rank+1:2d}. User {user_id:6d}: {sign}{abs(loading):.4f}")
            factor_results['top_users'].append({
                'user_id': user_id,
                'loading': loading
            })
        
        # Interpretation hint
        if factor_idx == 0:
            print(f"\n  [INTERPRET] Factor 1 likely represents the GLOBAL MEAN effect")
            print(f"              (captures overall rating tendency - most variance)")
            factor_results['interpretation'] = "Global mean effect / baseline rating tendency"
        elif factor_idx == 1:
            print(f"\n  [INTERPRET] Factor 2 may represent a major GENRE or PREFERENCE dimension")
            print(f"              (e.g., action vs drama, mainstream vs niche)")
            factor_results['interpretation'] = "Major genre/preference dimension"
        else:
            print(f"\n  [INTERPRET] Factor {factor_idx+1} captures finer preference distinctions")
            factor_results['interpretation'] = f"Fine-grained preference dimension {factor_idx+1}"
        
        results[f'factor_{factor_idx + 1}'] = factor_results
    
    # Save analysis to file
    analysis_path = os.path.join(RESULTS_DIR, 'latent_factor_analysis.txt')
    with open(analysis_path, 'w') as f:
        f.write("Latent Factor Interpretation Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        for factor_idx in range(min(top_n_factors, len(sigma))):
            factor_key = f'factor_{factor_idx + 1}'
            factor_data = results[factor_key]
            
            f.write(f"FACTOR {factor_idx + 1}\n")
            f.write(f"  Singular value: {factor_data['singular_value']:.4f}\n")
            f.write(f"  Variance explained: {factor_data['variance_pct']:.4f}%\n")
            f.write(f"  Interpretation: {factor_data['interpretation']}\n\n")
            
            f.write(f"  Top Items:\n")
            for item in factor_data['top_items']:
                f.write(f"    Item {item['item_id']}: {item['loading']:.4f}\n")
            
            f.write(f"\n  Top Users:\n")
            for user in factor_data['top_users']:
                f.write(f"    User {user['user_id']}: {user['loading']:.4f}\n")
            
            f.write("\n" + "-" * 50 + "\n\n")
    
    print(f"\n        [SAVED] {analysis_path}")
    
    return results


def visualize_latent_space(U, V, sigma, user_ids, item_ids, ratings_matrix):
    """
    6.3 Visualize latent space by projecting users and items onto first 2 factors.
    
    Creates:
    - Users projected onto factors 1 & 2
    - Items projected onto factors 1 & 2
    - Combined user-item scatter (sampled)
    - Color-coded by activity level
    
    Args:
        U: User latent factors matrix
        V: Item latent factors matrix (V = Vt.T)
        sigma: Singular values array
        user_ids: Array of user IDs
        item_ids: Array of item IDs
        ratings_matrix: Original ratings matrix for activity calculation
    """
    print("\n[PLOT] Creating latent space visualizations...")
    
    # Get first 2 latent factors
    # Scale by singular values for proper projection
    U_proj = U[:, :2] * sigma[:2]  # Users in 2D latent space
    V_proj = V[:, :2] * sigma[:2]  # Items in 2D latent space
    
    # Calculate activity levels
    user_activity = (~ratings_matrix.isna()).sum(axis=1).values  # Number of ratings per user
    item_popularity = (~ratings_matrix.isna()).sum(axis=0).values  # Number of ratings per item
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Users in latent space
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(U_proj[:, 0], U_proj[:, 1], 
                           c=user_activity, cmap='viridis', 
                           alpha=0.5, s=10)
    ax1.set_xlabel('Latent Factor 1', fontsize=11)
    ax1.set_ylabel('Latent Factor 2', fontsize=11)
    ax1.set_title('Users Projected onto First 2 Latent Factors', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('User Activity (# ratings)')
    
    # Plot 2: Items in latent space
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(V_proj[:, 0], V_proj[:, 1], 
                           c=item_popularity, cmap='plasma', 
                           alpha=0.5, s=10)
    ax2.set_xlabel('Latent Factor 1', fontsize=11)
    ax2.set_ylabel('Latent Factor 2', fontsize=11)
    ax2.set_title('Items Projected onto First 2 Latent Factors', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Item Popularity (# ratings)')
    
    # Plot 3: Combined (sampled for visibility)
    ax3 = axes[1, 0]
    # Sample for visibility
    n_sample = min(500, len(U_proj), len(V_proj))
    user_sample_idx = np.random.choice(len(U_proj), n_sample, replace=False)
    item_sample_idx = np.random.choice(len(V_proj), n_sample, replace=False)
    
    ax3.scatter(U_proj[user_sample_idx, 0], U_proj[user_sample_idx, 1], 
                c='blue', alpha=0.3, s=20, label='Users')
    ax3.scatter(V_proj[item_sample_idx, 0], V_proj[item_sample_idx, 1], 
                c='red', alpha=0.3, s=20, label='Items')
    ax3.set_xlabel('Latent Factor 1', fontsize=11)
    ax3.set_ylabel('Latent Factor 2', fontsize=11)
    ax3.set_title('Users and Items in Latent Space (Sampled)', fontsize=12, fontweight='bold')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax3.legend()
    
    # Plot 4: Distribution of latent factor values
    ax4 = axes[1, 1]
    ax4.hist(U_proj[:, 0], bins=50, alpha=0.5, label='Users - Factor 1', color='blue')
    ax4.hist(V_proj[:, 0], bins=50, alpha=0.5, label='Items - Factor 1', color='red')
    ax4.set_xlabel('Latent Factor 1 Value', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Distribution of Factor 1 Values', fontsize=12, fontweight='bold')
    ax4.legend()
    
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_DIR, 'latent_space_visualization.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"        [SAVED] {plot_path}")
    plt.close()
    
    return plot_path


# =============================================================================
# 7. SENSITIVITY ANALYSIS
# =============================================================================

def test_missing_data_robustness(ratings_matrix, item_averages, U, sigma, Vt, 
                                  missing_percentages=[10, 30, 50, 70]):
    """
    7.1 Test robustness to missing data.
    
    For each percentage:
    - Start with current filled matrix
    - Randomly mask additional ratings
    - Perform SVD and measure reconstruction error
    
    Args:
        ratings_matrix: Original sparse ratings matrix (before filling)
        item_averages: Item average ratings for filling
        U, sigma, Vt: Original SVD components
        missing_percentages: List of missing percentages to test
        
    Returns:
        pd.DataFrame: Results with error metrics per percentage
    """
    print("\n" + "=" * 60)
    print("7. SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    print("\n[7.1] Testing robustness to missing data...")
    print(f"      Percentages to test: {missing_percentages}%")
    
    results = []
    
    # Get current non-missing entries
    original_mask = ~ratings_matrix.isna()
    n_original_ratings = original_mask.sum().sum()
    
    print(f"      Original observed ratings: {n_original_ratings:,}")
    
    for pct in missing_percentages:
        print(f"\n      Testing {pct}% additional missing...")
        
        # Create copy of original matrix
        test_matrix = ratings_matrix.copy()
        
        # Get indices of observed ratings
        observed_indices = np.argwhere(original_mask.values)
        n_observed = len(observed_indices)
        
        # Randomly mask additional ratings
        n_to_mask = int(n_observed * pct / 100)
        mask_indices = np.random.choice(n_observed, n_to_mask, replace=False)
        
        for idx in mask_indices:
            i, j = observed_indices[idx]
            test_matrix.iloc[i, j] = np.nan
        
        # Calculate new sparsity
        new_missing = test_matrix.isna().sum().sum()
        total_cells = test_matrix.shape[0] * test_matrix.shape[1]
        new_sparsity = new_missing / total_cells * 100
        
        # Apply mean filling
        filled_test = test_matrix.values.astype(np.float32)
        item_avg_array = item_averages.values.astype(np.float32)
        
        for j in range(filled_test.shape[1]):
            col_mask = np.isnan(filled_test[:, j])
            filled_test[col_mask, j] = item_avg_array[j]
        
        # Handle any remaining NaN
        global_avg = np.nanmean(item_avg_array)
        filled_test = np.nan_to_num(filled_test, nan=global_avg)
        
        # Perform SVD (use reduced for speed)
        U_test, sigma_test, Vt_test = np.linalg.svd(filled_test, full_matrices=False)
        
        # Calculate reconstruction with k=100
        k = min(100, len(sigma_test))
        V_test = Vt_test.T
        R_hat = (U_test[:, :k] * sigma_test[:k]) @ V_test[:, :k].T
        
        # Calculate error vs original filled matrix
        error = filled_test - R_hat
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error ** 2))
        
        results.append({
            'missing_pct': pct,
            'n_masked': n_to_mask,
            'sparsity_pct': new_sparsity,
            'MAE': mae,
            'RMSE': rmse
        })
        
        print(f"        Masked {n_to_mask:,} ratings, Sparsity: {new_sparsity:.2f}%")
        print(f"        Reconstruction MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    robustness_df = pd.DataFrame(results)
    
    # Save results
    robustness_path = os.path.join(RESULTS_DIR, 'sensitivity_robustness.csv')
    robustness_df.to_csv(robustness_path, index=False)
    print(f"\n        [SAVED] {robustness_path}")
    
    return robustness_df


def compare_filling_strategies(ratings_matrix, user_ids, item_ids):
    """
    7.2 Compare different mean-filling strategies.
    
    Strategies:
    - Item mean: Fill missing with item's average rating
    - User mean: Fill missing with user's average rating
    
    Args:
        ratings_matrix: Original sparse ratings matrix
        user_ids: Array of user IDs
        item_ids: Array of item IDs
        
    Returns:
        dict: Comparison results
    """
    print("\n[7.2] Comparing filling strategies...")
    
    results = {}
    
    # Strategy 1: Item mean (current approach)
    print("\n      Strategy 1: Item Mean Filling")
    item_averages = ratings_matrix.mean(axis=0, skipna=True)
    filled_item = ratings_matrix.values.astype(np.float32).copy()
    
    for j in range(filled_item.shape[1]):
        col_mask = np.isnan(filled_item[:, j])
        filled_item[col_mask, j] = item_averages.iloc[j] if not np.isnan(item_averages.iloc[j]) else 3.0
    
    filled_item = np.nan_to_num(filled_item, nan=3.0)
    
    # SVD with item mean
    U_item, sigma_item, Vt_item = np.linalg.svd(filled_item, full_matrices=False)
    k = min(100, len(sigma_item))
    V_item = Vt_item.T
    R_hat_item = (U_item[:, :k] * sigma_item[:k]) @ V_item[:, :k].T
    
    mae_item = np.mean(np.abs(filled_item - R_hat_item))
    rmse_item = np.sqrt(np.mean((filled_item - R_hat_item) ** 2))
    
    results['item_mean'] = {
        'MAE': mae_item,
        'RMSE': rmse_item,
        'top_singular_values': sigma_item[:5].tolist()
    }
    print(f"        Reconstruction MAE: {mae_item:.4f}, RMSE: {rmse_item:.4f}")
    
    # Strategy 2: User mean
    print("\n      Strategy 2: User Mean Filling")
    user_averages = ratings_matrix.mean(axis=1, skipna=True)
    filled_user = ratings_matrix.values.astype(np.float32).copy()
    
    for i in range(filled_user.shape[0]):
        row_mask = np.isnan(filled_user[i, :])
        filled_user[i, row_mask] = user_averages.iloc[i] if not np.isnan(user_averages.iloc[i]) else 3.0
    
    filled_user = np.nan_to_num(filled_user, nan=3.0)
    
    # SVD with user mean
    U_user, sigma_user, Vt_user = np.linalg.svd(filled_user, full_matrices=False)
    V_user = Vt_user.T
    R_hat_user = (U_user[:, :k] * sigma_user[:k]) @ V_user[:, :k].T
    
    mae_user = np.mean(np.abs(filled_user - R_hat_user))
    rmse_user = np.sqrt(np.mean((filled_user - R_hat_user) ** 2))
    
    results['user_mean'] = {
        'MAE': mae_user,
        'RMSE': rmse_user,
        'top_singular_values': sigma_user[:5].tolist()
    }
    print(f"        Reconstruction MAE: {mae_user:.4f}, RMSE: {rmse_user:.4f}")
    
    # Comparison
    print("\n      Comparison:")
    print(f"        Item Mean - MAE: {mae_item:.4f}, RMSE: {rmse_item:.4f}")
    print(f"        User Mean - MAE: {mae_user:.4f}, RMSE: {rmse_user:.4f}")
    
    if mae_item < mae_user:
        print(f"        [RESULT] Item Mean filling shows lower error")
        results['better_strategy'] = 'item_mean'
    else:
        print(f"        [RESULT] User Mean filling shows lower error")
        results['better_strategy'] = 'user_mean'
    
    return results


def visualize_sensitivity_analysis(robustness_df, filling_comparison):
    """
    Visualize sensitivity analysis results.
    
    Args:
        robustness_df: DataFrame with robustness test results
        filling_comparison: Dict with filling strategy comparison
    """
    print("\n[PLOT] Creating sensitivity analysis visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Error vs Missing Percentage
    ax1 = axes[0]
    ax1.plot(robustness_df['missing_pct'], robustness_df['MAE'], 
             'b-o', linewidth=2, markersize=8, label='MAE')
    ax1.plot(robustness_df['missing_pct'], robustness_df['RMSE'], 
             'r-s', linewidth=2, markersize=8, label='RMSE')
    ax1.set_xlabel('Additional Missing Ratings (%)', fontsize=11)
    ax1.set_ylabel('Reconstruction Error', fontsize=11)
    ax1.set_title('Robustness to Missing Data', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(robustness_df['missing_pct'])
    
    # Plot 2: Filling Strategy Comparison
    ax2 = axes[1]
    strategies = ['Item Mean', 'User Mean']
    mae_values = [filling_comparison['item_mean']['MAE'], 
                  filling_comparison['user_mean']['MAE']]
    rmse_values = [filling_comparison['item_mean']['RMSE'], 
                   filling_comparison['user_mean']['RMSE']]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, mae_values, width, label='MAE', color='steelblue')
    bars2 = ax2.bar(x + width/2, rmse_values, width, label='RMSE', color='coral')
    
    ax2.set_xlabel('Filling Strategy', fontsize=11)
    ax2.set_ylabel('Reconstruction Error', fontsize=11)
    ax2.set_title('Filling Strategy Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_DIR, 'sensitivity_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"        [SAVED] {plot_path}")
    plt.close()
    
    return plot_path


def print_sensitivity_summary(robustness_df, filling_comparison):
    """
    Print summary for sensitivity analysis.
    """
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 60)
    
    print("\n  7.1 Robustness to Missing Data:")
    print(f"    Tested missing percentages: {robustness_df['missing_pct'].tolist()}%")
    print(f"    RMSE range: {robustness_df['RMSE'].min():.4f} - {robustness_df['RMSE'].max():.4f}")
    
    print("\n  7.2 Filling Strategy Comparison:")
    print(f"    Item Mean - MAE: {filling_comparison['item_mean']['MAE']:.4f}")
    print(f"    User Mean - MAE: {filling_comparison['user_mean']['MAE']:.4f}")
    print(f"    Better strategy: {filling_comparison['better_strategy'].replace('_', ' ').title()}")
    
    print("\n" + "=" * 60)
    print("[DONE] Sensitivity Analysis Complete!")
    print("=" * 60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run the complete SVD analysis pipeline.
    Includes: Data Preparation, Full SVD, Truncated SVD, Rating Prediction,
                Latent Factor Interpretation, and Sensitivity Analysis.
    """
    # 1. Data Preparation (meeting dataset requirements: ≥10,000 users, ≥500 items)
    ratings_matrix, user_ids, item_ids = load_ratings_matrix(max_users=10000, max_items=2000)
    
    # Keep a copy of the original sparse matrix for ground truth checking and sensitivity analysis
    original_ratings_matrix = ratings_matrix.copy()
    
    item_averages = calculate_item_averages(ratings_matrix)
    filled_matrix = apply_mean_filling(ratings_matrix, item_averages)
    
    verify_completeness(filled_matrix)
    
    # 2. Full SVD Decomposition
    U, sigma, Vt, eigenvalues, eigenvectors = compute_full_svd(filled_matrix)
    
    # 2.3 Verify orthogonality
    ortho_results = verify_orthogonality(U, Vt)
    
    # Save results
    sigma_df = save_svd_results(U, sigma, Vt, eigenvalues, eigenvectors, 
                                user_ids, item_ids, ortho_results)
    
    # 2.4 Visualize
    n_90, n_95, n_99 = visualize_singular_values(sigma, eigenvalues)
    
    # Print summary for Section 2
    print_summary(sigma, eigenvalues, ortho_results, n_90, n_95, n_99)
    
    # =========================================================================
    # 3. TRUNCATED SVD (LOW-RANK APPROXIMATION)
    # =========================================================================
    
    # 3.1-3.2 Compute truncated SVD for different k values
    k_values = [5, 20, 50, 100]
    approximations = compute_truncated_svd(U, sigma, Vt, k_values)
    
    # 3.3 Calculate reconstruction error
    error_df = calculate_reconstruction_error(filled_matrix, approximations, eigenvalues)
    
    # 3.4 Identify optimal k and visualize
    optimal_k = identify_optimal_k(error_df, eigenvalues)
    visualize_truncated_svd(error_df, eigenvalues, optimal_k)
    
    # =========================================================================
    # 4. RATING PREDICTION WITH TRUNCATED SVD
    # =========================================================================
    
    # 4.1-4.2 Predict missing ratings for target users and items
    predictions_df = predict_missing_ratings(
        U, sigma, Vt, optimal_k, 
        user_ids, item_ids, 
        original_ratings_matrix
    )
    
    # 4.4 Calculate prediction accuracy
    accuracy = calculate_prediction_accuracy(predictions_df)
    
    # 4.3 & 4.5 Save prediction results
    save_prediction_results(predictions_df, accuracy, optimal_k)
    
    # Print summary for Sections 3 & 4
    print_truncated_svd_summary(error_df, optimal_k, predictions_df, accuracy)
    
    # =========================================================================
    # 6. LATENT FACTOR INTERPRETATION
    # =========================================================================
    
    # Get V matrix from Vt
    V = Vt.T
    
    # 6.1-6.2 Analyze top-3 latent factors
    latent_factor_results = analyze_latent_factors(U, V, sigma, user_ids, item_ids)
    
    # 6.3 Visualize latent space
    visualize_latent_space(U, V, sigma, user_ids, item_ids, original_ratings_matrix)
    
    # =========================================================================
    # 7. SENSITIVITY ANALYSIS
    # =========================================================================
    
    # 7.1 Test robustness to missing data
    robustness_df = test_missing_data_robustness(
        original_ratings_matrix, item_averages, U, sigma, Vt
    )
    
    # 7.2 Compare filling strategies
    filling_comparison = compare_filling_strategies(
        original_ratings_matrix, user_ids, item_ids
    )
    
    # Visualize sensitivity analysis
    visualize_sensitivity_analysis(robustness_df, filling_comparison)
    
    # Print summary
    print_sensitivity_summary(robustness_df, filling_comparison)
    
    # Clean up
    del original_ratings_matrix
    gc.collect()
    
    return {
        'U': U,
        'sigma': sigma,
        'Vt': Vt,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'user_ids': user_ids,
        'item_ids': item_ids,
        'ortho_results': ortho_results,
        'filled_matrix': filled_matrix,
        'approximations': approximations,
        'error_df': error_df,
        'optimal_k': optimal_k,
        'predictions_df': predictions_df,
        'accuracy': accuracy,
        'latent_factor_results': latent_factor_results,
        'robustness_df': robustness_df,
        'filling_comparison': filling_comparison
    }


if __name__ == "__main__":
    results = main()


