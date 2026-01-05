# Section 1: Dimensionality Reduction 📉

This section explores dimensionality reduction techniques applied to the user-item rating matrix. The goal is to reduce the high-dimensional feature space while retaining the most significant patterns in user behavior.

## 📂 Project Structure

```
SECTION1_DimensionalityReduction/
├── code/
│   ├── pca_mean_filling.py     # PCA method 1: Imputing missing values with item means
│   ├── pca_mle.py              # PCA method 2: Maximum Likelihood Estimation (ignores missing)
│   ├── svd_analysis.py         # Full SVD decomposition and analysis (Sparsity optimized)
│   └── utils.py                # Shared utility functions
├── data/                       # Input datasets (100k.csv, 90k.csv, etc.)
├── results/                    # CSV outputs (Covariance matrices, Eigenvalues, Predictions)
├── plots/                      # Generated visualizations (Scree plots, Error charts)
└── tables/                     # Formatted tables for reports
```

## 🚀 Usage

Ensure you have the required dependencies installed. You can run each analysis script independently.

### 1. PCA with Mean Filling
Imputes missing values using item averages before computing the covariance matrix and PCA.
```bash
python code/pca_mean_filling.py
```
*   **Outputs:** Covariance matrices (full & reconstructed), Top-5/10 Eigenvalues, Rating predictions.

### 2. PCA with MLE (Maximum Likelihood Estimation)
Estimates the covariance matrix using only observed ratings (ignoring missing pairs), effectively handling sparsity without imputation.
```bash
python code/pca_mle.py
```
*   **Results:** Optimized for sparse data distributions. Matches Step 2 of the assignment.

### 3. SVD Analysis
Performs Singular Value Decomposition (SVD) on the ratings matrix to analyze latent factors.
```bash
python code/svd_analysis.py
```
*   **Features:**
    *   Full SVD Decomposition ($R = U \Sigma V^T$).
    *   Orthogonality Verification.
    *   Low-Rank Approximation (Truncated SVD) for various $k$ values.
    *   Elbow Method Analysis for optimal $k$.

## 🧠 Methodology

### Part 1: PCA (Mean Filling)
*   **Approach:** Addresses sparsity by filling missing entries with the column (item) mean.
*   **Steps:**
    1.  Centering the data (subtracting item means).
    2.  Computing the dense covariance matrix.
    3.  Eigendecomposition to find Principal Components.
    4.  Projecting users into the reduced $k$-dimensional space.

### Part 2: PCA (MLE)
*   **Approach:** Calculates covariance between items using only the subset of users who rated *both* items.
*   **Advantage:** Avoids the bias introduced by artificial mean filling.
*   **Scaling:** Adjusted by the number of common users ($N-1$).

### Part 3: SVD Matrix Factorization
*   **Technique:** Decomposes the mean-centered rating matrix into user factors ($U$), item factors ($V$), and singular values ($\Sigma$).
*   **Optimization:** Uses sparse matrix operations (`scipy.sparse.linalg.svds`) to handle large datasets memory-efficiently.
*   **Validation:** Reconstruction error (RMSE/MAE) is calculated for different rank approximations ($k=5, 20, 50, 100$).
