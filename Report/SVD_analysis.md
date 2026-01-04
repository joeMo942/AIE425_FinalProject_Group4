# SVD Analysis - Detailed Report

## Executive Summary

This report documents the complete SVD (Singular Value Decomposition) analysis performed on the MovieLens dataset for collaborative filtering.

**Two SVD approaches used:**
- **Full SVD (Section 2):** Sampled data (5,000 users × 2,000 items) - demonstrates complete decomposition
- **Truncated SVD (Sections 3-8):** Full dataset (147,914 users × 11,123 items) - for actual predictions

---

## Section 1: Data Preparation

### What Was Done
1. **Loaded the preprocessed dataset** from data directory
2. **For Full SVD:** Created sampled subset (5,000 users × 2,000 items)
3. **For Truncated SVD:** Created sparse matrix of full data

### Data Statistics

| Dataset | Users | Items | Ratings | Sparsity |
|---------|-------|-------|---------|----------|
| **Sampled (Full SVD)** | 5,000 | 2,000 | 444,950 | 95.55% |
| **Full (Truncated SVD)** | 147,914 | 11,123 | 2,149,655 | 99.87% |

---

## Section 2: FULL SVD Decomposition (Sampled Data)

### What Was Done

1. **Loaded sampled matrix:** 5,000 users × 2,000 items
2. **Calculated item averages (r̄ᵢ)** for each item
3. **Applied mean-filling:** Replaced NaN with item average
4. **Verified completeness:** 0 missing values after filling
5. **Computed FULL SVD:** `R = U × Σ × Vᵀ` using `np.linalg.svd()`
6. **Verified orthogonality:** U^T×U = I and V^T×V = I

### Full SVD Mathematical Details

```
R = U × Σ × Vᵀ

Dimensions:
- R: 5,000 × 2,000 (ratings matrix)
- U: 5,000 × 5,000 (user latent factors - FULL)
- Σ: 5,000 × 2,000 (diagonal singular values)
- Vᵀ: 2,000 × 2,000 (item latent factors - FULL)
```

### Eigenpair Computation (2.2)

| Metric | Value |
|--------|-------|
| Number of eigenvalues | 2,000 (min(m,n)) |
| Largest eigenvalue (λ₁) | 136,144,008.25 |
| Smallest eigenvalue (λₙ) | 4.94 |
| Condition number | ~27,000,000 |

### V Normalization Check

```
Normalizing eigenvectors: vᵢ → eᵢ = vᵢ/||vᵢ||
Column norms range: [1.000000, 1.000000]
✓ All columns already normalized (from SVD)
```

### U Computation Verification

```
Verifying: uᵢ = (R × eᵢ) / σᵢ
Relative reconstruction error: 9.85e-14
✓ Formula verified with numerical precision
```

### Orthogonality Verification (2.3)

| Check | Frobenius Deviation | Max Element Deviation | Status |
|-------|--------------------|-----------------------|--------|
| U^T × U = I | 2.36e-13 | 1.67e-14 | ✓ PASS |
| V^T × V = I | 1.92e-13 | 6.66e-15 | ✓ PASS |

### Variance Analysis (2.4)

**Components needed for variance thresholds:**

| Threshold | Components (k) | % of original |
|-----------|---------------|---------------|
| 90% variance | ~200 | 10% |
| 95% variance | ~400 | 20% |
| 99% variance | ~800 | 40% |

### Output Files (Full SVD)

| File | Description |
|------|-------------|
| `singular_values.csv` | All 2,000 singular values with variance |
| `U_matrix.csv` | User latent factors (first 50 components) |
| `V_matrix.csv` | Item latent factors (first 50 components) |
| `orthogonality_verification.txt` | U^T×U and V^T×V deviation report |
| `svd_analysis.png` | Singular value & scree plots |

---

## Section 3: TRUNCATED SVD (Full Dataset)

### What Was Done

1. **Loaded full sparse matrix:** 147,914 users × 11,123 items
2. **Applied mean-centering:** Subtracted global mean (3.7432)
3. **Computed truncated SVD:** k=100 using `scipy.sparse.linalg.svds()`
4. **Verified orthogonality** of truncated components

### Why Truncated SVD?

| Comparison | Full SVD | Truncated SVD |
|------------|----------|---------------|
| Matrix size | 147K × 11K | 147K × 11K |
| Memory for U | ~174 GB | ~118 MB |
| Computation time | Hours | **3.8 seconds** |
| Singular values | All 11,123 | Top 100 |

### Truncated SVD Results (k=100)

| Metric | Value |
|--------|-------|
| Decomposition time | 3.8 seconds |
| Memory usage | 190 MB |
| Variance explained | 100% (of top-100) |

---

## Section 4: Rating Prediction

### Prediction Formula

```python
# For user u and item i:
# 1. Get user latent vector
u_vector = U[user_idx, :]  # Shape: (100,)

# 2. Get item latent vector  
v_vector = V[item_idx, :]  # Shape: (100,)

# 3. Compute prediction
raw_pred = np.dot(u_vector * sigma, v_vector) + global_mean
predicted = np.clip(raw_pred, 1.0, 5.0)
```

**LaTeX:**
$$\hat{r}_{u,i} = \mu + \sum_{k=1}^{K} u_{u,k} \cdot \sigma_k \cdot v_{i,k}$$

**Where:**
- $\hat{r}_{u,i}$ = predicted rating for user $u$ on item $i$
- $\mu$ = global mean rating
- $u_{u,k}$ = user $u$'s latent factor $k$
- $\sigma_k$ = singular value for factor $k$
- $v_{i,k}$ = item $i$'s latent factor $k$

---

## Section 5: SVD vs PCA Comparison

| Metric | SVD (k=100) | PCA Mean-Fill | PCA MLE |
|--------|-------------|---------------|---------|
| Decomposition Time | **3.8 sec** | ~10-30 min | ~10-30 min |
| Memory | **191 MB** | ~2-3 GB | ~2-3 GB |
| Prediction Speed | **119,971/sec** | ~1,000/sec | ~1,000/sec |
| Handles Sparsity | ✓ Native | ✗ Dense only | ✗ Dense only |

---

## Section 6: Latent Factor Interpretation

### Factor 1 (σ=83.19, 3.56% variance)
**Interpretation:** Global mean effect / Overall popularity

### Factor 2 (σ=70.31, 2.54% variance)
**Interpretation:** Major genre dimension (e.g., action vs drama)

### Factor 3 (σ=62.01, 1.98% variance)
**Interpretation:** Finer preference distinctions

---

## Section 7: Sensitivity Analysis

| k | MAE | RMSE | Variance |
|---|-----|------|----------|
| 10 | 0.6376 | 0.8199 | 18.8% |
| 20 | 0.6274 | 0.8124 | 31.2% |
| 50 | 0.6018 | 0.7933 | 61.6% |
| 75 | 0.5857 | 0.7809 | 82.0% |
| **100** | **0.5704** | **0.7679** | **100%** |

---

## Section 8: Cold-Start Analysis

| User Type | Count | MAE |
|-----------|-------|-----|
| Cold-start (≤5 ratings) | 78,983 | 0.7069 |
| Warm (≥20 ratings) | 29,104 | 0.4185 |
| **Penalty** | | **+68.9%** |

---

## Summary

| Section | Method | Dataset | Key Result |
|---------|--------|---------|------------|
| 2 | **Full SVD** | 5K×2K sample | R=UΣVᵀ, orthogonality verified |
| 3-8 | **Truncated SVD** | 147K×11K full | k=100, MAE=0.57, 3.8 sec |
