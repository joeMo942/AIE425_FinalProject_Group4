---
marp: true
theme: default
paginate: true
---

# Section 1: Dimensionality Reduction
## Presentation Slides

**Dianping Dataset Analysis**

---

# Slide 1: PCA Mean-Filling Implementation

This approach handles missing ratings by filling them with item column means before computing the covariance matrix.

**Formula:**
$$Cov(i,j) = \frac{\sum_{u \in U} (R_{u,i} - \mu_i)(R_{u,j} - \mu_j)}{N - 1}$$

---

# Slide 2: Mean-Fill Covariance Matrix

| Metric | Value |
|--------|-------|
| Matrix Size | 11,123 × 11,123 |
| Var(I1) | 0.008386 |
| Var(I2) | 0.008237 |
| Division Factor | N-1 (sample covariance) |

---

# Slide 3: Mean-Fill Predictions

| PCs | Avg Error | Min Error | Max Error |
|-----|-----------|-----------|-----------|
| Top-5 | 0.507 | 0.29 | 0.84 |
| Top-10 | 0.385 | 0.14 | 0.50 |

**Key:** All predictions below item mean (downward bias)

---

# Slide 4: Mean-Fill Top-5 vs Top-10

| Metric | Top-5 PCs | Top-10 PCs | Winner |
|--------|-----------|------------|--------|
| Avg Error | 0.507 | **0.385** | Top-10 |
| Variance Explained | 1.37% | **2.30%** | Top-10 |
| Improvement | - | **24% better** | Top-10 |

---

# Slide 5: PCA MLE Implementation

Maximum Likelihood Estimation uses only observed ratings to compute covariance, avoiding artificial imputation.

**Formula:**
$$Cov_{MLE}(i,j) = \frac{\sum_{u \in Common(i,j)} (R_{u,i} - \mu_i)(R_{u,j} - \mu_j)}{|Common(i,j)| - 1}$$

---

# Slide 6: MLE Covariance Matrix

| Metric | Value |
|--------|-------|
| Matrix Size | 11,123 × 11,123 |
| Var(I1) | 0.557218 |
| Var(I2) | 0.636872 |
| Division Factor | \|Common(i,j)\| - 1 |

---

# Slide 7: MLE Prediction Formula

$$\hat{R}_{u,i} = \mu_i + \sum_{p=1}^{k} t_{u,p} \times W_{i,p}$$

| PCs | Avg Error | Min Error | Max Error |
|-----|-----------|-----------|-----------|
| Top-5 | 0.033 | 0.00 | 0.07 |
| Top-10 | **0.013** | 0.00 | 0.03 |

---

# Slide 8: MLE Top-5 vs Top-10

| Metric | Top-5 PCs | Top-10 PCs | Winner |
|--------|-----------|------------|--------|
| Avg Error | 0.033 | **0.013** | Top-10 |
| Variance Explained | 13.84% | **21.86%** | Top-10 |
| Zero-Error Predictions | 2 | **3** | Top-10 |
| Improvement | - | **60% better** | Top-10 |

---

# Slide 9: SVD Implementation

Singular Value Decomposition factorizes the rating matrix into user and item latent factors.

**Decomposition:** $R = U \times \Sigma \times V^T$

- Orthogonality verified: U^T×U = I, V^T×V = I
- 2,000 singular values extracted (Full SVD)

---

# Slide 10: Truncated SVD Results

| Metric | Value |
|--------|-------|
| Latent Factors (k) | 100 |
| Computation Time | **3.8 seconds** |
| Memory Usage | **191 MB** |
| Predictions/sec | **119,971** |

**Formula:** $\hat{r}_{u,i} = \mu + \sum_{k} u_{u,k} \cdot \sigma_k \cdot v_{i,k}$

---

# Slide 11: Cold-Start Analysis

| User Type | MAE | Impact |
|-----------|-----|--------|
| Cold-start (≤5 ratings) | 0.707 | +68.9% penalty |
| Warm (≥20 ratings) | 0.419 | Baseline |

**Challenge:** Matrix factorization cannot predict for users with zero interactions.

---

# Slide 12: Latent Factor Interpretation

| Factor | Singular Value | Variance | Interpretation |
|--------|----------------|----------|----------------|
| Factor 1 | σ = 83.19 | 3.56% | Global mean / Overall popularity |
| Factor 2 | σ = 70.31 | 2.54% | Major genre dimension |
| Factor 3 | σ = 62.01 | 1.98% | Finer preference distinctions |

---

# Slide 13: Method Comparison

| Metric | Mean-Fill | MLE | SVD (k=100) |
|--------|-----------|-----|-------------|
| **Avg Error** | 0.385 | **0.013** | ~0.20 |
| **Variance Captured** | 2.30% | 21.86% | ~85% |
| **Computation Time** | 10-30 min | 10-30 min | **3.8 sec** |
| **Memory** | ~2-3 GB | ~2-3 GB | **191 MB** |
| **Handles Sparsity** | ✗ | ✗ | ✓ |

---

# Slide 14: Complexity Analysis

| Method | Time Complexity | Space Complexity |
|--------|-----------------|------------------|
| PCA (Mean-Fill) | O(n³) | O(n²) |
| PCA (MLE) | O(n³) | O(n²) |
| SVD (Truncated) | **O(k × nnz)** | **O(nnz + k(m+n))** |

---

# Slide 15: Winner by Category

| Aspect | Winner |
|--------|--------|
| Prediction Accuracy | **MLE** |
| Scalability | **SVD** |
| Memory Efficiency | **SVD** |
| Statistical Rigor | **MLE** |
| Computational Speed | **SVD** |

---

# Slide 16: Dataset Overview

**Dianping Dataset**
- 147,914 users × 11,123 items × 2,149,655 ratings
- Sparsity: 99.87%
- **Target Users:** U1=134471, U2=27768, U3=16157
- **Target Items:** I1=1333, I2=1162

---

# Slide 17: Finding 1 - MLE vs Mean-Fill

| Metric | Mean-Fill | MLE | Improvement |
|--------|-----------|-----|-------------|
| Avg Error (Top-10) | 0.385 | 0.013 | **97% better** |
| Variance Explained | 2.30% | 21.86% | **10x more** |
| Zero-Error Predictions | 0 | 3 | MLE wins |

**Root Cause:** Mean-Fill distorts covariance; MLE uses only observed data.

---

# Slide 18: Finding 2 - SVD Scalability

| Metric | PCA Methods | SVD |
|--------|-------------|-----|
| Computation Time | 10-30 min | **3.8 sec** |
| Memory Usage | 2-3 GB | **191 MB** |
| Predictions/sec | ~1,000 | **119,971** |
| Sparse Matrix Support | ✗ | ✓ |

---

# Slide 19: Finding 3 - Variance & Quality

| Method | Variance Captured | Prediction Quality |
|--------|-------------------|-------------------|
| Mean-Fill (Top-10) | 2.30% | Poor (Error: 0.385) |
| MLE (Top-10) | 21.86% | Excellent (Error: 0.013) |
| SVD (k=100) | ~85% | Good (MAE: ~0.20) |

**Insight:** Higher variance = better latent representation = lower error

---

# Slide 20: Finding 4 - Cold-Start Challenge

| User Type | Rating Count | MAE | Impact |
|-----------|--------------|-----|--------|
| Cold-start | ≤5 ratings | 0.707 | +68.9% penalty |
| Warm users | ≥20 ratings | 0.419 | Baseline |

**Solution:** Hybrid approaches (CF + content features)

---

# Slide 21: Finding 5 - Top-10 vs Top-5

| Method | Top-5 Error | Top-10 Error | Improvement |
|--------|-------------|--------------|-------------|
| Mean-Fill | 0.507 | 0.385 | 24% |
| MLE | 0.033 | 0.013 | 60% |

**Reason:** More dimensions capture more user preference patterns.

---

# Slide 22: Recommendations

| Scenario | Recommended Method | Reason |
|----------|-------------------|--------|
| **Production** | Truncated SVD (k=100) | Speed, memory, scalability |
| **Max Accuracy** | PCA MLE (Top-10 PCs) | Lowest error (0.013) |
| **Cold-Start** | Hybrid approach | CF + content features |
| **Small Data** | PCA Mean-Fill | Simplicity |

---

# Slide 23: Final Verdict

## Best Overall: Truncated SVD
- ✓ **100x faster** computation
- ✓ **10x lower** memory usage
- ✓ **Native sparse** matrix support

## Best Accuracy: PCA MLE
- ✓ **97% error reduction** vs Mean-Fill
- ✓ **3 zero-error** predictions out of 6

---

# Slide 24: Conclusion

**Dimensionality reduction is essential** for recommendation systems.

| Trade-off | Choice |
|-----------|--------|
| Accuracy Priority | PCA MLE |
| Scalability Priority | Truncated SVD |

**For production systems with large datasets, Truncated SVD is the optimal choice.**

---

# Thank You

**Section 1: Dimensionality Reduction - Complete**
