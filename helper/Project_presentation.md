---
marp: true
theme: default
paginate: true
---

# Intelligent Recommender System
## Project Presentation

**Section 1: Dimensionality Reduction | Section 2: Domain Recommender**

---

# SECTION 1: Dimensionality Reduction

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

# Slide 16: Section 1 Conclusion

**Dimensionality reduction is essential** for recommendation systems.

| Trade-off | Choice |
|-----------|--------|
| Accuracy Priority | PCA MLE |
| Scalability Priority | Truncated SVD |

**For production systems with large datasets, Truncated SVD is the optimal choice.**

---

# SECTION 2: Domain Recommender

**Twitch.tv Streaming Platform**

---

# Slide 17: Domain Introduction

**Twitch.tv** - Live streaming platform for video games, esports, and creative content.

**System Objectives:**
1. Predict user preference for unobserved streamers
2. Handle the Cold-Start Problem
3. Improve accuracy using Hybrid Techniques
4. Recommend based on game genres and content similarity

---

# Slide 18: Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Ratings | ~600,000 interactions |
| Users | ~90,000 unique users |
| Items (Streamers) | ~1,400 active streamers |
| Rating Scale | 1-5 (implicit feedback) |
| Sparsity | >99% |

**Source:** Custom scraped dataset enriched with IGDB API metadata

---

# Slide 19: Feature Extraction

**Text Features (90% weight):**
- Game titles, genres, themes from IGDB
- TF-IDF Vectorization (max_features=1000)

**Numerical Features (10% weight):**
- Average Viewers, Followers
- Log-transformation + Min-Max Scaling

---

# Slide 20: Cascade Hybrid Architecture

```
User Query → Content-Based (Top 50) → SVD Ranking → Top 10
```

**Stage 1:** Content-Based scans 1,400+ items → selects Top 50 candidates

**Stage 2:** SVD predicts ratings only for 50 candidates → ranks Top 10

---

# Slide 21: Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Time Decay | Removed | 42-day dataset too short |
| SVD Factors | k=20 | Balance nuance vs overfitting |
| Feature Weight | 90% Text / 10% Pop | Avoid popularity bias |
| Discount Factor | β=1.0 | Penalize few co-ratings |

---

# Slide 22: Evaluation Results

| Method | RMSE | Hit Rate @ 10 |
|--------|------|---------------|
| Random Baseline | 1.854 | 0.02% |
| Popularity Baseline | 1.420 | 2.14% |
| Content-Based | 1.150 | 5.80% |
| Collaborative (SVD) | 0.982 | 8.45% |
| **Hybrid (Cascade)** | **0.965** | **9.12%** |

---

# Slide 23: Hybrid Superiority Analysis

**Why Hybrid Wins:**
- Content-Based filters irrelevant genres (RPG → FPS fan)
- SVD ranks remaining candidates with high precision
- 100% cold-start coverage via content-based fallback

**Improvement over SVD alone:** +7.9% Hit Rate

---

# Slide 24: Web Application

**Tech Stack:**
- Backend: FastAPI (Python)
- Frontend: HTML5, CSS3, Jinja2

**Features:**
- Visual game selection (IGDB covers)
- Real-time filtering (games, languages)
- Twitch API integration (profile pictures, follower counts)
- Cold-start simulation

---

# Slide 25: Section 2 Conclusion

**Cascade Hybrid Architecture** successfully:
- ✓ Reduced SVD computational load
- ✓ Improved accuracy by filtering noise
- ✓ Handled cold-start users via content fallback

**Feature weighting (90% text / 10% popularity)** prevented popularity bias.

---

# OVERALL PROJECT CONCLUSION

---

# Slide 26: Project Summary

| Section | Key Method | Best Result |
|---------|------------|-------------|
| **Section 1** | PCA MLE | 0.013 avg error (97% better) |
| **Section 1** | Truncated SVD | 3.8 sec, 191 MB (100x faster) |
| **Section 2** | Cascade Hybrid | 9.12% Hit Rate (best overall) |

---

# Slide 27: Key Takeaways

1. **MLE > Mean-Fill:** 97% error reduction by using observed data only
2. **SVD scales:** 100x faster, 10x less memory than PCA
3. **Hybrid wins:** Combining methods outperforms any single approach
4. **Cold-start needs content:** CF alone has 0% coverage for new users
5. **Feature weighting matters:** 90% text prevents popularity bias

---

# Slide 28: Final Recommendations

| Scenario | Recommended |
|----------|-------------|
| Production (large-scale) | Truncated SVD (k=100) |
| Maximum accuracy | PCA MLE (Top-10 PCs) |
| Cold-start handling | Content-Based fallback |
| Practical deployment | Cascade Hybrid |

---

# Thank You

**Intelligent Recommender System Project - Complete**

*Section 1: Dimensionality Reduction | Section 2: Domain Recommender*
