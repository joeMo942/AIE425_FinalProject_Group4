# PCA Mean-Filling Analysis Report

## Overview

This document provides a comprehensive analysis of the `pca_mean_filling.py` implementation for dimensionality reduction in a recommendation system.

**Dataset:** Amazon Movies & TV Reviews
- **Users:** 147,914
- **Items:** 11,123
- **Ratings:** 2,149,655

**Target Users:** U1=134471, U2=27768, U3=16157
**Target Items:** I1=1333, I2=1162

---

## Implementation Steps

### Step 0: Data Loading (Preparation)
- Loads user/item average ratings and target users/items
- r_u shape: (147,914 x 2)
- r_i shape: (11,123 x 2)

### Step 1 & 2: Calculate Average + Mean-Filling
- Calculates average rating for I1 and I2
- Mean-fills missing ratings with item column mean
- Missing values: 3,991 (49.08%)
- I1 mean: 3.69, I2 mean: 3.74

**Formula:**
```
μᵢ = (1/nᵢ) × Σ R_{u,i}  (for all users u who rated item i)

R'_{u,i} = R_{u,i}  if observed
R'_{u,i} = μᵢ      if missing
```

**LaTeX:**
$$\mu_i = \frac{1}{n_i} \sum_{u} R_{u,i}$$

$$R'_{u,i} = \begin{cases} R_{u,i} & \text{if observed} \\ \mu_i & \text{if missing} \end{cases}$$

### Step 3: Average Rating for Each Item
- Uses pre-computed item means from r_i

### Step 4: Centered Ratings
- Computes (rating - item_mean) for all 2,149,655 ratings

**Formula:**
```
Centered_{u,i} = R_{u,i} - μᵢ
```

**LaTeX:**
$$Centered_{u,i} = R_{u,i} - \mu_i$$

### Step 5 & 6: Covariance Matrix
- Computes covariance for each two items
- Generates 11,123 x 11,123 matrix
- Memory-efficient: only users who rated BOTH items contribute
- Divides by N-1 (sample covariance)

**Formula:**
```
Cov(i,j) = Σ (R_{u,i} - μᵢ)(R_{u,j} - μⱼ) / (N - 1)
         for all users u who rated BOTH items i and j
```

**LaTeX:**
$$Cov(i,j) = \frac{\sum_{u \in U_{i,j}} (R_{u,i} - \mu_i)(R_{u,j} - \mu_j)}{N - 1}$$

### Step 7: PCA Eigendecomposition + Top Peers
- Computes eigenvalues/eigenvectors
- Top-5 PCs explain 1.37% variance
- Top-10 PCs explain 2.30% variance
- Identifies top 5/10 peers for I1 and I2

**Formula:**
```
Σ × W = W × Λ  (eigenvalue equation)

Variance Explained = Σᵢ₌₁ᵏ λᵢ / Σᵢ₌₁ⁿ λᵢ × 100%

Reconstructed Σ ≈ W × Λ × Wᵀ
```

**LaTeX:**
$$\Sigma W = W \Lambda$$

$$\text{Variance Explained} = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{n} \lambda_i} \times 100\%$$

$$\hat{\Sigma} \approx W \Lambda W^T$$

### Step 8 & 10: User Projection
- Projects users to 5D and 10D latent space

**Formula:**
```
t_{u,p} = Σⱼ (R_{u,j} - μⱼ) × W_{j,p}  (for all items j rated by user u)
```

**LaTeX:**
$$t_{u,p} = \sum_{j \in Obs(u)} (R_{u,j} - \mu_j) \times W_{j,p}$$

### Step 9 & 11: Rating Prediction
- Uses k-NN with cosine similarity in latent space
- 20 nearest neighbors
- Predicts ratings for target users on target items

**Formula:**
```
cos(u, v) = (UserVector_u · UserVector_v) / (||UserVector_u|| × ||UserVector_v||)

Predicted_{u,i} = μᵢ + Σᵥ (sim(u,v) × Centered_{v,i}) / Σᵥ |sim(u,v)|
                 for v in k-nearest neighbors of u
```

**LaTeX:**
$$cos(u,v) = \frac{\vec{U}_u \cdot \vec{U}_v}{||\vec{U}_u|| \times ||\vec{U}_v||}$$

$$\hat{R}_{u,i} = \mu_i + \frac{\sum_{v \in N(u)} sim(u,v) \times (R_{v,i} - \mu_i)}{\sum_{v \in N(u)} |sim(u,v)|}$$

---


## Step 9 vs Step 11 Comparison

### Key Difference
- **Step 9:** Uses Top-5 Principal Components (5D space)
- **Step 11:** Uses Top-10 Principal Components (10D space)

### Prediction Results

| User | Item | Top-5 PCs (Step 9) | Top-10 PCs (Step 11) | Actual | Error (Top-5) | Error (Top-10) |
|------|------|-------------------|---------------------|--------|--------------|----------------|
| U1 | I1 | 3.30 | 3.55 | 3.69 | 0.39 | **0.14** |
| U1 | I2 | 3.10 | 3.25 | 3.74 | 0.64 | **0.49** |
| U2 | I1 | 3.20 | 3.20 | 3.69 | 0.49 | 0.50 |
| U2 | I2 | 3.35 | 3.45 | 3.74 | 0.39 | **0.29** |
| U3 | I1 | 2.85 | 3.25 | 3.69 | 0.84 | **0.45** |
| U3 | I2 | 3.45 | 3.30 | 3.74 | **0.29** | 0.44 |

### Error Summary

| Metric | Top-5 PCs (Step 9) | Top-10 PCs (Step 11) |
|--------|-------------------|---------------------|
| Average Error | 0.51 | 0.39 |
| Min Error | 0.29 | 0.14 |
| Max Error | 0.84 | 0.50 |

---

## Analysis and Comments

### 1. Top-10 PCs Generally Perform Better
Top-10 PCs achieve **lower average error (0.39)** compared to Top-5 PCs (0.51). This is expected because:
- More dimensions capture more variance in user preferences
- Better representation leads to more accurate neighbor finding

### 2. Exception Case: U3 on I2
Interestingly, Top-5 PCs performed better for U3 on I2:
- Top-5: Error = 0.29
- Top-10: Error = 0.44

This suggests that for some users, additional dimensions may introduce noise rather than signal.

### 3. All Predictions Below Item Mean
All predictions are below the actual (item mean), indicating a bias:
- Neighbors tend to have lower-than-average ratings
- The centered deviation weighted sum is negative on average

### 4. High Neighbor Similarity
Both approaches find neighbors with very high similarity (0.92-1.0), indicating:
- Many users have similar rating patterns in the reduced space
- The 5D/10D space effectively groups similar users

### 5. Variance Explained is Low
Only 1.37% (Top-5) and 2.30% (Top-10) of variance is explained, meaning:
- User preferences are highly diverse
- Most variance is in the "long tail" of less important dimensions
- More PCs might improve predictions further

---

## Conclusion

**Top-10 PCs (Step 11) is the better choice overall**, achieving 24% lower average error than Top-5 PCs. However, the optimal number of PCs may vary per user, suggesting an adaptive approach could yield even better results.

### Recommendations
1. Consider using more PCs (15-20) if computational resources allow
2. Investigate user-specific optimal dimensionality
3. Address the prediction bias (all predictions below mean)
