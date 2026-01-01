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

### Step 1-2: Data Loading
- Loads user/item average ratings and target users/items
- r_u shape: (147,914 x 2)
- r_i shape: (11,123 x 2)

### Step 3: Mean-Filling
- Creates user-item matrix for target items
- Missing values: 3,991 (49.08%)
- Fills missing with item column mean
- I1 mean: 3.69, I2 mean: 3.74

### Step 4-5: Centered Ratings
- Computes (rating - item_mean) for all 2,149,655 ratings

### Step 6: Covariance Matrix
- 11,123 x 11,123 matrix
- Memory-efficient: only users who rated BOTH items contribute
- Divides by N-1 (sample covariance)

### Step 7: PCA Eigendecomposition
- Computes eigenvalues/eigenvectors
- Top-5 PCs explain 1.37% variance
- Top-10 PCs explain 2.30% variance
- Identifies top peers for I1 and I2

### Step 8 & 10: User Projection
- Projects users to 5D and 10D latent space
- Formula: UserVector = CenteredRatings × W

### Step 9 & 11: Rating Prediction
- Uses k-NN with cosine similarity in latent space
- 20 nearest neighbors
- Predicts ratings for target users on target items

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
