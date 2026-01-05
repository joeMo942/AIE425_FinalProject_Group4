# PCA Mean-Filling Method: Complete Roadmap (Steps 1-7)

## Overview

This document provides a detailed explanation of the PCA (Principal Component Analysis) mean-filling method implementation for dimensionality reduction in a recommendation system.

**Dataset:** Amazon Movies & TV Reviews
- **Users:** 147,914
- **Items:** 11,123
- **Ratings:** 2,149,655

**Target Items:**
- I1 = Item 1333
- I2 = Item 1162

---

## Step 1: Load Data

### What We Did:
- Loaded user average ratings (`r_u`) from preprocessed data
- Loaded item average ratings (`r_i`) from preprocessed data
- Loaded target users and target items from text files

### Output:
- `r_u`: (147,914 users x 2 columns) - user, r_u_bar
- `r_i`: (11,123 items x 2 columns) - item, r_i_bar

### Saved Files:
- `step1_r_u.csv`
- `step1_r_i.csv`

---

## Step 2: Load Target Users and Items

### What We Did:
- Loaded 3 target users: [134471, 27768, 16157]
- Loaded 2 target items: [1333, 1162]

### Saved Files:
- `step2_targets.csv`

---

## Step 3: Mean-Filling Method

### What We Did:
1. Created user-item rating matrix for target items only
2. Identified missing values (users who didn't rate target items)
3. Filled missing values with **item column mean**

### Statistics:
- Rating Matrix Shape: (4,066 users x 2 items)
- Missing values before filling: 3,991 (49.08%)
- Missing values after filling: 0

### Item Means:
- I1 (Item 1333): Mean = 3.694656
- I2 (Item 1162): Mean = 3.737722

### Saved Files:
- `step3_original_matrix.csv`
- `step3_mean_filled_matrix.csv`
- `step3_item_means.csv`

---

## Step 4: Calculate Item Average Ratings

### What We Did:
- Used pre-computed item means from `r_i`
- These are the average ratings for all 11,123 items

### Saved Files:
- `step4_all_item_means.csv`

---

## Step 5: Calculate Centered Ratings

### Formula:
```
centered_rating = actual_rating - item_mean
```

### What We Did:
- For each rating in the dataset, computed: `(r_ui - r_bar_i)`
- This centers ratings around zero for each item

### Output:
- 2,149,655 centered ratings for all user-item pairs

### Saved Files:
- `step5_centered_ratings.csv`

---

## Step 6: Compute Covariance Matrix

### Formula:
```
Cov(i,j) = Sum[(r_ui - r_bar_i)(r_uj - r_bar_j)] / (N-1)
```

### What We Did:
1. For each pair of items (i, j):
   - Found users who rated **both** items
   - Computed sum of products of centered ratings
   - Divided by (N-1) for sample covariance

2. Memory-efficient approach:
   - Did NOT create dense filled matrix
   - Mean-filled values contribute zero after centering
   - Only users who rated both items contribute

### Output:
- Covariance Matrix: 11,123 x 11,123
- Symmetric matrix (Cov(i,j) = Cov(j,i))
- Diagonal = Variances

### Target Items Covariance:
- Var(I1): 0.008386
- Var(I2): 0.008237
- Cov(I1, I2): 0.000067

### Saved Files:
- `step6_covariance_matrix.csv`

---

## Step 7: PCA Eigendecomposition + Top Peers

### Part A: Eigenvalue Decomposition

#### What We Did:
1. Computed eigenvalues and eigenvectors of covariance matrix
2. Sorted in descending order (largest eigenvalue first)
3. Created projection matrix W from top-k eigenvectors

#### Top 10 Eigenvalues:
| Rank | Eigenvalue | Variance % |
|------|------------|------------|
| 1 | 0.044118 | 0.48% |
| 2 | 0.021755 | 0.24% |
| 3 | 0.021163 | 0.23% |
| 4 | 0.020252 | 0.22% |
| 5 | 0.019145 | 0.21% |
| 6 | 0.018785 | 0.20% |
| 7 | 0.018204 | 0.20% |
| 8 | 0.016950 | 0.18% |
| 9 | 0.016168 | 0.17% |
| 10 | 0.015986 | 0.17% |

#### Variance Explained:
- Top-5 PCs: 1.37%
- Top-10 PCs: 2.30%
- Total Variance: 9.258360

### Part B: Top Peers for Target Items (Based on Covariance)

#### I1 (Item 1333) - Top 5 Peers:
| Rank | Peer Item | Covariance |
|------|-----------|------------|
| 1 | 22 | 0.000318 |
| 2 | 316 | 0.000269 |
| 3 | 1175 | 0.000228 |
| 4 | 1197 | 0.000202 |
| 5 | 907 | 0.000197 |

#### I2 (Item 1162) - Top 5 Peers:
| Rank | Peer Item | Covariance |
|------|-----------|------------|
| 1 | 581 | 0.000387 |
| 2 | 623 | 0.000346 |
| 3 | 627 | 0.000261 |
| 4 | 108 | 0.000248 |
| 5 | 277 | 0.000245 |

### Saved Files:
- `step7_eigenvalues.csv`
- `step7_top5_eigenvectors.csv`
- `step7_top10_eigenvectors.csv`
- `step7_target_item_peers.csv`

---

## Summary of Results Files

| Step | File | Description |
|------|------|-------------|
| 1 | step1_r_u.csv | User average ratings |
| 1 | step1_r_i.csv | Item average ratings |
| 2 | step2_targets.csv | Target users and items |
| 3 | step3_original_matrix.csv | Original rating matrix (with NaN) |
| 3 | step3_mean_filled_matrix.csv | Mean-filled rating matrix |
| 3 | step3_item_means.csv | Item means for target items |
| 4 | step4_all_item_means.csv | All item means |
| 5 | step5_centered_ratings.csv | Centered ratings for all ratings |
| 6 | step6_covariance_matrix.csv | Full covariance matrix (11123x11123) |
| 7 | step7_eigenvalues.csv | All eigenvalues with variance % |
| 7 | step7_top5_eigenvectors.csv | Top-5 principal components |
| 7 | step7_top10_eigenvectors.csv | Top-10 principal components |
| 7 | step7_target_item_peers.csv | Top peers for I1 and I2 |
