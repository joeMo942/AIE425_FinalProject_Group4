# PCA for Recommender Systems: Implementation Specification

This document outlines the exact mathematical workflow for implementing Principal Component Analysis (PCA) using the mean-filling method as detailed in the lecture slides.

---

## Phase 1: Preprocessing & Data Augmentation
**Goal:** Convert a sparse ratings matrix $R$ into a complete centered matrix $R_f$.

### 1. Augmentation (Mean-Filling)
* **Initialize Matrix $R$**: Represent the incomplete ratings matrix.
* **Column Mean Calculation**: For each product (column), calculate the mean rating ($\bar{r}$) based on existing values.
* **Impute Missing Values**: Replace every missing entry (`NaN` or `?`) in a column with that column's mean value.

### 2. Mean-Centering
* **Subtract Means**: Subtract the product mean ($\bar{r}$) from every entry in its respective column.
* **Centered Matrix ($R'_f$)**: The resulting matrix ensures each column has a mean of zero.
    * *Formula*: $X'_i = P1_i - \bar{P1}$.

---

## Phase 2: Covariance Matrix Computation
**Goal:** Capture statistical relationships between products.

### 3. Compute Item Covariance Matrix ($\Sigma$)
* **Formula**: Use the unbiased sample covariance formula ($n-1$) for every product pair ($X, Y$):
  $$Cov(X, Y) = \frac{\sum_{i}(X_i - \bar{X})(Y_i - \bar{Y})}{n - 1}$$
* **Process**:
    1. Multiply centered deviations of two products for each user.
    2. Sum the products across all users.
    3. Divide by total users minus one ($n-1$).
* **Validation**: The diagonal of $\Sigma$ must equal the variance of each product.

---

## Phase 3: Dimensionality Reduction (The PCA Engine)
**Goal:** Project high-dimensional user data into a reduced space.

### 4. Eigen-Decomposition
* **Eigenvalues ($\lambda$)**: Solve the characteristic equation $det(\Sigma - \lambda I) = 0$.

* **Eigenvectors ($p$)**: Solve $(\Sigma - \lambda I)p = 0$ for each $\lambda$ to find the principal directions.

### 5. Construct Projection Matrix ($W$)
* **Selection**: Select the top-k eigenvectors corresponding to the k largest eigenvalues.
* **Stacking**: Stack them vertically to form a projection matrix $W$.

### 6. User Projection
* **Matrix Multiplication**: Multiply each user's centered row vector $r'_i$ by $W$: $u_i = r'_i W$.

---

## Phase 4: Prediction & Similarity
**Goal:** Identify k-peers and predict ratings in the reduced space.

### 7. Similarity Computation
* **Cosine Similarity**: Compute similarity between reduced user vectors $u_a$ and $u_b$:
  $$sim(a, b) = \frac{u_a \cdot u_b}{\|u_a\| \cdot \|u_b\|}$$
* **Peer Selection**: Identify **k-peers** by selecting only users with $sim > 0$.

### 8. Rating Prediction
* **Weighted Average**: Calculate the predicted rating for user $u$ on item $p$:
  $$\hat{r}_{u,p} = \bar{r}_p + \frac{\sum_{v \in N(u)} sim(u,v) \cdot (r_{v,p} - \bar{r}_v)}{\sum_{v \in N(u)} |sim(u,v)|}$$
* **Baseline**: Use the item mean $\bar{r}_p$ as the baseline for the prediction.

