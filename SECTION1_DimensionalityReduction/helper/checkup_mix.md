# Technical Implementation Guide: PCA with Mean-Filling (I1 & I2 Focus)

This guide provides the mathematical framework and procedural steps for implementing the PCA-based recommendation method specifically for target items **I1** and **I2**, adhering to the architectural structure defined in the project specification.

---

## Phase 1: Data Augmentation & Preprocessing
**Goal:** Create a complete, centered matrix $R'_f$ to allow for covariance computation.

1. **Calculate Target Item Means**:
   * Compute the average rating for items **I1** and **I2** using only available ratings.
   * **Equation**: $\bar{r}_j = \frac{1}{|U_j|} \sum_{u \in U_j} r_{u,j}$, where $U_j$ is the set of users who rated item $j$.

2. **Mean-Fill Target Items**:
   * Replace all missing entries in columns **I1** and **I2** with their respective calculated means ($\bar{r}_{I1}, \bar{r}_{I2}$).

3. **Calculate All Item Averages**:
   * Calculate the mean rating for every item in the dataset after the target items have been filled.

4. **Compute Centered Deviations**:
   * For every entry in the augmented matrix, subtract the item's mean to center the data around zero.
   * **Equation**: $r'_{u,j} = r_{u,j} - \bar{r}_j$.



---

## Phase 2: Covariance Matrix Generation
**Goal:** Quantify the statistical relationships between all item pairs using an unbiased estimator.

5. **Compute Pairwise Covariance**:
   * Calculate the covariance between every pair of items $X$ and $Y$ using the $n-1$ sample formula.
   * **Equation**: $Cov(X, Y) = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{n - 1}$.

6. **Generate Covariance Matrix ($\Sigma$)**:
   * Populate a square matrix where each element $\Sigma_{j,k}$ is the covariance between $Item_j$ and $Item_k$.
   * **Note**: The diagonal $\Sigma_{j,j}$ must represent the **variance** of item $j$.



---

## Phase 3: Peer Selection & Dimensionality Reduction
**Goal:** Identify neighbors for I1/I2 and reduce the dimensionality for the Top 5-peer group.

7. **Determine k-Peers**:
   * For target items **I1** and **I2**, sort other items by their covariance values in descending order.
   * Identify the **Top 5-peers** and **Top 10-peers** for each target item.

8. **Reduced Dimensional Space (Top 5-Peers)**:
   * Perform Eigen-decomposition on the covariance matrix to find Eigenvalues ($\lambda$) and Eigenvectors ($p$).
   * **Equation**: $det(\Sigma - \lambda I) = 0$.
   * Construct a projection matrix $W$ by selecting the eigenvectors associated with the largest principal components.
   * Project each user vector into the reduced space: $u_i = r'_i W$.



---

## Phase 4: Rating Prediction
**Goal:** Calculate the final predictions for the original missing values of I1 and I2.

9. **Compute Predictions**:
   * Calculate User-User similarity ($sim(u,v)$) using **Cosine Similarity** on the reduced vectors from Step 8.
   * **Equation**: $sim(u, v) = \frac{u_u \cdot u_v}{\|u_u\| \cdot \|u_v\|}$.
   * Apply the weighted average prediction formula using the item mean as the baseline.
   * **Equation**: $\hat{r}_{u,p} = \bar{r}_p + \frac{\sum_{v \in N(u)} sim(u,v) \cdot (r_{v,p} - \bar{r}_v)}{\sum_{v \in N(u)} |sim(u,v)|}$.