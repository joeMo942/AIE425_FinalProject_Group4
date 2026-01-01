# Technical Implementation Guide: PCA with Maximum Likelihood Estimation (MLE)

This guide outlines the implementation of the PCA-based recommendation method using MLE logic to predict missing ratings for target items **I1** and **I2**, strictly following the observed-data approach.

---

## Phase 1: Preprocessing & MLE-Based Centering
**Goal:** Prepare the data for covariance estimation without the bias of mean-filling.

* [ ] **Calculate MLE Item Means**: For every item column, compute the mean rating ($\bar{r}_j$) using **only** observed ratings.
    * **Rule**: Ignore missing entries ("?"); do not use filler values.
* [ ] **Compute Centered Deviations**: For every specified rating, subtract its corresponding MLE item mean.
    * **Equation**: $r'_{u,j} = r_{u,j} - \bar{r}_j$.
    * **Logic**: If a rating is missing, the centered value must remain **blank/null**.



---

## Phase 2: Covariance Matrix Generation
**Goal:** Mathematically define item relationships using only co-rated data points.

* [ ] **Pairwise MLE Covariance**: For every unique item pair ($A, B$), calculate the covariance using the unbiased sample formula restricted to co-rated users.
    * **Equation**: $Cov(A, B) = \frac{\sum_{i \in co-rated(A,B)}(A_i - \bar{A})(B_i - \bar{B})}{n_{AB} - 1}$.
    * **Implementation Constraint**: If $n_{AB} = 0$ (no users in common), set the covariance to **0**.
* [ ] **Assemble Matrix ($\Sigma$)**: Construct a square covariance matrix where each entry $\Sigma_{j,k}$ is the result of the pairwise MLE calculation.



---

## Phase 3: Peer Selection & Dimensionality Reduction
**Goal:** Identify neighbors and project users into reduced latent spaces.

* [ ] **Determine Peers**: For target items **I1** and **I2**, sort other items by their covariance values in descending order.
    * Identify the **Top 5-peers** and **Top 10-peers** for both target items.
* [ ] **Eigen-Decomposition**: Solve for the eigenvalues ($\lambda$) and eigenvectors ($p$) of the MLE covariance matrix.
    * **Equation**: $det(\Sigma - \lambda I) = 0$.
* [ ] **Construct Projection Matrix ($W$)**: Create distinct projection matrices based on the selected peer groups.
* [ ] **Reduced Space User Projection**: Project each user into the latent space using the top principal components.
    * **Equation**: $t_i = r'_i \cdot p_1$.
    * **Zero-Padding Rule**: During matrix multiplication, treat missing centered values ("?") as **0**.



---

## Phase 4: Multi-Tiered Rating Prediction
**Goal:** Reconstruct ratings for I1 and I2 using the two distinct peer-group representations.

* [ ] **Top 5-Peer Prediction**:
    * Compute rating predictions using the reduced dimensional space derived from the top 5-peers.
    * **Equation**: $\hat{r}_{u,p} = \bar{r}_p + \hat{x}_{u,p}$, where $\hat{x}_{u,p} \approx t_i \cdot p_p$.
* [ ] **Top 10-Peer Prediction**:
    * Compute rating predictions using the reduced dimensional space derived from the top 10-peers.
    * **Logic**: Repeat the reconstruction process using the expanded peer set to compare predictive accuracy.