# PCA Method with Mean-Filling (Optimized)

## Phase 1: Preprocessing & Centering (Steps 1-4)
Instead of physically filling the matrix with means (which wastes memory), we calculate the means and logically assume that any missing value equals the mean.

**1. Calculate Item Means**
* **Goal:** Find the average rating for every item (column) to use for centering.
* **Formula:** For each item $j$:
    $$\mu_j = \frac{\sum_{u \in Observed} R_{u,j}}{|Observed|}$$
* **Optimization:** Ignore missing values; average only what exists.

**2. Implicit Mean-Filling & Centering**
* **Concept:** Standard Mean-Filling sets $R_{missing} = \mu_j$.
* **Centering:** We subtract the mean: $R'_{u,j} = R_{u,j} - \mu_j$.
* **The Optimization Trick:**
    * For observed ratings: $R'_{u,j} = R_{u,j} - \mu_j$
    * For missing ratings: $R'_{u,j} = \mu_j - \mu_j = 0$
    * *Result:* We treat the centered matrix as a sparse matrix where missing entries are exactly **0**.

---

## Phase 2: Covariance Matrix (Steps 5-6)
We calculate the covariance between items. Because missing centered values are 0, we only sum the product when a user rated **both** items.

**3. Compute Covariance Matrix ($\Sigma$)**
* **Input:** Total number of users $M$ (e.g., if you have 12 users, $M=12$).
* **Formula:** For every pair of items $(i, j)$:
    $$Cov(i, j) = \frac{\sum_{u \in Common} (R'_{u,i} \times R'_{u,j})}{M - 1}$$
    * $Common$: The set of users who explicitly rated **both** item $i$ and item $j$.
* **Crucial Note:** Even though we sum only over common users, we divide by $(M - 1)$ because Mean-Filling assumes we have a "complete" dataset of $M$ users (the rest are just zeros).

---

## Phase 3: Eigen Decomposition (Step 7)
Determine the principal components that capture the most information.

**4. Eigen Decomposition**
* **Formula:** Solve $\det(\Sigma - \lambda I) = 0$
* **Output:**
    * Eigenvalues ($\lambda$): Represent the variance captured.
    * Eigenvectors ($V$): Represent the direction of the components.
* **Selection:** Sort Eigenvectors by their Eigenvalues in descending order.
    * *Top 5-Peers:* Select the top 5 vectors ($W_5$).
    * *Top 10-Peers:* Select the top 10 vectors ($W_{10}$).

---

## Phase 4: Dimensionality Reduction (Step 8 & 10)
Project the users from the original item-space into the reduced latent space.

**5. Project Users (Generate Reduced Space)**
* **Goal:** Create a user vector $U_k$ of size $k$ (where $k=5$ or $10$).
* **Formula:**
    $$UserVector_u = R'_u \times W_k$$
* **Optimization:** Since $R'_u$ has **0** for missing items, we only iterate over items the user actually rated:
    $$UserVector_u[dim] = \sum_{j \in Observed} (R_{u,j} - \mu_j) \times W_{j, dim}$$

---

## Phase 5: Prediction (Steps 9 & 11)
Use the users' positions in the reduced space to find similarities and predict the missing target items.

**6. Calculate User Similarity (in Reduced Space)**
* **Goal:** Find how similar Target User ($u$) is to every other user ($v$) using their new low-dim vectors.
* **Formula (Cosine Similarity):**
    $$Sim(u, v) = \frac{UserVector_u \cdot UserVector_v}{||UserVector_u|| \times ||UserVector_v||}$$

**7. Predict Rating**
* **Goal:** Predict rating for Target User $u$ on Item $i$.
* **Formula (Weighted Average):** Select the nearest neighbors ($N$) based on similarity.
    $$\hat{r}_{u,i} = \mu_i + \frac{\sum_{v \in N} Sim(u, v) \times (R_{v,i} - \mu_i)}{\sum_{v \in N} |Sim(u, v)|}$$
    *(Note: Using $\mu_i$ (Item Mean) as the baseline is standard for Item-based filling, but if your lecture uses User Mean $\bar{r}_u$ as the baseline, swap $\mu_i$ for $\bar{r}_u$. Based on the Mean-Filling context, Item Mean is consistent.)*