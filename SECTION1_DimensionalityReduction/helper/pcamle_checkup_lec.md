# PCA Strategy: Maximum Likelihood Estimation (MLE) Workflow

Based on the core principles of the lectures, the MLE method is designed to solve the "Bias Problem" found in traditional mean-filling. By ignoring missing data instead of inventing it with averages, we preserve the true strength of item relationships.

---

## 1. Data Centering (Preprocessing)
The goal is to calculate how much a user's rating differs from the average behavior for that item.

* **Calculate MLE Item Means**: For every product, calculate the average rating based **only** on the users who actually provided a score.
* **The "Ignore" Rule**: Do not use zeros or filler values for missing ratings ("?"); simply skip them in the average calculation.
* **Center the Matrix**: Subtract the calculated item mean from every existing rating.
* **Equation**: $X'_i = P1_i - \bar{P1}$.
* **Crucial Handling**: If a rating is missing, the centered value remains **blank**. This prevents the data from being biased at the source.



---

## 2. The MLE Covariance Matrix
This step builds a map of how items are statistically linked to one another.

* **Co-rated Pairings**: To find the relationship between Item A and Item B, you only look at users who have provided ratings for **both**.
* **MLE-Style Covariance Equation**: Use the unbiased sample formula restricted to co-rated data:
    $$Cov(A, B) = \frac{\sum_{i \in co-rated(A,B)}(A_i - \bar{A})(B_i - \bar{B})}{n_{AB} - 1}$$
* **Resulting Logic**: Because we don't use "fake" filled data, the covariance reflects reality. For example, the P2-P3 link remains strong at **12**, while mean-filling would have incorrectly diluted it to **3.27**.



---

## 3. Dimensionality Reduction (PCA Engine)
We identify the "Principal Components"—the dominant hidden patterns in user behavior.

* **Eigenvalues ($\lambda$)**: These tell us how much information each pattern captures.
* **Equation**: Solve $det(\Sigma - \lambda I) = 0$.
* **Lecture Insight**: The first pattern ($\lambda_1$) is so strong it explains **94.3%** of all user variation in the dataset.
* **Eigenvectors ($p$)**: This is the "recipe" or weight given to each item within that pattern.
* **Equation**: Solve $(\Sigma - \lambda I)p = 0$.
* **Lecture Recipe (PC1)**: $0.41 \cdot P1 + 0.59 \cdot P2 + 0.69 \cdot P3$.



---

## 4. User Projection & Reconstruction (Prediction)
We use the identified pattern to "fill in the blanks" for missing ratings.

* **User Score ($t_i$)**: We calculate a score for each user based on the pattern.
* **Handling Missing Data**: If a user is missing P3, we calculate their score using only their known ratings for P1 and P2.
* **Projection Equation**: $t_i = 0.408(X'_i) + 0.594(Y'_i)$.
* **Reconstructing the Rating**: We add the item's average back to the predicted deviation to get the final score.
* **Equation**: $\hat{r}_{u,p} = \bar{r}_p + (t_i \cdot p_p)$.
* **Final Example**: For User 5, the model predicts a rating of roughly **3** for the missing item P3.