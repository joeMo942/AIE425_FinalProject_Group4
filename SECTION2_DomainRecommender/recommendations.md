# Recommendations for Improving Recommender System Accuracy

Based on the analysis of `content_based.py` and `collaborative.py`, here are specific recommendations to improve the accuracy of your results. These suggestions focus on hyperparameter tuning and minor logical adjustments that don't require rewriting the core architecture.

## 1. Content-Based Recommendations (`content_based.py`)

### A. Feature Engineering (TF-IDF)
The current `TfidfVectorizer` settings are conservative. Capturing more granular text features can improve similarity matching.

*   **Increase `max_features`**: Currently **500**.
    *   **Recommendation**: Increase to **1000** or **2000**.
    *   **Why**: 500 terms might be too few to distinguish between nuanced streamer genres or game types.
*   **Adjust `min_df`**: Currently **1**.
    *   **Recommendation**: Increase to **2** or **5**.
    *   **Why**: Terms appearing in only 1 document are often noise or typos and don't help with generalization.
*   **N-grams**: Currently `(1, 2)`.
    *   **Recommendation**: Keep as is, this is good.

### B. Feature Weighting
The code currently hardcodes the balance between text and numerical features:
```python
# Current
tfidf_matrix.toarray() * 0.7
numerical_normalized * 0.3
```
*   **Recommendation**: Treat these weights as hyperparameters to tune.
    *   Try **(0.9, 0.1)** if text (game descriptions, tags) is the primary driver of similarity.
    *   Try **(0.5, 0.5)** if popularity metrics (viewers, rank) are equally important.
    *   **Tip**: If you find recommendations are just showing "popular" streamers rather than "relevant" ones, **decrease** the numerical weight (e.g., to 0.1 or 0.2).

### C. Numerical Feature Scaling
The code uses `MinMaxScaler` on `['rank', 'avg_viewers', 'followers']`.
*   **Issue**: Viewership and follower counts often follow a **Power Law** distribution (a few streamers have millions, most have few). Linear MinMax scaling will squash 99% of your data into a tiny range (e.g., 0.0 to 0.05), making them indistinguishable.
*   **Recommendation**: Apply a **Log Transformation** before scaling.
    *   Change: `np.log1p(df[col])` before `scaler.fit_transform`.
    *   This will make the "popularity" metric much more meaningful for distinguishing between small, medium, and large streamers.

### D. User Profile Decay Rate
The time decay is set to `decay_rate = 0.01`.
*   **Recommendation**: Tuning this depends on user behavior.
    *   If user preferences change constrainedly (e.g., they switch games every week), **increase** to `0.05`.
    *   If user preferences are stable (e.g., "I always watch FPS"), **decrease** to `0.001` or remove it to use their full history equally.

---

## 2. Collaborative Filtering (`collaborative.py`)

### A. k-NN Neighbors
*   **Current**: `K_NEIGHBORS = 50`.
*   **Recommendation**: Experiment with **lower values (20-30)**.
    *   **Why**: 50 neighbors might be "smoothing out" specific tastes too much. If users have unique tastes, a smaller neighborhood (20) often yields more accurate, albeit riskier, predictions.

### B. SVD Latent Factors
*   **Current**: `k = 20`.
*   **Recommendation**: **Increase k to 50 or 100**.
    *   **Why**: 20 latent factors is quite low for capturing the complexity of user-item interactions, roughly reducing all streamer characteristics to just 20 dimensions. Increasing this allows the model to capture more subtle patterns (e.g., "likes cozy morning streams" vs "likes competitive evening streams").
    *   **Trade-off**: Higher `k` increases risk of overfitting, but 50-100 is standard for most recommendation datasets.

### C. Similarity Metric
*   **Current**: Cosine Similarity.
*   **Recommendation**: Since the prediction code manually implements "Deviation from Mean" (which effectively makes it Pearson Correlation), the logic is sound.
    *   However, consider adding **Shrinkage**. If two users only have 1 item in common and rated it the same, their similarity is 1.0, which is unreliable.
    *   **Fix**: Weight the similarity by `min(n_common_items / threshold, 1.0)`. This down-weights similarities based on very sparse overlaps.

### D. Data Sparsity
*   **Observation**: The code has a check `if n_users <= 20000`.
*   **Recommendation**: If you are running on a machine with sufficient RAM (16GB+), you can increase this limit to allow full matrix computation, which is often faster than the on-demand loop for density < 1%.

---

## Summary of Quick Wins (Minimal Edits)

1.  **Change `content_based.py`**:
    *   `max_features` -> **1000**
    *   `min_df` -> **2**
    *   Try decreasing numerical weight to **0.1** to improve relevance over popularity.
    *   Add `np.log1p()` to numerical features.

2.  **Change `collaborative.py`**:
    *   `K_NEIGHBORS` -> **30**
    *   `compute_svd(..., k=50)` (Change from 20 to 50).

These changes should yield immediate improvements in recommendation relevance and accuracy metrics (RMSE/Hit Rate).
