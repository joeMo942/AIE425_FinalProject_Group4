# SECTION 2: Live Stream Domain Recommender - Technical Report

## Overview

This section implements a complete recommendation system for Twitch live streamers using three approaches:
1. **Content-Based Filtering** - Recommends based on item features (TF-IDF + numerical)
2. **Collaborative Filtering** - Recommends based on user-user similarity
3. **Hybrid System** - Combines both approaches with weighted scoring

---

## Pipeline Execution Order

```
1. data_preprocessing.py  → processed_ratings.csv
2. fetch_streamer_data.py → streamer_metadata.csv (optional: web scraping)
3. merge_data.py          → final_ratings.csv, final_items.csv
4. eda.py                 → Analysis plots and statistics
5. content_based.py       → content_based_model.pkl
6. collaborative.py       → collaborative_model.pkl
7. hybrid.py              → hybrid_recommendations.csv
```

---

## 1. Data Preprocessing (`data_preprocessing.py`)

**Purpose:** Convert raw Twitch viewing data into user-item ratings.

### Step 1: Load Raw Data
```python
# Input: 100k_a.csv with columns:
# [user_id, stream_id, streamer_username, time_start, time_stop]
```

### Step 2: Clean Usernames
```python
streamer_username = streamer_username.lower().strip()
```

### Step 3: Calculate Watch Duration
**Formula:**
$$\text{duration\_minutes} = (\text{time\_stop} - \text{time\_start}) \times 10$$

*The dataset uses 10-minute intervals.*

### Step 4: Aggregate Interactions
For each `(user_id, streamer_username)` pair:
$$\text{total\_minutes} = \sum \text{duration\_minutes}$$

### Step 5: Convert to 1-5 Ratings (Log-Transformed Min-Max Normalization)

> [!IMPORTANT]
> We use **log transformation** to compress the long-tail distribution. This fixes the issue where short watch times (e.g., 30 mins) were unfairly rated 1-star when compared to outliers (e.g., 100 hours).

**Step 5a:** Apply log transformation to compress outliers:
$$\text{log\_minutes}_{u,i} = \log(1 + \text{total\_minutes}_{u,i})$$

**Step 5b:** Calculate user-specific min/max on LOG values:
$$\text{log\_min}_u = \min_{i} (\text{log\_minutes}_{u,i})$$
$$\text{log\_max}_u = \max_{i} (\text{log\_minutes}_{u,i})$$

**Step 5c:** Normalize LOG values to 0-1:
$$\text{normalized}_{u,i} = \frac{\text{log\_minutes}_{u,i} - \text{log\_min}_u}{\text{log\_max}_u - \text{log\_min}_u}$$

**Step 5d:** Map to 1-5 ratings:
| Normalized Range | Rating |
|------------------|--------|
| 0.0 - 0.2 | 1 |
| 0.2 - 0.4 | 2 |
| 0.4 - 0.6 | 3 |
| 0.6 - 0.8 | 4 |
| 0.8 - 1.0 | 5 |

**Rationale:** A 30-minute session now maps closer to 3/5 rating rather than 1/5.

### Output
- `processed_ratings.csv`: `[user_id, streamer_username, rating, total_minutes]`

---

## 2. Fetch Streamer Data (`fetch_streamer_data.py`)

**Purpose:** Web scrape TwitchTracker for streamer metadata (optional).

### Process
1. Load list of streamer usernames from `unique_streamers.txt`
2. For each streamer, scrape TwitchTracker page
3. Extract: rank, followers, avg_viewers, games, language, type
4. Handle rate limiting with random delays
5. Save checkpoint every 100 streamers

### Output
- `streamer_metadata.csv`: Streamer features for content-based filtering

---

## 3. Merge Data (`merge_data.py`)

**Purpose:** Join ratings with item metadata, filter to common items.

### Step 1: Load Ratings
```python
df_ratings = pd.read_csv("processed_ratings.csv")
```

### Step 2: Load Item Metadata
Try scraped data first, fallback to `datasetV2.csv` archive.

### Step 3: Create Text Features
```python
text_features = f"{type} {1st_game} {2nd_game} {language}"
```

### Step 4: Inner Join
```python
df_merged = df_ratings.merge(df_items, on='streamer_username', how='inner')
```
*Only keeps streamers with BOTH ratings AND metadata.*

### Step 5: Validation
- Users ≥ 5,000 ✓
- Items ≥ 500 ✓
- Interactions ≥ 50,000 ✓

### Output
- `final_ratings.csv`: `[user_id, streamer_username, rating]`
- `final_items.csv`: `[streamer_username, text_features, language, rank, avg_viewers, ...]`

---

## 4. Exploratory Data Analysis (`eda.py`)

**Purpose:** Analyze dataset characteristics for the assignment report.

### Analyses Performed

#### 4.1 Basic Statistics
- Number of users, items, interactions
- **Sparsity Formula:**
$$\text{Sparsity} = 1 - \frac{|\text{Ratings}|}{|\text{Users}| \times |\text{Items}|} \times 100\%$$

#### 4.2 Rating Distribution
- Count ratings 1-5
- Visualize histogram

#### 4.3 User Activity (Long-Tail Analysis)
- Min/Max/Mean/Median ratings per user
- **Long-tail check:**
$$\text{Top 10\% users contribute } X\% \text{ of ratings}$$

#### 4.4 Item Popularity (Long-Tail Analysis)
- Min/Max/Mean/Median ratings per item
- Top 10 most popular streamers

### Output
- `Sec2_basic_statistics.txt`
- `Sec2_rating_distribution.png`
- `Sec2_user_activity.png`
- `Sec2_item_popularity.png`

---

## 5. Content-Based Filtering (`content_based.py`)

**Purpose:** Recommend items similar to what the user liked based on item features.

### Step 1: Prepare Item Features

#### 1a: TF-IDF on Text Features
Extract term frequencies from `text_features` column:
$$\text{TF-IDF}_{t,d} = \text{TF}_{t,d} \times \log\left(\frac{N}{df_t}\right)$$

Where:
- $\text{TF}_{t,d}$ = Term frequency of term $t$ in document $d$
- $N$ = Total number of documents
- $df_t$ = Number of documents containing term $t$

**Parameters:**
- `max_features=500`
- `ngram_range=(1, 2)` (unigrams + bigrams)
- `stop_words='english'`

#### 1b: Normalize Numerical Features
Apply Min-Max scaling to `rank`, `avg_viewers`, `followers`:
$$x_{normalized} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

#### 1c: Combine Features
```python
combined = [TF-IDF * 0.7, Numerical * 0.3]  # 70% text, 30% numerical
```

### Step 2: Compute Item Similarity
**Cosine Similarity:**
$$\text{sim}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}$$

### Step 3: Build User Profiles (with Time Decay)

> [!NOTE]
> **Implements Lecture 13**: User Profile = Centroid (Weighted Average) of the vectors of items the user liked.

For each user $u$, the profile is a **time-decayed weighted average** of rated item features:

**Time Decay Formula:**
$$\text{weight}_{u,i} = r_{u,i} \times \frac{1}{1 + \text{decay\_rate} \times \text{days\_since}}$$

**User Profile (Centroid):**
$$\text{UserProfile}_u = \frac{\sum_{i \in \text{rated}(u)} \text{weight}_{u,i} \times \text{ItemFeatures}_i}{\sum_{i \in \text{rated}(u)} \text{weight}_{u,i}}$$

**Parameters:**
- `decay_rate = 0.01` (recent interactions weighted higher)

### Step 4: Generate Recommendations
For user $u$, find items most similar to their profile:
$$\text{score}_i = \text{cosine\_similarity}(\text{UserProfile}_u, \text{ItemFeatures}_i)$$

Return top-N items not already rated.

### Step 5: Evaluation
**Hit Rate @ N:**
$$\text{Hit Rate} = \frac{\text{Number of users where highest-rated item is in top-N}}{|\text{Test Users}|}$$

### Output
- `content_based_model.pkl`: Saved model for hybrid use

---

## 6. Collaborative Filtering (`collaborative.py`)

**Purpose:** Recommend items based on similar users' preferences.

### Step 1: Create User-Item Matrix
Sparse matrix $R$ where $R_{u,i} = $ rating of user $u$ on item $i$ (0 if unrated).

### Step 2: Compute User Means
For each user $u$:
$$\bar{r}_u = \frac{\sum_{i \in \text{rated}(u)} r_{u,i}}{|\text{rated}(u)|}$$

### Step 3: Compute User Similarity
**Cosine Similarity between user rating vectors:**
$$\text{sim}(u, v) = \frac{\sum_i r_{u,i} \cdot r_{v,i}}{\sqrt{\sum_i r_{u,i}^2} \times \sqrt{\sum_i r_{v,i}^2}}$$

### Step 4: Predict Ratings
For user $u$ on item $i$, using k-nearest neighbors:

$$\hat{r}_{u,i} = \bar{r}_u + \frac{\sum_{v \in N_k(u)} \text{sim}(u,v) \times (r_{v,i} - \bar{r}_v)}{\sum_{v \in N_k(u)} |\text{sim}(u,v)|}$$

Where:
- $N_k(u)$ = Top-k most similar users to $u$ who rated item $i$
- $\bar{r}_u$ = Mean rating of user $u$
- $\bar{r}_v$ = Mean rating of neighbor $v$

### Step 5: Generate Recommendations (K-NN)
Predict ratings for all unrated items, return top-N.

### Step 6: Evaluation
- **Hit Rate @ N**: Same as content-based
- **RMSE (on hits):**
$$\text{RMSE} = \sqrt{\frac{\sum (\hat{r} - r)^2}{n}}$$

---

### SVD Matrix Factorization (Assignment Part 3, Section 8.2)

> [!IMPORTANT]
> SVD addresses the **sparsity problem** by learning latent factors that capture user preferences and item characteristics.

**Step 1: Normalize Matrix**
$$R_{normalized} = R - \bar{r}_u \text{ (for each row)}$$

**Step 2: Compute Truncated SVD**
$$U, \Sigma, V^T = \text{svds}(R_{normalized}, k=20)$$

**Step 3: Reconstruct Predictions**
$$R_{pred} = U \cdot \text{diag}(\Sigma) \cdot V^T + \bar{r}_u$$

**Parameters:**
- `k = 20` latent factors

### Output
- `collaborative_model.pkl`: K-NN model metadata
- `user_similarity.npy`: User similarity matrix
- `user_item_matrix.npz`: Sparse rating matrix
- `svd_predictions.npy`: SVD prediction matrix
- `svd_model.pkl`: SVD model components (U, Σ, V^T)

---

## 7. Hybrid Recommender (`hybrid.py`)

**Purpose:** Combine content-based and collaborative filtering using **Cascade Hybrid Strategy (Option C)**.

### Step 1: Load Pre-trained Models
- `content_based_model.pkl`
- `collaborative_model.pkl`
- `svd_predictions.npy`
- `user_similarity.npy`

### Step 2: Cascade Hybrid Strategy

> [!IMPORTANT]
> **Option C: Cascade Hybrid** - Content-based filters to top-50 candidates, then SVD ranks final top-10.

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   ALL 396 ITEMS     │ --> │  CB → TOP 50       │ --> │  SVD → TOP 10       │
│   (unrated items)   │     │  (fast filtering)   │     │  (accurate ranking) │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

| Stage | System | Role |
|-------|--------|------|
| **Stage 1** | Content-Based | Filter all items to top-50 candidates |
| **Stage 2** | SVD | Rank 50 candidates, return top-10 |

### Step 3: Cold-Start Handling

| User Rating Count | Strategy | Rationale |
|-------------------|----------|-----------|
| **0 ratings** (cold-start) | Global Popularity | No user data available |
| **< 5 ratings** | Content-Based only | Not enough data for CF/SVD |
| **≥ 5 ratings** | **Cascade (CB → SVD)** | Both systems reliable |

### Step 4: Why Cascade?
1. **Efficiency**: SVD only scores 50 items instead of all 396
2. **Quality**: CB ensures candidates are content-relevant, SVD refines by user preference
3. **Coverage**: CB covers items with no ratings, SVD handles preference nuances

### Output
- `hybrid_recommendations.csv`: Recommendations with CB and SVD scores

---

## Summary of Formulas

| Component | Formula |
|-----------|---------|
| **Log Transform** | $\log(1 + x)$ |
| **Watch Duration** | $d = (\text{stop} - \text{start}) \times 10$ |
| **Rating Normalization** | $\text{norm} = \frac{\log(x) - \log(\min)}{\log(\max) - \log(\min)}$ |
| **Sparsity** | $S = 1 - \frac{R}{U \times I}$ |
| **TF-IDF** | $\text{TF} \times \log(N / df)$ |
| **Cosine Similarity** | $\frac{A \cdot B}{\|A\| \|B\|}$ |
| **Time Decay** | $w = r \times \frac{1}{1 + \text{decay} \times \text{days}}$ |
| **User Profile** | $\frac{\sum w_i \times f_i}{\sum w_i}$ |
| **CF Prediction (K-NN)** | $\bar{r}_u + \frac{\sum \text{sim} \times (r_v - \bar{r}_v)}{\sum |\text{sim}|}$ |
| **SVD Reconstruction** | $R_{pred} = U \cdot \Sigma \cdot V^T + \bar{r}_u$ |
| **Hybrid Score** | $\alpha \times CB + (1-\alpha) \times SVD$ |

---

## Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| `data_preprocessing.py` | ~250 | Log-transformed ratings conversion |
| `fetch_streamer_data.py` | ~490 | Web scrape streamer metadata |
| `merge_data.py` | ~240 | Join ratings + metadata |
| `eda.py` | ~270 | Exploratory analysis |
| `content_based.py` | ~600 | TF-IDF + Time Decay recommendations |
| `collaborative.py` | ~600 | K-NN + SVD matrix factorization |
| `hybrid.py` | ~510 | Switching hybrid system |
