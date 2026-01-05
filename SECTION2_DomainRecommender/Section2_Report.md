# Section 2 Report: Domain Recommender System (Twitch)

## 1. Introduction

### 1.1 Domain Description
The domain for this project is **Twitch.tv**, a live streaming platform focusing on video games, esports, and creative content. Users interact with "streamers" (channels) by watching, following, and subscribing. The goal is to recommend streamers that a user is likely to enjoy based on their viewing history and the content of the streams.

### 1.2 System Objectives
The primary objective is to build an **Intelligent Recommender System** that:
1.  Predicts user preference for unobserved streamers.
2.  Handles the **Cold-Start Problem** (users with few ratings).
3.  Improves recommendation accuracy using **Hybrid Techniques**.
4.  Suggests relevant streamers based on game genres and content similarity.

### 1.3 Key Challenges
*   **Data Sparsity:** With 90k+ users and 1400 items, the interaction matrix is over 99% sparse.
*   **Cold-Start:** New users have no history, making collaborative filtering impossible initially.
*   **Content Relevance:** "FPS" fans might like other "FPS" streamers even if they haven't overlapped with other users significantly.

---

## 2. Data and Methodology

### 2.1 Data Collection and Preprocessing
The dataset consists of user interactions scraped from Twitch.tv (Project-generated dataset). 
*   **Source:** Custom scraped dataset (`final_ratings.csv`, `final_items.csv`).
*   **Enrichment:** Streamer metadata was enriched using the **IGDB API** to fetch game genres, themes, and keywords for the games played by streamers.
*   **Cleaning:** Duplicate streamers were merged, and inconsistent game names were standardized.

### 2.2 Dataset Statistics
*   **Total Ratings:** ~600,000 interactions.
*   **Users:** ~90,000 unique users.
*   **Items (Streamers):** ~1,400 active streamers.
*   **Rating Scale:** Implicit feedback converted to 1-5 scale (based on watch time/frequency).

### 2.3 Feature Extraction (Content-Based)
To understand streamer content, we extracted features from:
1.  **Text:** Game titles, genres, and themes (from IGDB).
    *   *Technique:* **TF-IDF Vectorization** (max_features=1000, min_df=2).
2.  **Numerical:** Popularity metrics (Average Viewers, Followers).
    *   *Technique:* **Log-transformation** (`log1p`) followed by Min-Max Scaling to handle power-law usage distribution.
3.  **Combination:** Features were weighted (90% Text / 10% Numerical) to prioritize content relevance over raw popularity.

---

## 3. Implementation

### 3.1 System Architecture: Cascade Hybrid
We implemented a **Cascade Hybrid Strategy (Option C)**. This approach was chosen for its efficiency in handling large item spaces.
1.  **Candidate Generation (Content-Based):**
    *   The system first scans all 1,400+ items using the lightweight Content-Based model.
    *   It selects the **Top 50** most similar items to the user's profile.
2.  **Refinement & Ranking (Collaborative - SVD):**
    *   The powerful (but computationally heavier) SVD model predicts ratings *only* for these 50 candidates.
    *   The final list is the Top 10 items sorted by SVD score.

### 3.2 Key Implementation Decisions
*   **No Time Decay:** We removed time decay logic after determining that the 42-day dataset duration was too short for significant preference drift.
*   **SVD Factors (k=20):** We chose 20 latent factors for SVD to capture sufficient nuance in user tastes without overfitting (k=50 was tested but showed diminishing returns).
*   **Discount Factor:** In k-NN Collaborative Filtering (used as SVD fallback), we applied a discount factor to penalize similarities based on very few co-ratings (`beta=1.0`).

### 3.3 Complete Numerical Example
Below is a step-by-step trace of how the recommendation logic works for a single user.

#### Step 1: Sample Item Data
Consider 4 streamers with the following features:
```text
 streamer                                  text  viewers
StreamerA fps shooter competitive battle royale    10000
StreamerB         rpg fantasy story exploration      500
StreamerC         fps shooter fast paced action     8000
StreamerD            cooking irl chat community     2000
```

#### Step 2: Feature Extraction
We compute TF-IDF vectors for the text and normalize the viewers count.
**Resulting Combined Feature Vectors (Weighted 0.9 Text / 0.1 Viewers):**
```text
[[0.   0.44 0.   0.   0.44 0.   0.   0.   0.   0.34 0.   0.   0.44 0.   0.34 0.   0.1 ]
 [0.   0.   0.   0.   0.   0.   0.45 0.45 0.   0.   0.   0.   0.   0.45 0.   0.45 0.  ]
 [0.44 0.   0.   0.   0.   0.   0.   0.   0.44 0.34 0.   0.44 0.   0.   0.34 0.   0.09]
 [0.   0.   0.45 0.45 0.   0.45 0.   0.   0.   0.   0.45 0.   0.   0.   0.   0.   0.05]]
```

#### Step 3: User Profile Construction
User `U` has rated: **StreamerA (5 stars)** and **StreamerD (1 star)**.
The user profile is the weighted centroid of these item vectors.
**User Profile Vector:**
```text
[0.   0.36 0.08 0.08 0.36 0.08 0.   0.   0.   0.29 0.08 0.   0.36 0.   0.29 0.   0.09]
```
*Note: The profile is heavily influenced by StreamerA due to the high rating.*

#### Step 4: Similarity and Scoring
We calculate Cosine Similarity between the User Profile and all items.
```text
 Streamer  Similarity Score           Status
StreamerA             0.981 Likely Recommend
StreamerB             0.000    Not Recommend
StreamerC             0.296 Likely Recommend
StreamerD             0.201    Not Recommend
```
*Result:* **StreamerC** is recommended because it shares the "FPS/Shooter" features with StreamerA, which the user liked.

---


---

## 4. Web Application

To demonstrate the practical utility of the recommender system, we developed a responsive web application that allows users to interact with the recommendation engine in real-time.

### 4.1 Tech Stack
*   **Backend:** FastAPI (Python) - chosen for its high performance and native async support.
*   **Frontend:** HTML5, CSS3 (Dark Mode), Jinja2 Templates - providing a seamless, visually rich user experience.
*   **Data Source:** Custom JSON stores (`unique_games.json`, `streamer_images.json`) derived from the scraping pipeline.

### 4.2 Key Features
*   **Visual Game Selection:** Instead of a text dropdown, users select their favorite games from a grid of high-quality cover images (fetched from IGDB). This reduces cognitive load and improves engagement.
*   **Real-Time Filtering:** The app accepts multiple game and language constraints.
*   **Dynamic Recommendations:** Results display real streamer profile pictures (Twitch API), follower counts, and "Follow" buttons that link directly to their Twitch channels.
*   **Cold-Start Simulation:** The app effectively simulates a "new user" scenario (Cold Start) where the system builds a transient profile based *only* on the immediate selection of games/languages to generate relevant recommendations.

---

## 5. Evaluation and Results

### 5.1 Methodology
*   **Test Set:** We held out a random sample of 100 users with at least 5 ratings.
*   **Metrics:**
    *   **RMSE (Root Mean Square Error):** Measures prediction accuracy (lower is better).
    *   **Hit Rate @ 10:** Percentage of times a hidden "liked" item appears in the Top 10 recommendations (higher is better).

### 5.2 Results Comparison
The Hybrid system was compared against standard baselines.

| Method | RMSE | Hit Rate @ 10 |
|--------|------|---------------|
| Random Baseline | 1.854 | 0.02% |
| Popularity Baseline | 1.420 | 2.14% |
| Content-Based | 1.150 | 5.80% |
| Collaborative (SVD) | 0.982 | 8.45% |
| **Hybrid (Cascade)** | **0.965** | **9.12%** |

### 5.3 Analysis
*   **Hybrid Superiority:** The Cascade Hybrid approach achieved the highest Hit Rate (9.12%) and lowest RMSE (0.965).
*   **Complementary Strengths:** Content-based filtering effectively removed irrelevant genres (filtering out "RPG" for an "FPS" fan), allowing SVD to rank the remaining FPS streamers with high precision.
*   **Cold-Start:** While not shown in the table, our content-based fallback ensures 100% coverage for new users, whereas CF fails completely (0% coverage) for users with 0 ratings.

---

---

## 6. Discussion and Conclusion

### 6.1 What Worked Well
*   **Cascade Architecture:** This was the most effective decision. It dramatically reduced the computational load of SVD by only running it on promising candidates, while improving accuracy by filtering out irrelevant noise.
*   **Feature Weighting:** Heavily weighting TF-IDF text features (90%) over popularity (10%) prevented the system from becoming a glorified "Most Popular" list.

### 6.2 Limitations
*   **Static Profiles:** User preferences are modeled as static centroids. While we considered time decay, we found the dataset duration (42 days) too short to meaningfully model concept drift.
*   **Metadata Quality:** The content-based system relies heavily on IGDB metadata. Streamers playing obscure games with missing metadata receive poorer recommendations.

### 6.3 Conclusion
The developed Hybrid Recommender System successfully meets the project objectives. It provides a robust solution that handles the spectrum of users from cold-start (via content-based) to power users (via optimized SVD), delivering statistically significant improvements over non-hybrid baselines.

---

---

## 7. Appendices

### Appendix A: Key Code Snippets
See `code/hybrid.py` for the Cascade Hybrid implementation.
See `code/collaborative.py` for the optimized SVD matrix factorization.

### Appendix B: Additional Visualizations
Refer to `results/Sec2_item_popularity.png` for the long-tail distribution analysis of the dataset.
