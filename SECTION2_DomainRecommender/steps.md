Here is the step-by-step guide to transforming your raw Twitch datasets into the structured format required for your assignment.

### **Phase 0: The Strategy**

You need to build **three** specific tables (DataFrames) to satisfy the assignment requirements:

1. **`df_ratings`**: The User-Item Interaction matrix (from Dataset 2).
2. **`df_items`**: The Content features for Streamers (from Dataset 1 + APIs).
3. **`df_users`**: User profiles (derived from `df_ratings` aggregation).

---

### **Phase 1: Process Interactions (Dataset 2)**

*Source: `100k.csv` (The 100k subset is perfect for this assignment; 15M is too large for local processing).*

**Goal:** Convert "Time Start" and "Time Stop" into a **1-5 Rating**.

1. **Load Data:** Read `100k.csv`.
2. **Clean Usernames:** Convert `Streamer Username` to lowercase and remove leading/trailing whitespace. This is crucial for merging later.
3. **Calculate Duration:**
* The dataset uses "10-minute intervals."
* Formula: 



4. **Aggregate Duplicates:**
* Users might watch the same streamer multiple times.
* **Group By:** `[User ID, Streamer Username]`
* **Aggregation:** Sum the `Duration`. (Total time a user spent watching that streamer).


5. **Create Implicit Ratings (1-5 Scale):**
* You cannot arbitrary decide 60 mins = 5 stars. You must use **Quantiles (`pd.qcut`)** to distribute ratings fairly based on the data distribution.
* *Logic:*
* Top 20% of watch times  **5**
* Next 20%  **4**
* ...
* Bottom 20%  **1**




6. **Filter:** Drop any interactions where Rating is NaN or 0.

**Outcome:** `df_ratings` with columns `['user_id', 'streamer_username', 'rating', 'total_minutes']`.

---

### **Phase 2: Process Item Features (Dataset 1)**

*Source: Twitch Tracker Dataset*

**Goal:** Create the metadata vector for the **Content-Based Recommender**.

1. **Load Data:** Read the Twitch Tracker CSV.
2. **Clean Usernames:** Apply the exact same cleaning (lowercase, strip) to the `name` column as you did in Phase 1.
3. **Select & Rename Columns:**
* `name`  `streamer_username` (Join Key)
* `1st most streamed game`  `category` (Categorical Feature)
* `language`  `language` (Categorical Feature)
* `average viewers`  `popularity_score` (Numerical Feature)
* `stream duration`  `avg_stream_len` (Numerical Feature)


4. **Normalize Numerical Features:**
* Use Min-Max Scaling on `popularity_score` and `avg_stream_len` so they are between 0 and 1.



**Outcome:** `df_items_base` containing the statistical metadata.

---

### **Phase 3: Text Enrichment (API Integration)**

*Source: Twitch API / IGDB API*

**Goal:** The assignment **requires** "Text Feature Extraction" (TF-IDF). Your current datasets are mostly numbers. You need text descriptions.

1. **Identify Missing Data:**
* Get the list of unique `streamer_usernames` from your `df_items_base`.


2. **Fetch Streamer Bios (Twitch API):**
* Endpoint: `GET https://api.twitch.tv/helix/users?login=ninja`
* Extract: `description` (The channel bio).


3. **Fetch Game Summaries (IGDB API):**
* Take the `1st most streamed game` column.
* Endpoint: `GET https://api.igdb.com/v4/games`
* Extract: `summary` or `storyline`.


4. **Create the "Document":**
* Create a new column called `text_features`.
* Combine fields: `text_features = Streamer Bio + " " + Game Name + " " + Game Summary`.
* *Example:* "Pro FPS player... playing Valorant... A 5v5 character-based tactical shooter..."



**Outcome:** `df_items_enriched` with a rich text column ready for TF-IDF.

---

### **Phase 4: Merging & Final Filtering**

**Goal:** Create the final consistent datasets.

1. **Merge Items:**
* Perform an **Inner Join** between `df_ratings` and `df_items_enriched` on `streamer_username`.
* *Why Inner Join?* You can only recommend streamers for whom you have metadata (for content-based) AND ratings (for collaborative filtering).


2. **Sparsity Check (Assignment Requirement 2.2):**
* Count distinct users and items.
* **Assignment Rule:** Must have > 5,000 users and > 500 items.
* If you have too many users (e.g., 100k), you can random sample down to 10k-20k to make computations faster, but keep the assignment minimums in mind.


3. **Cold-Start Split:**
* Identify users with  ratings. Flag them as `is_cold_start`.
* Set them aside for the specific "Cold Start" evaluation section (Part 3, Section 10).



---

### **Summary of Final Data Structure**

You will end up with two main files to start coding the assignment:

**1. `final_ratings.csv**`
| user_id | streamer_username | rating (1-5) |
| :--- | :--- | :--- |
| user_123 | shroud | 5 |
| user_456 | lirik | 3 |

**2. `final_items.csv**`
| streamer_username | text_features (Bio+Game) | popularity_score (0-1) | language |
| :--- | :--- | :--- | :--- |
| shroud | "FPS legend... Valorant..." | 0.98 | EN |
| lirik | "Variety streamer... GTA V..." | 0.85 | EN |

Would you like me to write the **Python code for Phase 1 (Processing Interactions)** now? This is the most complex data manipulation step.