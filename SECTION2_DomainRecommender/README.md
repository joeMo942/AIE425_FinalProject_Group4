# Section 2: Twitch Streamer Recommender System 🎮

This project implements a comprehensive **Hybrid Recommendation System** for Twitch streamers. It combines content-based filtering (using bio, games, and language) with collaborative filtering (user-item ratings) to provide personalized recommendations. It also features a **FastAPI Web Application** for users to explore and interact with the system.

## 📂 Project Structure

```
SECTION2_DomainRecommender/
├── code/
│   ├── main.py                 # Main Orchestrator & Web App Entry Point
│   ├── data_preprocessing.py   # Data cleaning, merging, and EDA
│   ├── content_based.py        # Content-Based Filtering Model
│   ├── collaborative.py        # Collaborative Filtering (K-NN & SVD)
│   └── hybrid.py               # Hybrid Logic (Cascade Strategy)
├── data/
│   ├── scraping/               # Web scrapers for data enrichment
│   │   ├── fetch_igdb_data.py
│   │   └── fetch_streamer_data.py
│   ├── templates/              # HTML Templates for Web App
│   │   ├── index.html
│   │   └── results.html
│   ├── final_ratings.csv       # Processed user ratings
│   └── final_items_enriched.csv # Streamer metadata
├── results/                    # Saved models (.pkl) and metric reports
└── Section2_Report.md          # Detailed analysis and answer to assignment questions
```

## 🚀 Quick Start

### 0. Dataset Setup (CRITICAL)
Before running anything, you must download the necessary dataset:

1.  **Download:** [INSERT_DATASET_LINK_HERE]
2.  **Extract:** Unzip the contents directly into the `data/` folder.
    *   Ensure the file `final_ratings.csv` is located at `SECTION2_DomainRecommender/data/final_ratings.csv`.
    *   Ensure the file `final_items_enriched.csv` is located at `SECTION2_DomainRecommender/data/final_items_enriched.csv`.

### 1. Setup Environment
Ensure you have Python 3.8+ and the required dependencies installed:
```bash
pip install -r ../requirements.txt
```



### 2. Run the Full System
The `main.py` script serves as the **single entry point**. It will automatically:
1.  **Check the Pipeline:** Verify if data exists and models are trained.
2.  **Training:** If models are missing, it will automatically run `data_preprocessing.py`, `content_based.py`, and `collaborative.py`.
3.  **Launch App:** Starts the Web UI.

```bash
python code/main.py
```

### 3. Access the Web App
Once running, open your browser to:
**http://127.0.0.1:8000**

## 🧠 System Architecture

### 1. Data Pipeline (`data_preprocessing.py`)
-   **Preprocessing:** Converts raw 100k log entries into explicit 1-5 ratings using user-specific normalization.
-   **Merging:** Enriches ratings with streamer metadata (games, language, followers) scraped from TwitchTracker and IGDB.
-   **EDA:** Generates statistical reports and plots on sparsity and distributions.

### 2. Recommendation Models
*   **Content-Based (`content_based.py`):** Uses TF-IDF on streamer bios and metadata to find similar streamers. Handles "Unique Tastes."
*   **Collaborative Filtering (`collaborative.py`):**
    *   **User-Based K-NN:** Finds users with similar rating patterns.
    *   **SVD Matrix Factorization:** Latent factor model to handle sparsity.
*   **Hybrid Engine (`hybrid.py`):**
    *   **Strategy:** Cascade Hybrid.
    *   **Logic:** Uses Content-Based filtering to select top-50 candidates (fast), then re-ranks them using SVD/Collaborative scores (accurate).
    *   **Cold-Start:** Uses specific fallback logic (Popularity -> Content-Based -> Hybrid) depending on user history length.

## 📊 Evaluation & Reports

Final evaluation metrics and the detailed report can be found in `Section2_Report.md`.
The system generates numerical examples, hit-rate comparisons, and sparsity analysis automatically in the `results/` folder.

## 🛠️ Web App Features
-   **Visual Selection:** Select favorite games from a grid of cover images.
-   **Real-time Recommendations:** Generates recommendations instantly based on your selections.
-   **Interactive UI:** Clean, modern Dark Mode interface.
