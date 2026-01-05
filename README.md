# AIE425 Final Project: Advanced Recommender Systems

This repository contains the final project for the AIE425 course, focusing on **Dimensionality Reduction** techniques and building a robust **Domain-Specific Recommender System** for Twitch streamers.

## 📂 Project Structure

 The project is divided into two main sections, each with its own codebase and documentation:

### [Section 1: Dimensionality Reduction](./SECTION1_DimensionalityReduction/README.md)
Focuses on analyzing the underlying structure of user-item interaction matrices using mathematical techniques.
*   **Techniques:** PCA (with Mean Filling), PCA (MLE), and SVD Matrix Factorization.
*   **Goal:** efficient compression of sparse datasets while retaining variance.
*   **Key Files:** `pca_mean_filling.py`, `svd_analysis.py`

### [Section 2: Twitch Recommender System](./SECTION2_DomainRecommender/README.md)
Implements a complete, end-to-end hybrid recommender system for Twitch.tv.
*   **Techniques:** Content-Based Filtering (TF-IDF), Collaborative Filtering (K-NN & SVD), and Cascade Hybrid Strategy.
*   **Features:** Data extraction pipeline, cold-start handling, and a **FastAPI Web Application** for real-time interaction.
*   **Key Files:** `main.py`, `hybrid.py`, `fetch_streamer_data.py`

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/AIE425_FinalProject_Group4.git
    cd AIE425_FinalProject_GroupX
    ```

2.  **Install Dependencies:**
    All python dependencies are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Navigate to Sections:**
    Follow the detailed instructions in each section's README to run the specific experiments.

    *   [Go to Section 1 README](./SECTION1_DimensionalityReduction/README.md)
    *   [Go to Section 2 README](./SECTION2_DomainRecommender/README.md)

---
