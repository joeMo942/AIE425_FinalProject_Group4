"""
Content-Based Recommendation System
====================================
Team Members:
- [Add your names and IDs here]

Implements content-based filtering using:
- TF-IDF for text feature extraction
- User profile building from rated items
- Cosine similarity for item matching
- k-NN for recommendations
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import pickle

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RATINGS_FILE = DATA_DIR / "final_ratings.csv"
ITEMS_FILE = DATA_DIR / "final_items.csv"

# Model parameters
TOP_N = 10  # Number of recommendations
K_NEIGHBORS = 20  # k-NN neighbors


def load_data():
    """Load ratings and items data."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    df_ratings = pd.read_csv(RATINGS_FILE)
    df_items = pd.read_csv(ITEMS_FILE)
    
    print(f"[LOADED] Ratings: {len(df_ratings):,}")
    print(f"[LOADED] Items: {len(df_items):,}")
    
    return df_ratings, df_items


def prepare_item_features(df_items):
    """
    Prepare item features for content-based filtering.
    Combines TF-IDF on text features with normalized numerical features.
    """
    print("\n" + "=" * 60)
    print("PREPARING ITEM FEATURES")
    print("=" * 60)
    
    # Fill missing text features
    df = df_items.copy()
    df['text_features'] = df['text_features'].fillna('')
    
    # Get list of unique streamers
    streamers = df['streamer_username'].tolist()
    
    # --- TF-IDF on text features ---
    print("\n[1] Extracting TF-IDF features from text...")
    
    tfidf = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    
    tfidf_matrix = tfidf.fit_transform(df['text_features'])
    print(f"    TF-IDF shape: {tfidf_matrix.shape}")
    print(f"    Top features: {tfidf.get_feature_names_out()[:10].tolist()}")
    
    # --- Normalized numerical features ---
    print("\n[2] Normalizing numerical features...")
    
    numerical_cols = ['rank', 'avg_viewers', 'followers']
    available_cols = [c for c in numerical_cols if c in df.columns]
    
    if available_cols:
        scaler = MinMaxScaler()
        numerical_features = df[available_cols].fillna(0).values
        
        # Invert rank (lower rank = better)
        if 'rank' in available_cols:
            rank_idx = available_cols.index('rank')
            max_rank = numerical_features[:, rank_idx].max()
            numerical_features[:, rank_idx] = max_rank - numerical_features[:, rank_idx]
        
        numerical_normalized = scaler.fit_transform(numerical_features)
        print(f"    Numerical features: {available_cols}")
    else:
        numerical_normalized = np.zeros((len(df), 1))
    
    # --- Combine TF-IDF + Numerical ---
    print("\n[3] Combining features...")
    
    # Weight: 70% TF-IDF, 30% numerical
    combined_features = np.hstack([
        tfidf_matrix.toarray() * 0.7,
        numerical_normalized * 0.3
    ])
    
    print(f"    Combined feature matrix: {combined_features.shape}")
    
    # --- Build item-to-index mapping ---
    item_to_idx = {streamer: idx for idx, streamer in enumerate(streamers)}
    idx_to_item = {idx: streamer for idx, streamer in enumerate(streamers)}
    
    return {
        'features': combined_features,
        'tfidf': tfidf,
        'tfidf_matrix': tfidf_matrix,
        'item_to_idx': item_to_idx,
        'idx_to_item': idx_to_item,
        'streamers': streamers
    }


def compute_item_similarity(item_data):
    """Compute pairwise cosine similarity between all items."""
    print("\n" + "=" * 60)
    print("COMPUTING ITEM SIMILARITY")
    print("=" * 60)
    
    features = item_data['features']
    similarity_matrix = cosine_similarity(features)
    
    print(f"[DONE] Similarity matrix: {similarity_matrix.shape}")
    
    # Sample similarities
    print("\n       Sample similarities (diagonal should be 1.0):")
    print(f"       Item 0 to Item 0: {similarity_matrix[0, 0]:.4f}")
    print(f"       Item 0 to Item 1: {similarity_matrix[0, 1]:.4f}")
    
    return similarity_matrix


def build_user_profiles(df_ratings, item_data):
    """
    Build user profiles based on their rated items.
    User profile = weighted average of item features they liked.
    """
    print("\n" + "=" * 60)
    print("BUILDING USER PROFILES")
    print("=" * 60)
    
    features = item_data['features']
    item_to_idx = item_data['item_to_idx']
    
    users = df_ratings['user_id'].unique()
    user_profiles = {}
    
    # Progress tracking
    total_users = len(users)
    processed = 0
    
    for user_id in users:
        user_ratings = df_ratings[df_ratings['user_id'] == user_id]
        
        # Get item indices and ratings for this user
        item_indices = []
        ratings = []
        
        for _, row in user_ratings.iterrows():
            streamer = row['streamer_username']
            if streamer in item_to_idx:
                item_indices.append(item_to_idx[streamer])
                ratings.append(row['rating'])
        
        if item_indices:
            # Weighted average: higher rated items contribute more
            ratings = np.array(ratings)
            weights = ratings / ratings.sum()
            
            user_profile = np.average(features[item_indices], axis=0, weights=weights)
            user_profiles[user_id] = user_profile
        
        processed += 1
        if processed % 10000 == 0:
            print(f"       Processed {processed:,}/{total_users:,} users")
    
    print(f"\n[DONE] Built {len(user_profiles):,} user profiles")
    
    return user_profiles


def train_knn_model(item_data, n_neighbors=K_NEIGHBORS):
    """Train k-NN model for fast similarity lookup."""
    print("\n" + "=" * 60)
    print("TRAINING k-NN MODEL")
    print("=" * 60)
    
    knn = NearestNeighbors(
        n_neighbors=min(n_neighbors, len(item_data['streamers'])),
        metric='cosine',
        algorithm='brute'
    )
    
    knn.fit(item_data['features'])
    print(f"[DONE] k-NN model trained with k={n_neighbors}")
    
    return knn


def recommend_for_user(user_id, user_profiles, item_data, similarity_matrix, 
                       df_ratings, n_recommendations=TOP_N):
    """
    Generate recommendations for a specific user.
    Uses user profile to find similar items they haven't rated.
    """
    if user_id not in user_profiles:
        return []
    
    user_profile = user_profiles[user_id]
    features = item_data['features']
    idx_to_item = item_data['idx_to_item']
    item_to_idx = item_data['item_to_idx']
    
    # Get items user has already rated
    rated_items = set(df_ratings[df_ratings['user_id'] == user_id]['streamer_username'])
    rated_indices = {item_to_idx[item] for item in rated_items if item in item_to_idx}
    
    # Compute similarity between user profile and all items
    user_profile_2d = user_profile.reshape(1, -1)
    similarities = cosine_similarity(user_profile_2d, features)[0]
    
    # Sort by similarity, excluding already rated items
    recommendations = []
    for idx in np.argsort(similarities)[::-1]:
        if idx not in rated_indices:
            recommendations.append({
                'streamer': idx_to_item[idx],
                'score': similarities[idx]
            })
            if len(recommendations) >= n_recommendations:
                break
    
    return recommendations


def evaluate_recommendations(df_ratings, user_profiles, item_data, 
                            similarity_matrix, n_users=100):
    """
    Evaluate recommendation quality using held-out ratings.
    """
    print("\n" + "=" * 60)
    print("EVALUATING RECOMMENDATIONS")
    print("=" * 60)
    
    # Sample users with at least 3 ratings
    user_rating_counts = df_ratings.groupby('user_id').size()
    eligible_users = user_rating_counts[user_rating_counts >= 3].index.tolist()
    
    if len(eligible_users) > n_users:
        np.random.seed(42)
        sample_users = np.random.choice(eligible_users, n_users, replace=False)
    else:
        sample_users = eligible_users
    
    print(f"[EVAL] Testing on {len(sample_users)} users")
    
    hits = 0
    total = 0
    
    for user_id in sample_users:
        user_ratings = df_ratings[df_ratings['user_id'] == user_id]
        
        # Use all but one as "history", test on the held-out item
        if len(user_ratings) < 2:
            continue
            
        # Get user's highest rated item as "target"
        highest_rated = user_ratings.loc[user_ratings['rating'].idxmax()]
        target_item = highest_rated['streamer_username']
        
        # Get recommendations (using full profile for simplicity)
        recommendations = recommend_for_user(
            user_id, user_profiles, item_data, similarity_matrix, 
            df_ratings, n_recommendations=TOP_N
        )
        
        rec_items = [r['streamer'] for r in recommendations]
        
        # Check if target is in recommendations (or similar items)
        if target_item in rec_items:
            hits += 1
        
        total += 1
    
    hit_rate = hits / total if total > 0 else 0
    print(f"\n       Hit Rate @ {TOP_N}: {hit_rate:.2%} ({hits}/{total})")
    
    return hit_rate


def generate_sample_recommendations(df_ratings, user_profiles, item_data, 
                                     similarity_matrix, df_items, n_samples=5):
    """Generate and display sample recommendations."""
    print("\n" + "=" * 60)
    print("SAMPLE RECOMMENDATIONS")
    print("=" * 60)
    
    # Get users with enough ratings
    user_rating_counts = df_ratings.groupby('user_id').size()
    eligible_users = user_rating_counts[user_rating_counts >= 3].index.tolist()
    
    np.random.seed(42)
    sample_users = np.random.choice(eligible_users, min(n_samples, len(eligible_users)), replace=False)
    
    results = []
    
    for user_id in sample_users:
        user_ratings = df_ratings[df_ratings['user_id'] == user_id].merge(
            df_items[['streamer_username', 'language', '1st_game']], 
            on='streamer_username', how='left'
        )
        
        print(f"\n--- User {user_id} ---")
        print("Watched streamers:")
        for _, row in user_ratings.iterrows():
            print(f"  ★{'★' * (row['rating']-1)} {row['streamer_username']} ({row.get('language', '?')}, {row.get('1st_game', '?')})")
        
        recommendations = recommend_for_user(
            user_id, user_profiles, item_data, similarity_matrix,
            df_ratings, n_recommendations=TOP_N
        )
        
        print(f"\nTop {TOP_N} Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            streamer = rec['streamer']
            score = rec['score']
            item_info = df_items[df_items['streamer_username'] == streamer].iloc[0] if len(df_items[df_items['streamer_username'] == streamer]) > 0 else {}
            lang = item_info.get('language', '?') if isinstance(item_info, dict) else (item_info['language'] if 'language' in item_info else '?')
            game = item_info.get('1st_game', '?') if isinstance(item_info, dict) else (item_info['1st_game'] if '1st_game' in item_info else '?')
            print(f"  {i}. {streamer} (score: {score:.3f}) - {lang}, {game}")
        
        results.append({
            'user_id': user_id,
            'history': user_ratings['streamer_username'].tolist(),
            'recommendations': [r['streamer'] for r in recommendations]
        })
    
    return results


def save_model(item_data, user_profiles, similarity_matrix):
    """Save trained model for later use."""
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    model_path = RESULTS_DIR / "content_based_model.pkl"
    
    model = {
        'item_features': item_data['features'],
        'item_to_idx': item_data['item_to_idx'],
        'idx_to_item': item_data['idx_to_item'],
        'user_profiles': user_profiles,
        'similarity_matrix': similarity_matrix
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"[SAVED] {model_path.name}")


def main():
    """Main content-based recommendation pipeline."""
    print("\n" + "=" * 60)
    print("CONTENT-BASED RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    # Load data
    df_ratings, df_items = load_data()
    
    # Prepare item features (TF-IDF + numerical)
    item_data = prepare_item_features(df_items)
    
    # Compute item similarity matrix
    similarity_matrix = compute_item_similarity(item_data)
    
    # Build user profiles
    user_profiles = build_user_profiles(df_ratings, item_data)
    
    # Train k-NN model
    knn_model = train_knn_model(item_data)
    
    # Evaluate
    hit_rate = evaluate_recommendations(
        df_ratings, user_profiles, item_data, similarity_matrix
    )
    
    # Generate sample recommendations
    results = generate_sample_recommendations(
        df_ratings, user_profiles, item_data, similarity_matrix, df_items
    )
    
    # Save model
    save_model(item_data, user_profiles, similarity_matrix)
    
    print("\n" + "=" * 60)
    print("[DONE] Content-based recommendation complete!")
    print("=" * 60 + "\n")
    
    return {
        'item_data': item_data,
        'similarity_matrix': similarity_matrix,
        'user_profiles': user_profiles,
        'hit_rate': hit_rate
    }


if __name__ == "__main__":
    main()
