# Utility Functions
# Data loader functions adapted from AIE425-Assignment-Group7

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os

# Define base paths relative to the code directory
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CODE_DIR, '..', 'data')

# File paths
DATASET_PATH = os.path.join(DATA_DIR, 'preprocessed_data.csv')
TARGET_USERS_PATH = os.path.join(DATA_DIR, 'target_users.txt')
TARGET_ITEMS_PATH = os.path.join(DATA_DIR, 'target_items.txt')
USER_AVG_RATINGS_PATH = os.path.join(DATA_DIR, 'r_u.csv')
ITEM_AVG_RATINGS_PATH = os.path.join(DATA_DIR, 'r_i.csv')


def get_preprocessed_dataset():
    """
    Loads the preprocessed dataset from CSV.
    
    Returns:
        pd.DataFrame: The dataset containing user, item, and rating.
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


def get_target_users():
    """
    Loads the list of target users from the data directory.
    
    Returns:
        list: A list of target user IDs (as integers).
    """
    if not os.path.exists(TARGET_USERS_PATH):
        raise FileNotFoundError(f"Target users file not found at {TARGET_USERS_PATH}")
    
    with open(TARGET_USERS_PATH, 'r') as f:
        users = [int(line.strip()) for line in f if line.strip()]
    return users


def get_target_items():
    """
    Loads the list of target items from the data directory.
    
    Returns:
        list: A list of target item IDs (as integers).
    """
    if not os.path.exists(TARGET_ITEMS_PATH):
        raise FileNotFoundError(f"Target items file not found at {TARGET_ITEMS_PATH}")
    
    with open(TARGET_ITEMS_PATH, 'r') as f:
        items = [int(line.strip()) for line in f if line.strip()]
    return items


def get_user_avg_ratings():
    """
    Loads the user average ratings from CSV.
    
    Returns:
        pd.DataFrame: DataFrame containing user and their average rating.
    """
    if not os.path.exists(USER_AVG_RATINGS_PATH):
        raise FileNotFoundError(f"User average ratings file not found at {USER_AVG_RATINGS_PATH}")
    return pd.read_csv(USER_AVG_RATINGS_PATH)


def get_item_avg_ratings():
    """
    Loads the item average ratings from CSV.
    
    Returns:
        pd.DataFrame: DataFrame containing item and their average rating.
    """
    if not os.path.exists(ITEM_AVG_RATINGS_PATH):
        raise FileNotFoundError(f"Item average ratings file not found at {ITEM_AVG_RATINGS_PATH}")
    return pd.read_csv(ITEM_AVG_RATINGS_PATH)


def create_user_item_matrix(df, user_col='user', item_col='item', rating_col='rating'):
    """
    Creates a user-item rating matrix from the dataset.
    
    Args:
        df: DataFrame containing user, item, and rating columns
        user_col: Name of the user column
        item_col: Name of the item column
        rating_col: Name of the rating column
    
    Returns:
        pd.DataFrame: User-item matrix with users as rows and items as columns
    """
    return df.pivot(index=user_col, columns=item_col, values=rating_col)


def mean_fill_matrix(matrix):
    """
    Fills missing values in the matrix using column (item) means.
    
    For each column (item), calculates the mean of existing ratings
    and replaces NaN values with that mean.
    
    Args:
        matrix: User-item rating matrix with NaN for missing values
    
    Returns:
        pd.DataFrame: Matrix with missing values filled using column means
    """
    import numpy as np
    
    # Convert to numpy for memory efficiency
    values = matrix.values.copy()
    
    # Calculate mean for each column (item), ignoring NaN
    col_means = np.nanmean(values, axis=0)
    
    # Find indices of NaN values
    nan_indices = np.where(np.isnan(values))
    
    # Replace NaN with corresponding column mean
    values[nan_indices] = np.take(col_means, nan_indices[1])
    
    # Create DataFrame with filled values
    filled_matrix = pd.DataFrame(values, index=matrix.index, columns=matrix.columns)
    return filled_matrix


def compute_centered_ratings(df, item_means_df, item_col='item', rating_col='rating'):
    """
    Calculate centered ratings (actual rating - item mean) for each rating.
    
    Args:
        df: DataFrame containing user, item, and rating columns
        item_means_df: DataFrame with item and r_i_bar (item mean) columns
        item_col: Name of the item column
        rating_col: Name of the rating column
    
    Returns:
        pd.DataFrame: Original df with additional 'centered_rating' column
    """
    # Create a mapping from item to its mean
    item_mean_map = item_means_df.set_index('item')['r_i_bar'].to_dict()
    
    # Calculate centered rating for each row
    df_centered = df.copy()
    df_centered['item_mean'] = df_centered[item_col].map(item_mean_map)
    df_centered['centered_rating'] = df_centered[rating_col] - df_centered['item_mean']
    
    return df_centered


def compute_covariance_matrix_efficient(df, items_list, item_means_df, total_users, show_progress=True):
    """
    Compute covariance matrix for items memory-efficiently.
    
    Uses the optimization that mean-filled values contribute zero after centering.
    Only users who rated BOTH items contribute to the covariance sum.
    Divides by N (total users), not just common users count.
    
    Args:
        df: DataFrame containing user, item, and rating columns
        items_list: List of item IDs (can be all items or subset)
        item_means_df: DataFrame with item and r_i_bar columns
        total_users: Total number of users N in the dataset
        show_progress: Whether to show progress updates
    
    Returns:
        pd.DataFrame: Covariance matrix (n x n) for items
    """
    import numpy as np
    from scipy import sparse
    
    # Create item mean mapping
    item_mean_map = item_means_df.set_index('item')['r_i_bar'].to_dict()
    
    # Filter to only the items in items_list
    df_filtered = df[df['item'].isin(items_list)].copy()
    
    # Add centered ratings
    df_filtered['item_mean'] = df_filtered['item'].map(item_mean_map)
    df_filtered['centered_rating'] = df_filtered['rating'] - df_filtered['item_mean']
    
    if show_progress:
        print(f"Building user-item centered rating structure...")
    
    # Create item index mapping for matrix positions
    item_to_idx = {item: idx for idx, item in enumerate(items_list)}
    n_items = len(items_list)
    
    # Build user ratings dictionary more efficiently using groupby
    user_ratings = {}
    for user, group in df_filtered.groupby('user'):
        user_ratings[user] = dict(zip(group['item'], group['centered_rating']))
    
    if show_progress:
        print(f"Number of users with ratings: {len(user_ratings):,}")
        print(f"Number of items: {n_items:,}")
        print(f"Computing covariance matrix ({n_items} x {n_items})...")
    
    # Initialize covariance matrix using numpy
    cov_matrix = np.zeros((n_items, n_items), dtype=np.float64)
    
    # For each user, add their contribution to the covariance matrix
    # This is more efficient than iterating over all item pairs
    user_count = 0
    for user, ratings in user_ratings.items():
        items_rated = list(ratings.keys())
        n_rated = len(items_rated)
        
        # For each pair of items this user rated
        for i in range(n_rated):
            item_i = items_rated[i]
            idx_i = item_to_idx[item_i]
            centered_i = ratings[item_i]
            
            for j in range(i, n_rated):
                item_j = items_rated[j]
                idx_j = item_to_idx[item_j]
                centered_j = ratings[item_j]
                
                # Add product to covariance sum
                product = centered_i * centered_j
                cov_matrix[idx_i, idx_j] += product
                if i != j:
                    cov_matrix[idx_j, idx_i] += product  # Symmetric
        
        user_count += 1
        if show_progress and user_count % 50000 == 0:
            print(f"  Processed {user_count:,} users...")
    
    # Divide by N-1 (sample covariance) to get covariance
    cov_matrix /= (total_users - 1)
    
    if show_progress:
        print(f"Covariance matrix computation complete!")
    
    # Create DataFrame with item labels
    cov_df = pd.DataFrame(cov_matrix, index=items_list, columns=items_list)
    cov_df.index.name = 'item'
    
    return cov_df


def get_top_peers(cov_matrix, item_id, n_peers):
    """
    Get top n peers for an item based on covariance values.
    
    Args:
        cov_matrix: Covariance matrix DataFrame
        item_id: The item to find peers for
        n_peers: Number of top peers to return
    
    Returns:
        pd.Series: Top n peers with their covariance values
    """
    cov_values = cov_matrix.loc[item_id].copy()
    cov_values = cov_values.drop(item_id)  # Remove self
    top_peers = cov_values.sort_values(ascending=False).head(n_peers)
    return top_peers


def project_user(user_id, W, all_items, item_means_dict, user_ratings):
    """
    Project a single user into the reduced latent space.
    
    Formula: UserVector_u[dim] = Sum_{j in Observed} (R_{u,j} - μ_j) × W_{j, dim}
    
    Args:
        user_id: The user to project
        W: The projection matrix (n_items x k)
        all_items: List of all items (defines row order in W)
        item_means_dict: Dictionary {item_id: mean}
        user_ratings: Dictionary {user_id: {item_id: rating}}
    
    Returns:
        np.array: User vector of size k
    """
    import numpy as np
    
    k = W.shape[1]  # Number of dimensions (5 or 10)
    user_vector = np.zeros(k)
    
    if user_id not in user_ratings:
        return user_vector
    
    # Only iterate over items the user actually rated
    for item_id, rating in user_ratings[user_id].items():
        if item_id in item_means_dict:
            # Get item index in W
            item_idx = all_items.index(item_id)
            # Centered rating: R_{u,j} - μ_j
            centered_rating = rating - item_means_dict[item_id]
            # Add contribution to each dimension
            user_vector += centered_rating * W[item_idx, :]
    
    return user_vector


def cosine_similarity(vec_u, vec_v):
    """
    Calculate cosine similarity between two user vectors.
    
    Formula: Sim(u, v) = (UserVector_u · UserVector_v) / (||UserVector_u|| × ||UserVector_v||)
    
    Args:
        vec_u: First vector
        vec_v: Second vector
    
    Returns:
        float: Cosine similarity in range [-1, 1]
    """
    import numpy as np
    
    norm_u = np.linalg.norm(vec_u)
    norm_v = np.linalg.norm(vec_v)
    
    if norm_u == 0 or norm_v == 0:
        return 0.0
    
    return np.dot(vec_u, vec_v) / (norm_u * norm_v)


def predict_rating(target_user, target_item, user_vectors, item_means_dict, 
                   user_ratings, n_neighbors=20):
    """
    Predict rating for a target user on a target item.
    
    Formula: r_hat_{u,i} = μ_i + (Sum_{v in N} Sim(u,v) × (R_{v,i} - μ_i)) / (Sum_{v in N} |Sim(u,v)|)
    
    Args:
        target_user: The user for whom to predict
        target_item: The item to predict rating for
        user_vectors: Dictionary {user_id: user_vector}
        item_means_dict: Dictionary {item_id: mean}
        user_ratings: Dictionary {user_id: {item_id: rating}}
        n_neighbors: Number of nearest neighbors to use
    
    Returns:
        tuple: (predicted_rating, neighbors_used, neighbor_details)
    """
    target_vec = user_vectors[target_user]
    item_mean = item_means_dict.get(target_item, 0)
    
    # Calculate similarity with all other users who rated the target item
    similarities = []
    for user_id, user_vec in user_vectors.items():
        if user_id == target_user:
            continue
        # Check if this user rated the target item
        if user_id in user_ratings and target_item in user_ratings[user_id]:
            sim = cosine_similarity(target_vec, user_vec)
            actual_rating = user_ratings[user_id][target_item]
            similarities.append((user_id, sim, actual_rating))
    
    if not similarities:
        # No neighbors found, return item mean
        return item_mean, 0, []
    
    # Sort by similarity (descending) and take top N neighbors
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_neighbors = similarities[:n_neighbors]
    
    # Calculate weighted average
    numerator = 0.0
    denominator = 0.0
    neighbor_details = []
    
    for user_id, sim, actual_rating in top_neighbors:
        centered_rating = actual_rating - item_mean
        numerator += sim * centered_rating
        denominator += abs(sim)
        neighbor_details.append({
            'neighbor_user': user_id,
            'similarity': sim,
            'actual_rating': actual_rating,
            'centered_rating': centered_rating
        })
    
    if denominator == 0:
        return item_mean, len(top_neighbors), neighbor_details
    
    predicted = item_mean + (numerator / denominator)
    
    # Clip to valid rating range [1, 5]
    predicted = max(1.0, min(5.0, predicted))
    
    return predicted, len(top_neighbors), neighbor_details


def compute_covariance_matrix_mle(df, items_list, item_means_df, show_progress=True):
    """
    Compute covariance matrix for items using MLE method.
    
    MLE method: Divides by (|Common(i,j)| - 1) instead of total users.
    Only users who rated BOTH items contribute to the covariance sum.
    If |Common(i,j)| < 2, set Cov(i,j) = 0.
    
    Formula: Cov(i,j) = Sum_{u in Common(i,j)} (X'_{u,i} × X'_{u,j}) / (|Common(i,j)| - 1)
    
    Args:
        df: DataFrame containing user, item, and rating columns
        items_list: List of item IDs
        item_means_df: DataFrame with item and r_i_bar columns
        show_progress: Whether to show progress updates
    
    Returns:
        pd.DataFrame: Covariance matrix (n x n) for items
    """
    import numpy as np
    
    # Create item mean mapping
    item_mean_map = item_means_df.set_index('item')['r_i_bar'].to_dict()
    
    # Filter to only the items in items_list
    df_filtered = df[df['item'].isin(items_list)].copy()
    
    # Add centered ratings
    df_filtered['item_mean'] = df_filtered['item'].map(item_mean_map)
    df_filtered['centered_rating'] = df_filtered['rating'] - df_filtered['item_mean']
    
    if show_progress:
        print(f"Building user-item centered rating structure...")
    
    # Create item index mapping for matrix positions
    item_to_idx = {item: idx for idx, item in enumerate(items_list)}
    n_items = len(items_list)
    
    # Build user ratings dictionary
    user_ratings = {}
    for user, group in df_filtered.groupby('user'):
        user_ratings[user] = dict(zip(group['item'], group['centered_rating']))
    
    if show_progress:
        print(f"Number of users with ratings: {len(user_ratings):,}")
        print(f"Number of items: {n_items:,}")
        print(f"Computing MLE covariance matrix ({n_items} x {n_items})...")
    
    # Initialize covariance sum matrix and count matrix
    cov_sum = np.zeros((n_items, n_items), dtype=np.float64)
    common_count = np.zeros((n_items, n_items), dtype=np.int64)
    
    # For each user, add their contribution to the covariance matrix
    user_count = 0
    for user, ratings in user_ratings.items():
        items_rated = list(ratings.keys())
        n_rated = len(items_rated)
        
        # For each pair of items this user rated
        for i in range(n_rated):
            item_i = items_rated[i]
            idx_i = item_to_idx[item_i]
            centered_i = ratings[item_i]
            
            for j in range(i, n_rated):
                item_j = items_rated[j]
                idx_j = item_to_idx[item_j]
                centered_j = ratings[item_j]
                
                # Add product to covariance sum
                product = centered_i * centered_j
                cov_sum[idx_i, idx_j] += product
                common_count[idx_i, idx_j] += 1
                if i != j:
                    cov_sum[idx_j, idx_i] += product  # Symmetric
                    common_count[idx_j, idx_i] += 1
        
        user_count += 1
        if show_progress and user_count % 50000 == 0:
            print(f"  Processed {user_count:,} users...")
    
    # Compute covariance: divide by (common_count - 1)
    # If common_count < 2, set covariance to 0
    cov_matrix = np.zeros((n_items, n_items), dtype=np.float64)
    for i in range(n_items):
        for j in range(n_items):
            if common_count[i, j] >= 2:
                cov_matrix[i, j] = cov_sum[i, j] / (common_count[i, j] - 1)
            else:
                cov_matrix[i, j] = 0.0
    
    if show_progress:
        print(f"MLE Covariance matrix computation complete!")
    
    # Create DataFrame with item labels
    cov_df = pd.DataFrame(cov_matrix, index=items_list, columns=items_list)
    cov_df.index.name = 'item'
    
    return cov_df


def predict_rating_reconstruction(target_user, target_item, user_scores, W, 
                                   item_means_dict, all_items):
    """
    Predict rating using reconstruction method (for MLE PCA).
    
    Formula: r_hat_{u,i} = μ_i + Sum_{p=1}^k (t_{u,p} × W_{i,p})
    
    Args:
        target_user: The user for whom to predict
        target_item: The item to predict rating for
        user_scores: Dictionary {user_id: user_score_vector (T_u)}
        W: Projection matrix (n_items x k)
        item_means_dict: Dictionary {item_id: mean}
        all_items: List of all items (defines row order in W)
    
    Returns:
        float: Predicted rating
    """
    import numpy as np
    
    # Get item mean
    item_mean = item_means_dict.get(target_item, 0)
    
    # Get user score vector (T_u)
    if target_user not in user_scores:
        return item_mean
    
    T_u = user_scores[target_user]
    
    # Get item index in W
    if target_item not in all_items:
        return item_mean
    
    item_idx = all_items.index(target_item)
    
    # Get item loadings (W_{i,:})
    W_i = W[item_idx, :]
    
    # Compute prediction: μ_i + Sum(t_{u,p} × W_{i,p})
    predicted = item_mean + np.dot(T_u, W_i)
    
    # Clip to valid rating range [1, 5]
    predicted = max(1.0, min(5.0, predicted))
    
    return predicted

