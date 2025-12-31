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
