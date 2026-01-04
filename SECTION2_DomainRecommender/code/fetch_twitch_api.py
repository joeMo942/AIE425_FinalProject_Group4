"""
Twitch Helix API Data Fetcher
=============================
Team Members:
- [Add your names and IDs here]

Uses Twitch OAuth to fetch streamer metadata directly from Twitch Helix API.
This is more reliable than web scraping and provides official data.

API Endpoints Used:
- GET /helix/users - User info (bio, profile image, type)
- GET /helix/channels - Channel info (game, language, title)
- GET /helix/videos - Stream history (for average duration calculation)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import requests
import pandas as pd
import time
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# ============================================================================
# Configuration
# ============================================================================
load_dotenv("/home/yousef/AIE425_FinalProject_GroupX/.env")

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

STREAMERS_FILE = DATA_DIR / "top_2000_streamers.txt"
OUTPUT_FILE = DATA_DIR / "twitch_streamer_metadata.csv"
CHECKPOINT_FILE = DATA_DIR / "twitch_api_checkpoint.json"

# Twitch API
TWITCH_API_URL = "https://api.twitch.tv/helix"
CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
ACCESS_TOKEN = os.getenv("TWITCH_ACCESS_TOKEN")

# Rate limiting (Twitch allows 800 requests/minute)
REQUEST_DELAY = 0.1  # seconds between requests
BATCH_SIZE = 100  # Max users per request


def get_headers():
    """Get API headers with Twitch OAuth."""
    return {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {ACCESS_TOKEN}",
    }


def test_api_connection():
    """Test Twitch API connection."""
    print("=" * 60)
    print("TESTING TWITCH HELIX API CONNECTION")
    print("=" * 60)
    
    if not CLIENT_ID or not ACCESS_TOKEN:
        print("[ERROR] Missing credentials. Check .env file.")
        return False
    
    # Test with users endpoint
    response = requests.get(
        f"{TWITCH_API_URL}/users",
        headers=get_headers(),
        params={"login": "ninja"}
    )
    
    if response.status_code == 200:
        data = response.json()
        if data.get('data'):
            user = data['data'][0]
            print(f"[SUCCESS] API connected!")
            print(f"         Test user: {user.get('display_name', 'N/A')}")
            print(f"         Followers: Available via /channels endpoint")
            return True
    
    print(f"[ERROR] API returned {response.status_code}")
    print(f"        Response: {response.text[:200]}")
    return False


def fetch_users_batch(usernames: list) -> dict:
    """
    Fetch user info for a batch of usernames (max 100).
    Returns dict mapping username -> user data.
    """
    if not usernames:
        return {}
    
    response = requests.get(
        f"{TWITCH_API_URL}/users",
        headers=get_headers(),
        params={"login": usernames}
    )
    
    if response.status_code == 200:
        data = response.json()
        return {user['login'].lower(): user for user in data.get('data', [])}
    
    return {}


def fetch_channels_batch(user_ids: list) -> dict:
    """
    Fetch channel info (game, language) for a batch of user IDs (max 100).
    Returns dict mapping user_id -> channel data.
    """
    if not user_ids:
        return {}
    
    response = requests.get(
        f"{TWITCH_API_URL}/channels",
        headers=get_headers(),
        params={"broadcaster_id": user_ids}
    )
    
    if response.status_code == 200:
        data = response.json()
        return {ch['broadcaster_id']: ch for ch in data.get('data', [])}
    
    return {}


def fetch_followers_count(user_id: str) -> int:
    """Fetch follower count for a user (requires separate request)."""
    response = requests.get(
        f"{TWITCH_API_URL}/channels/followers",
        headers=get_headers(),
        params={"broadcaster_id": user_id, "first": 1}
    )
    
    if response.status_code == 200:
        data = response.json()
        return data.get('total', 0)
    
    return 0


def load_checkpoint():
    """Load progress checkpoint."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'completed': [], 'results': []}


def save_checkpoint(checkpoint):
    """Save progress checkpoint."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)


def fetch_all_streamers():
    """Fetch metadata for all streamers using Twitch API."""
    print("\n" + "=" * 60)
    print("FETCHING STREAMER DATA FROM TWITCH API")
    print("=" * 60)
    
    # Load streamer list
    if not STREAMERS_FILE.exists():
        print(f"[ERROR] Streamer list not found: {STREAMERS_FILE}")
        return None
    
    with open(STREAMERS_FILE, 'r') as f:
        streamers = [line.strip().lower() for line in f if line.strip()]
    
    print(f"[FOUND] {len(streamers)} streamers to fetch")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    completed = set(checkpoint['completed'])
    results = checkpoint['results']
    
    # Filter out completed
    remaining = [s for s in streamers if s not in completed]
    print(f"[RESUME] {len(remaining)} streamers remaining")
    
    # Process in batches
    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i:i + BATCH_SIZE]
        print(f"\r[{i + len(batch)}/{len(remaining)}] Processing batch...", end="", flush=True)
        
        # Step 1: Get user info
        users = fetch_users_batch(batch)
        time.sleep(REQUEST_DELAY)
        
        if not users:
            # Mark as completed even if not found
            for username in batch:
                completed.add(username)
            continue
        
        # Step 2: Get channel info
        user_ids = [u['id'] for u in users.values()]
        channels = fetch_channels_batch(user_ids)
        time.sleep(REQUEST_DELAY)
        
        # Step 3: Combine data
        for username in batch:
            if username in users:
                user = users[username]
                user_id = user['id']
                channel = channels.get(user_id, {})
                
                # Get follower count (one request per user - can be slow)
                # Skip for now to save API calls, or enable for accurate data
                # followers = fetch_followers_count(user_id)
                
                result = {
                    'RANK': None,  # Not available from Twitch API directly
                    'NAME': username,
                    'LANGUAGE': channel.get('broadcaster_language', ''),
                    'TYPE': user.get('broadcaster_type', 'affiliate'),  # affiliate, partner, ""
                    'MOST_STREAMED_GAME': channel.get('game_name', ''),
                    '2ND_MOST_STREAMED_GAME': '',  # Would need video history analysis
                    'AVERAGE_STREAM_DURATION': None,  # Would need video history analysis
                    'FOLLOWERS_GAINED_PER_STREAM': None,
                    'AVG_VIEWERS_PER_STREAM': None,  # Not directly available
                    'AVG_GAMES_PER_STREAM': None,
                    'TOTAL_TIME_STREAMED': None,
                    'TOTAL_FOLLOWERS': None,  # Would need extra API call
                    'TOTAL_VIEWS': user.get('view_count', 0),
                    'TOTAL_GAMES_STREAMED': None,
                    'ACTIVE_DAYS_PER_WEEK': None,
                    'MOST_ACTIVE_DAY': '',
                    'DAY_WITH_MOST_FOLLOWERS_GAINED': '',
                    # Extra fields from Twitch API
                    'DESCRIPTION': user.get('description', ''),
                    'PROFILE_IMAGE': user.get('profile_image_url', ''),
                    'CREATED_AT': user.get('created_at', ''),
                }
                results.append(result)
            
            completed.add(username)
        
        # Checkpoint every batch
        checkpoint['completed'] = list(completed)
        checkpoint['results'] = results
        save_checkpoint(checkpoint)
        
        time.sleep(REQUEST_DELAY)
    
    print("\n")
    
    # Save final results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"[SAVED] {OUTPUT_FILE.name} ({len(df)} streamers)")
    
    # Stats
    with_game = df['MOST_STREAMED_GAME'].apply(lambda x: len(str(x)) > 0).sum()
    print(f"[STATS] Streamers with game: {with_game}/{len(df)}")
    
    return df


def main():
    """Main execution."""
    # Test connection
    if not test_api_connection():
        print("\n[ABORT] Fix API connection and retry.")
        return
    
    # Fetch all streamers
    df = fetch_all_streamers()
    
    if df is not None:
        # Show sample
        print("\n" + "=" * 60)
        print("SAMPLE DATA")
        print("=" * 60)
        print(df[['NAME', 'LANGUAGE', 'TYPE', 'MOST_STREAMED_GAME', 'TOTAL_VIEWS']].head(10))
    
    print("\n" + "=" * 60)
    print("[DONE] Twitch API fetch complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
