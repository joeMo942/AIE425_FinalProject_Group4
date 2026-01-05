"""
IGDB Game Data Fetcher
======================
Team Members:
- [Add your names and IDs here]

Fetches game metadata from IGDB API using Twitch OAuth.
Enriches text features with game descriptions, genres, and themes.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import requests
import pandas as pd
import time
import json
from pathlib import Path
from dotenv import load_dotenv

# ============================================================================
# Configuration
# ============================================================================
# Load environment from project root (use absolute path)
load_dotenv("/home/yousef/AIE425_FinalProject_GroupX/.env")

DATA_DIR = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

ITEMS_FILE = DATA_DIR / "streamer_metadata.csv"
OUTPUT_FILE = DATA_DIR / "game_metadata.csv"
CHECKPOINT_FILE = DATA_DIR / "igdb_checkpoint.json"

# IGDB API
IGDB_URL = "https://api.igdb.com/v4/games"
CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
ACCESS_TOKEN = os.getenv("TWITCH_ACCESS_TOKEN")

# Rate limiting
REQUEST_DELAY = 0.3  # seconds between requests


def get_headers():
    """Get API headers with Twitch OAuth."""
    return {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Accept": "application/json"
    }


def test_api_connection():
    """Test IGDB API connection."""
    print("=" * 60)
    print("TESTING IGDB API CONNECTION")
    print("=" * 60)
    
    if not CLIENT_ID or not ACCESS_TOKEN:
        print("[ERROR] Missing credentials. Check .env file.")
        return False
    
    # Test with a known game
    query = 'search "Fortnite"; fields name,summary; limit 1;'
    
    try:
        response = requests.post(
            IGDB_URL,
            headers=get_headers(),
            data=query
        )
        
        if response.status_code == 200:
            data = response.json()
            if data:
                print(f"[SUCCESS] API connected!")
                print(f"         Test game: {data[0].get('name', 'N/A')}")
                return True
            else:
                print("[WARNING] API returned empty response")
                return True
        else:
            print(f"[ERROR] API returned {response.status_code}")
            print(f"        Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return False


def fetch_game_metadata(game_name: str) -> dict:
    """
    Fetch metadata for a single game from IGDB.
    Returns dict with summary, genres, themes.
    """
    # Clean game name for search
    clean_name = game_name.replace(":", "").replace("'", "").strip()
    
    query = f'''
    search "{clean_name}";
    fields name,summary,genres.name,themes.name,keywords.name;
    limit 1;
    '''
    
    try:
        response = requests.post(
            IGDB_URL,
            headers=get_headers(),
            data=query
        )
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                game = data[0]
                
                # Extract genres
                genres = []
                if 'genres' in game:
                    genres = [g.get('name', '') for g in game['genres']]
                
                # Extract themes
                themes = []
                if 'themes' in game:
                    themes = [t.get('name', '') for t in game['themes']]
                
                # Extract keywords
                keywords = []
                if 'keywords' in game:
                    keywords = [k.get('name', '') for k in game['keywords'][:5]]  # Limit keywords
                
                return {
                    'game_name': game_name,
                    'igdb_name': game.get('name', ''),
                    'summary': game.get('summary', ''),
                    'genres': ', '.join(genres),
                    'themes': ', '.join(themes),
                    'keywords': ', '.join(keywords)
                }
        
        return {'game_name': game_name, 'igdb_name': '', 'summary': '', 'genres': '', 'themes': '', 'keywords': ''}
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch {game_name}: {e}")
        return {'game_name': game_name, 'igdb_name': '', 'summary': '', 'genres': '', 'themes': '', 'keywords': ''}


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


def fetch_all_games():
    """Fetch metadata for all unique games in the dataset."""
    print("\n" + "=" * 60)
    print("FETCHING GAME METADATA FROM IGDB")
    print("=" * 60)
    
    # Load items
    df_items = pd.read_csv(ITEMS_FILE)
    
    # Get unique games from MOST_STREAMED_GAME and 2ND_MOST_STREAMED_GAME columns
    games = set()
    if 'MOST_STREAMED_GAME' in df_items.columns:
        games.update(df_items['MOST_STREAMED_GAME'].dropna().unique())
    if '2ND_MOST_STREAMED_GAME' in df_items.columns:
        games.update(df_items['2ND_MOST_STREAMED_GAME'].dropna().unique())
    
    # Remove empty strings
    games = [g for g in games if g and len(str(g).strip()) > 0]
    
    print(f"[FOUND] {len(games)} unique games to fetch")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    completed = set(checkpoint['completed'])
    results = checkpoint['results']
    
    # Filter out already completed
    games_to_fetch = [g for g in games if g not in completed]
    print(f"[RESUME] {len(games_to_fetch)} games remaining")
    
    # Fetch each game
    for i, game in enumerate(games_to_fetch):
        print(f"\r[{i+1}/{len(games_to_fetch)}] Fetching: {game[:30]:.<35}", end="", flush=True)
        
        metadata = fetch_game_metadata(game)
        results.append(metadata)
        completed.add(game)
        
        # Checkpoint every 10 games
        if (i + 1) % 10 == 0:
            checkpoint['completed'] = list(completed)
            checkpoint['results'] = results
            save_checkpoint(checkpoint)
        
        time.sleep(REQUEST_DELAY)
    
    print("\n")
    
    # Save final results
    checkpoint['completed'] = list(completed)
    checkpoint['results'] = results
    save_checkpoint(checkpoint)
    
    # Create DataFrame and save
    df_games = pd.DataFrame(results)
    df_games.to_csv(OUTPUT_FILE, index=False)
    
    print(f"[SAVED] {OUTPUT_FILE.name} ({len(df_games)} games)")
    
    # Stats
    with_summary = df_games['summary'].apply(lambda x: len(str(x)) > 10 if pd.notna(x) else False).sum()
    print(f"[STATS] Games with summaries: {with_summary}/{len(df_games)}")
    
    return df_games


def enrich_items_with_game_data():
    """Merge game metadata into items dataset."""
    print("\n" + "=" * 60)
    print("ENRICHING ITEMS WITH GAME DATA")
    print("=" * 60)
    
    if not OUTPUT_FILE.exists():
        print("[ERROR] Game metadata not found. Run fetch first.")
        return None
    
    df_items = pd.read_csv(ITEMS_FILE)
    df_games = pd.read_csv(OUTPUT_FILE)
    
    # Create lookup dict
    game_lookup = {}
    for _, row in df_games.iterrows():
        game_name = row['game_name']
        text = f"{row['summary']} {row['genres']} {row['themes']}"
        game_lookup[game_name] = text.strip() if pd.notna(text) else ''
    
    # Enrich text_features
    def enrich_text(row):
        base_text = str(row['text_features']) if pd.notna(row['text_features']) else ''
        
        game1_text = game_lookup.get(row.get('MOST_STREAMED_GAME', ''), '')
        game2_text = game_lookup.get(row.get('2ND_MOST_STREAMED_GAME', ''), '')
        
        enriched = f"{base_text} {game1_text} {game2_text}".strip()
        return enriched
    
    df_items['text_features_enriched'] = df_items.apply(enrich_text, axis=1)
    
    # Save enriched items
    enriched_file = DATA_DIR / "final_items_enriched.csv"
    df_items.to_csv(enriched_file, index=False)
    
    print(f"[SAVED] {enriched_file.name}")
    
    # Sample
    print("\nSample enriched text features:")
    sample = df_items[['streamer_username', 'text_features_enriched']].head(3)
    for _, row in sample.iterrows():
        text = row['text_features_enriched'][:150] + "..." if len(row['text_features_enriched']) > 150 else row['text_features_enriched']
        print(f"  {row['streamer_username']}: {text}")
    
    return df_items


def main():
    """Main execution."""
    # Test connection first
    if not test_api_connection():
        print("\n[ABORT] Fix API connection and retry.")
        return
    
    # Fetch all game metadata
    fetch_all_games()
    
    # Enrich items
    # enrich_items_with_game_data()
    
    print("\n" + "=" * 60)
    print("[DONE] IGDB data fetch complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
