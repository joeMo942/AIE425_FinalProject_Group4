"""
Main Orchestrator & Web App for Section 2: Domain Recommender System
====================================================================
This script serves as the single entry point.
1. Pipeline Verification: Checks if valid data and models exist. (Runs scripts if missing)
2. Web Application: Launches the FastAPI server for the Recommender UI.
"""

import sys
import subprocess
import time
import json
import pickle
import pandas as pd
import numpy as np
import uvicorn
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ============================================================================
# Configuration & Paths
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
CODE_DIR = BASE_DIR / "code"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
TEMPLATES_DIR = DATA_DIR / "templates"

# Pipeline Files
DATA_FILES = [DATA_DIR / "final_ratings.csv", DATA_DIR / "final_items_enriched.csv"]
CB_MODEL = RESULTS_DIR / "content_based_model.pkl"
CF_MODELS = [RESULTS_DIR / "collaborative_model.pkl", RESULTS_DIR / "svd_model.pkl", RESULTS_DIR / "svd_predictions.npy"]


# ============================================================================
# App Global State
# ============================================================================
app = FastAPI()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

unique_games = []
unique_languages = []
df_items = None
streamer_images = {}
tfidf_features = None

# ============================================================================
# Pipeline Logic
# ============================================================================
def run_script(script_name, description):
    """Run a python script as a subprocess."""
    script_path = CODE_DIR / script_name
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_path.name}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=BASE_DIR.parent
        )
        duration = time.time() - start_time
        print(f"\n[SUCCESS] {script_name} completed in {duration:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {script_name} failed with exit code {e.returncode}")
        return False

def check_pipeline():
    """Verify data and models exist, run scripts if missing."""
    print("Checking pipeline integrity...")
    
    # 1. Check Data
    missing_data = [f.name for f in DATA_FILES if not f.exists()]
    if missing_data:
        print(f"[MISSING] Datasets: {', '.join(missing_data)}")
        if not run_script("data_preprocessing.py", "Data Preprocessing"):
            return False
            
    # 2. Check Models
    if not CB_MODEL.exists():
        print(f"[MISSING] Content-Based Model")
        if not run_script("content_based.py", "Content-Based Training"):
            return False
            
    missing_cf = [f.name for f in CF_MODELS if not f.exists()]
    if missing_cf:
        print(f"[MISSING] Collaborative Models: {', '.join(missing_cf)}")
        if not run_script("collaborative.py", "Collaborative Filtering Training"):
            return False
            
    print("[OK] Pipeline verified.")
    return True

# ============================================================================
# Web App Logic
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """Run pipeline checks and load data on startup."""
    # 1. Pipeline Check
    if not check_pipeline():
        print("[FATAL] Pipeline verification failed.")
        sys.exit(1)
        
    # 2. Load App Data
    global unique_games, unique_languages, df_items, streamer_images, tfidf_features
    print("Loading App Data...")
    
    # Static Data
    if (DATA_DIR / "unique_games.json").exists():
        with open(DATA_DIR / "unique_games.json", 'r') as f:
            unique_games = json.load(f)
            
    if (DATA_DIR / "unique_languages.json").exists():
        with open(DATA_DIR / "unique_languages.json", 'r') as f:
            unique_languages = json.load(f)

    if (DATA_DIR / "streamer_images.json").exists():
        with open(DATA_DIR / "streamer_images.json", 'r') as f:
            streamer_images = json.load(f)
            
    # Load DataFrame
    df_items = pd.read_csv(DATA_DIR / "final_items_enriched.csv")
    
    # Load Model Features (for context only, logic is simplified below)
    if CB_MODEL.exists():
        with open(CB_MODEL, 'rb') as f:
            model_data = pickle.load(f)
            # tfidf_features = model_data['item_features'] # Not strictly needed for metadata match
            
    print("Web App Ready!")


# ============================================================================
# FastAPI Routes
# ============================================================================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "games": unique_games,
        "languages": unique_languages
    })

@app.post("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, selected_games: list = Form(...), languages: list = Form(...)):
    # Content-Based Matching Logic
    scores = []
    
    for _, row in df_items.iterrows():
        score = 0
        feature_text = str(row['text_features']).lower()
        
        # Game Match (High Weight)
        for game in selected_games:
            if game.lower() in feature_text:
                score += 5
                
        # Language Match (Critical Weight)
        for lang in languages:
            if lang.lower() in str(row['language']).lower():
                score += 10
        
        # Popularity Boost
        pop_score = np.log1p(row['avg_viewers']) / 15.0
        score += pop_score
        
        # Image Lookup
        username = str(row['streamer_username'])
        img_url = streamer_images.get(username, streamer_images.get(username.lower()))
        if not img_url:
             # Fallback
             img_url = f"https://ui-avatars.com/api/?name={username}&background=6441a5&color=fff"
        
        scores.append({
            'streamer': username,
            'score': score,
            'url': f"https://twitch.tv/{username}",
            'img': img_url,
            'game': row['1st_game'],
            'lang': row['language'],
            'viewers': int(row['avg_viewers']) if pd.notna(row['avg_viewers']) else 0
        })
        
    # Sort and return top 10
    scores.sort(key=lambda x: x['score'], reverse=True)
    top_10 = scores[:10]
    
    return templates.TemplateResponse("results.html", {
        "request": request,
        "recommendations": top_10
    })


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING SECTION 2 RECOMMENDER SYSTEM")
    print("="*60)
    print("Access the UI at: http://127.0.0.1:8000\n")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
