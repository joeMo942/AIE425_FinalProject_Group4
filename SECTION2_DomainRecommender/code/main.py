"""
Main Orchestrator for Section 2: Domain Recommender System
===========================================================
This script serves as the single entry point for the recommender system.
It automatically manages the pipeline dependencies:

1. Checks for Data:
   - If 'final_ratings.csv' or 'final_items_enriched.csv' are missing:
     -> Runs data_preprocessing.py

2. Checks for Models:
   - If 'content_based_model.pkl' is missing:
     -> Runs content_based.py
   - If 'collaborative_model.pkl' or 'svd_model.pkl' are missing:
     -> Runs collaborative.py

3. Launches Hybrid System:
   - Once prerequisites are met, runs hybrid.py
"""

import sys
import subprocess
from pathlib import Path
import time

# Configuration
BASE_DIR = Path(__file__).parent.parent
CODE_DIR = BASE_DIR / "code"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# File Paths
DATA_FILES = [
    DATA_DIR / "final_ratings.csv",
    DATA_DIR / "final_items_enriched.csv"
]

CB_MODEL = RESULTS_DIR / "content_based_model.pkl"
CF_MODELS = [
    RESULTS_DIR / "collaborative_model.pkl",
    RESULTS_DIR / "svd_model.pkl",
    RESULTS_DIR / "svd_predictions.npy"
]

def run_script(script_name, description):
    """Run a python script as a subprocess."""
    script_path = CODE_DIR / script_name
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_path.name}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    try:
        # Run using the same python interpreter
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=BASE_DIR.parent  # Run from project root to match verified working directory
        )
        duration = time.time() - start_time
        print(f"\n[SUCCESS] {script_name} completed in {duration:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {script_name} failed with exit code {e.returncode}")
        return False

def check_data():
    """Check if data files exist, run preprocessing if not."""
    print("Checking datasets...")
    missing = [f.name for f in DATA_FILES if not f.exists()]
    
    if missing:
        print(f"[MISSING] Datasets not found: {', '.join(missing)}")
        print("-> Initiating Data Preprocessing Pipeline...")
        return run_script("data_preprocessing.py", "Data Preprocessing & Merging")
    else:
        print("[OK] Datasets found.")
        return True

def check_models():
    """Check if models exist, run training if not."""
    print("Checking models...")
    
    # 1. Content-Based
    if not CB_MODEL.exists():
        print(f"[MISSING] Content-Based model not found: {CB_MODEL.name}")
        print("-> Training Content-Based Model...")
        if not run_script("content_based.py", "Content-Based Model Training"):
            return False
    else:
        print(f"[OK] Content-Based model found.")

    # 2. Collaborative
    missing_cf = [f.name for f in CF_MODELS if not f.exists()]
    if missing_cf:
        print(f"[MISSING] Collaborative models not found: {', '.join(missing_cf)}")
        print("-> Training Collaborative & SVD Models...")
        if not run_script("collaborative.py", "Collaborative Filtering & SVD Training"):
            return False
    else:
        print(f"[OK] Collaborative models found.")
        
    return True

def main():
    print("\n" + "="*60)
    print("SECTION 2: DOMAIN RECOMMENDER SYSTEM - ORCHESTRATOR")
    print("="*60)
    
    # Step 1: Data
    if not check_data():
        print("[FATAL] Data preparation failed. Exiting.")
        sys.exit(1)
        
    # Step 2: Models
    if not check_models():
        print("[FATAL] Model training failed. Exiting.")
        sys.exit(1)
        
    # Step 3: Hybrid System
    print("\n" + "="*60)
    print("ALL SYSTEMS READY -> LAUNCHING HYBRID RECOMMENDER")
    print("="*60)
    run_script("hybrid.py", "Hybrid Recommendation System")

if __name__ == "__main__":
    main()
