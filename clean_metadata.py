
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path('/home/yousef/AIE425_FinalProject_GroupX')
METADATA_FILE = BASE_DIR / 'SECTION2_DomainRecommender/data/game_metadata.csv'

def clean_metadata():
    print(f"Reading from: {METADATA_FILE}")
    
    if not METADATA_FILE.exists():
        print("Error: Metadata file not found.")
        return

    try:
        df = pd.read_csv(METADATA_FILE)
        initial_count = len(df)
        print(f"Initial rows: {initial_count}")

        # Remove rows where igdb_name is NaN or empty
        # We also treat string 'nan' as null just in case
        df['igdb_name'] = df['igdb_name'].replace('nan', pd.NA)
        df_clean = df.dropna(subset=['igdb_name'])
        
        # Filter out empty strings if any
        df_clean = df_clean[df_clean['igdb_name'].astype(str).str.strip() != '']

        removed_count = initial_count - len(df_clean)
        print(f"Removed {removed_count} rows with missing igdb_name.")

        # Sort by game_name
        df_clean = df_clean.sort_values(by='game_name', key=lambda col: col.str.lower())
        print("Sorted by game_name.")

        # Save back
        df_clean.to_csv(METADATA_FILE, index=False)
        print(f"Saved cleaned file to: {METADATA_FILE}")
        print(f"Final rows: {len(df_clean)}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    clean_metadata()
