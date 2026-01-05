"""
Merge streamer_metadata.csv with datasetV2.csv
Logic: Add streamers from datasetV2 that don't exist in streamer_metadata. 
       Leave RANK blank for added streamers. Delete datasetV2 afterward.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path('/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/data')
MAIN_FILE = DATA_DIR / 'streamer_metadata.csv'
ARCHIVE_FILE = DATA_DIR / 'datasetV2.csv'

def main():
    print("=" * 60)
    print("MERGING STREAMER METADATA WITH DATASETV2")
    print("=" * 60)
    
    # Load both files
    df_main = pd.read_csv(MAIN_FILE)
    df_archive = pd.read_csv(ARCHIVE_FILE)
    
    print(f"[LOADED] Main file: {len(df_main)} rows, columns: {list(df_main.columns)}")
    print(f"[LOADED] Archive file: {len(df_archive)} rows")
    
    # Normalize NAME column for matching
    df_main['NAME'] = df_main['NAME'].astype(str).str.lower().str.strip()
    df_archive['NAME'] = df_archive['NAME'].astype(str).str.lower().str.strip()
    
    # Get existing names in main file
    existing_names = set(df_main['NAME'])
    
    # Columns in main file (target schema)
    main_columns = list(df_main.columns)
    
    # Map archive columns to main columns (only keep what exists in main)
    # Common columns between the two files
    common_cols = [c for c in main_columns if c in df_archive.columns]
    print(f"[INFO] Common columns: {common_cols}")
    
    # Find streamers in archive that don't exist in main
    new_streamers = df_archive[~df_archive['NAME'].isin(existing_names)].copy()
    print(f"[INFO] Found {len(new_streamers)} new streamers in archive")
    
    # Build rows for new streamers with main file schema
    new_rows = []
    for _, row in new_streamers.iterrows():
        new_row = {}
        for col in main_columns:
            if col == 'RANK':
                new_row[col] = None  # Leave RANK blank
            elif col in df_archive.columns:
                new_row[col] = row[col]
            else:
                new_row[col] = None
        new_rows.append(new_row)
    
    # Add new rows to main dataframe
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_main = pd.concat([df_main, df_new], ignore_index=True)
        print(f"[MERGED] Added {len(new_rows)} new streamers")
    
    print(f"[RESULT] Final row count: {len(df_main)}")
    
    # Save merged file
    df_main.to_csv(MAIN_FILE, index=False)
    print(f"\n[SAVED] {MAIN_FILE.name}")
    
    # Delete archive file
    ARCHIVE_FILE.unlink()
    print(f"[DELETED] {ARCHIVE_FILE.name}")
    
    print("\n" + "=" * 60)
    print("[DONE] Merge complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
