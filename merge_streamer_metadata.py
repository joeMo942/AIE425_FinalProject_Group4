"""
Merge streamer_metadata.csv with streamer_metadata_fixed.csv
Logic: For each streamer, if data is missing in the main file, fill from fixed file.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path('/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/data')
MAIN_FILE = DATA_DIR / 'streamer_metadata.csv'
FIXED_FILE = DATA_DIR / 'streamer_metadata_fixed.csv'

def main():
    print("=" * 60)
    print("MERGING STREAMER METADATA FILES")
    print("=" * 60)
    
    # Load both files
    df_main = pd.read_csv(MAIN_FILE)
    df_fixed = pd.read_csv(FIXED_FILE)
    
    print(f"[LOADED] Main file: {len(df_main)} rows")
    print(f"[LOADED] Fixed file: {len(df_fixed)} rows")
    
    # Normalize NAME column for matching
    df_main['NAME'] = df_main['NAME'].astype(str).str.lower().str.strip()
    df_fixed['NAME'] = df_fixed['NAME'].astype(str).str.lower().str.strip()
    
    # Create a lookup from fixed file
    fixed_lookup = df_fixed.set_index('NAME').to_dict('index')
    
    # Track stats
    filled_count = 0
    added_count = 0
    
    # For each row in main, fill missing values from fixed
    for idx, row in df_main.iterrows():
        name = row['NAME']
        if name in fixed_lookup:
            fixed_row = fixed_lookup[name]
            for col in df_main.columns:
                if col == 'NAME':
                    continue
                # If main value is null/empty, use fixed value
                main_val = row[col]
                if pd.isna(main_val) or (isinstance(main_val, str) and main_val.strip() == ''):
                    fixed_val = fixed_row.get(col)
                    if pd.notna(fixed_val) and not (isinstance(fixed_val, str) and fixed_val.strip() == ''):
                        df_main.at[idx, col] = fixed_val
                        filled_count += 1
    
    # Add streamers that exist in fixed but not in main
    main_names = set(df_main['NAME'])
    for name, fixed_row in fixed_lookup.items():
        if name not in main_names:
            # Add this row to main
            new_row = {'NAME': name}
            for col in df_main.columns:
                if col == 'NAME':
                    continue
                new_row[col] = fixed_row.get(col)
            df_main = pd.concat([df_main, pd.DataFrame([new_row])], ignore_index=True)
            added_count += 1
    
    print(f"\n[MERGED] Filled {filled_count} missing values from fixed file")
    print(f"[MERGED] Added {added_count} new streamers from fixed file")
    print(f"[RESULT] Final row count: {len(df_main)}")
    
    # Save merged file
    df_main.to_csv(MAIN_FILE, index=False)
    print(f"\n[SAVED] {MAIN_FILE.name}")
    
    # Delete fixed file
    FIXED_FILE.unlink()
    print(f"[DELETED] {FIXED_FILE.name}")
    
    print("\n" + "=" * 60)
    print("[DONE] Merge complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
