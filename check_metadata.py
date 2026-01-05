
import pandas as pd

file_path = '/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/data/game_metadata.csv'

try:
    df = pd.read_csv(file_path)
    print(f"Total rows: {len(df)}")
    
    # Check for exact duplicates
    exact_dups = df[df.duplicated(subset=['game_name'], keep=False)]
    if not exact_dups.empty:
        print("\nExact Duplicates in game_name:")
        print(exact_dups['game_name'].value_counts())
    
    # Check for whitespace/case duplicates
    df['normalized_name'] = df['game_name'].astype(str).str.strip().str.lower()
    norm_dups = df[df.duplicated(subset=['normalized_name'], keep=False)]
    if not norm_dups.empty:
        print("\nNormalized Duplicates (stripped & lowercase):")
        print(norm_dups[['game_name', 'normalized_name']].sort_values('normalized_name'))
    else:
        print("\nNo normalized duplicates found.")

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

except Exception as e:
    print(f"Error reading file: {e}")
