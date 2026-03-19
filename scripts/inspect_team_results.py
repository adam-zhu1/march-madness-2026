import pandas as pd
from pathlib import Path

data_dir = Path("data")

files = [
    "Team Results.csv",
    "Tournament Matchups.csv",
    "Seed Results.csv",
    "Coach Results.csv",
    "Conference Results.csv",
]

for file_name in files:
    print("\n" + "=" * 80)
    print(file_name)
    path = data_dir / file_name
    df = pd.read_csv(path)

    print("shape:", df.shape)
    print("columns:", list(df.columns))

    preview_cols = [c for c in ["YEAR", "TEAM", "TEAM NO", "SEED", "ROUND", "CURRENT ROUND", "SCORE", "W", "L"] if c in df.columns]
    if preview_cols:
        print("\npreview:")
        print(df[preview_cols].head(10).to_string(index=False))
    else:
        print("\npreview:")
        print(df.head(10).to_string(index=False))