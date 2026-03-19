import pandas as pd
from pathlib import Path

data_dir = Path("data")

files = [
    "Tournament Matchups.csv",
    "Tournament Simulation.csv",
    "Public Picks.csv",
    "KenPom Barttorvik.csv",
    "EvanMiya.csv",
    "Resumes.csv",
]

for file_name in files:
    path = data_dir / file_name
    df = pd.read_csv(path)
    print("\n" + "=" * 60)
    print(file_name)
    print("shape:", df.shape)
    print("columns:", list(df.columns[:12]))

    if "YEAR" in df.columns:
        years = sorted(df["YEAR"].dropna().unique().tolist())
        print("years:", years[:10])

    preview_cols = [c for c in ["YEAR", "TEAM", "TEAM NO", "SEED", "ROUND"] if c in df.columns]
    if preview_cols:
        print(df[preview_cols].head(5).to_string(index=False))