import pandas as pd
from pathlib import Path

data_dir = Path("data")

matchups = pd.read_csv(data_dir / "Tournament Matchups.csv")
kenpom = pd.read_csv(data_dir / "KenPom Barttorvik.csv")
evanmiya = pd.read_csv(data_dir / "EvanMiya.csv")
resumes = pd.read_csv(data_dir / "Resumes.csv")
public = pd.read_csv(data_dir / "Public Picks.csv")

# Filter to 2026
matchups_2026 = matchups[matchups["YEAR"] == 2026].copy()
kenpom_2026 = kenpom[kenpom["YEAR"] == 2026].copy()
evan_2026 = evanmiya[evanmiya["YEAR"] == 2026].copy()
resumes_2026 = resumes[resumes["YEAR"] == 2026].copy()
public_2026 = public[public["YEAR"] == 2026].copy()

# Start with tournament teams only
teams = matchups_2026[["TEAM", "TEAM NO", "SEED"]].drop_duplicates().copy()

# Merge key model inputs
teams = teams.merge(
    kenpom_2026[
        ["TEAM NO", "KADJ EM", "KADJ O", "KADJ D", "BARTHAG", "ELITE SOS", "TALENT", "EXP"]
    ],
    on="TEAM NO",
    how="left",
)

teams = teams.merge(
    evan_2026[
        ["TEAM NO", "O RATE", "D RATE", "RELATIVE RATING", "INJURY RANK", "ROSTER RANK"]
    ],
    on="TEAM NO",
    how="left",
)

teams = teams.merge(
    resumes_2026[
        ["TEAM NO", "NET RPI", "RESUME", "WAB RANK", "ELO", "B POWER", "Q1 W", "Q2 W"]
    ],
    on="TEAM NO",
    how="left",
)

teams = teams.merge(
    public_2026[
        ["TEAM NO", "R64", "R32", "S16", "E8", "F4", "FINALS"]
    ],
    on="TEAM NO",
    how="left",
)

teams = teams.sort_values(["SEED", "TEAM"]).reset_index(drop=True)

print("\nMerged 2026 tournament table")
print("shape:", teams.shape)
print("\nMissing values by column:")
print(teams.isna().sum())

print("\nPreview:")
print(teams.head(20).to_string(index=False))

output_path = data_dir / "teams_2026_clean.csv"
teams.to_csv(output_path, index=False)

print(f"\nSaved to: {output_path}")