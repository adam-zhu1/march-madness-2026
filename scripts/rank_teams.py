import pandas as pd
from pathlib import Path

data_path = Path("data/teams_2026_clean.csv")
df = pd.read_csv(data_path)

# Higher is better for these
good_cols = [
    "KADJ EM",
    "KADJ O",
    "BARTHAG",
    "ELITE SOS",
    "TALENT",
    "EXP",
    "O RATE",
    "RELATIVE RATING",
    "RESUME",
    "ELO",
]

# Lower is better for these
bad_cols = [
    "KADJ D",
    "D RATE",
    "NET RPI",
    "WAB RANK",
    "INJURY RANK",
    "ROSTER RANK",
]

for col in good_cols:
    df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

for col in bad_cols:
    df[col + "_z"] = -1 * (df[col] - df[col].mean()) / df[col].std(ddof=0)

df["model_score"] = (
    1.8 * df["KADJ EM_z"]
    + 1.2 * df["BARTHAG_z"]
    + 1.0 * df["RELATIVE RATING_z"]
    + 0.8 * df["ELO_z"]
    + 0.7 * df["RESUME_z"]
    + 0.5 * df["ELITE SOS_z"]
    + 0.5 * df["TALENT_z"]
    + 0.3 * df["EXP_z"]
    + 0.8 * df["KADJ D_z"]
    + 0.5 * df["D RATE_z"]
    + 0.4 * df["NET RPI_z"]
    + 0.3 * df["WAB RANK_z"]
    + 0.2 * df["INJURY RANK_z"]
)

df = df.sort_values("model_score", ascending=False).reset_index(drop=True)

print("\nTop 20 title candidates:\n")
print(
    df[["TEAM", "SEED", "model_score", "KADJ EM", "RELATIVE RATING", "ELO", "NET RPI"]]
    .head(20)
    .to_string(index=False)
)

output_path = Path("data/team_rankings_2026.csv")
df.to_csv(output_path, index=False)
print(f"\nSaved rankings to: {output_path}")