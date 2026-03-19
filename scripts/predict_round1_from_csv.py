import math
import pandas as pd
from pathlib import Path

data_dir = Path("data")

teams = pd.read_csv(data_dir / "team_rankings_2026.csv")
matchups = pd.read_csv(data_dir / "round1_matchups_2026.csv")

name_map = {
    "hawai'i": "hawaii",
    "mcneese": "mcneese st.",
    "long island university": "liu brooklyn",
    "saint mary's": "saint mary's",
    "miami (fla.)": "miami fl",
    "north carolina": "north carolina",
    "utah state": "utah st.",
    "wright state": "wright st.",
    "tennessee state": "tennessee st.",
    "kennesaw state": "kennesaw st.",
    "ohio state": "ohio st.",
}

teams["TEAM_CLEAN"] = teams["TEAM"].str.strip().str.lower()
teams = teams.drop_duplicates(subset=["TEAM_CLEAN"]).copy()

matchups["team1_clean"] = matchups["team1"].str.strip().str.lower()
matchups["team2_clean"] = matchups["team2"].str.strip().str.lower()

matchups["team1_clean"] = matchups["team1_clean"].replace(name_map)
matchups["team2_clean"] = matchups["team2_clean"].replace(name_map)

lookup = teams.set_index("TEAM_CLEAN")[["TEAM", "SEED", "model_score"]]

results = []
missing = []

for _, row in matchups.iterrows():
    team1_key = row["team1_clean"]
    team2_key = row["team2_clean"]

    if "/" in team1_key or "/" in team2_key:
        missing.append(f"{row['team1']} vs {row['team2']}  <-- First Four placeholder")
        continue

    if team1_key not in lookup.index:
        missing.append(row["team1"])
        continue
    if team2_key not in lookup.index:
        missing.append(row["team2"])
        continue

    t1 = lookup.loc[team1_key]
    t2 = lookup.loc[team2_key]

    score_diff = float(t1["model_score"] - t2["model_score"])
    p1 = 1 / (1 + math.exp(-score_diff / 1.8))
    p2 = 1 - p1

    if p1 >= p2:
        pick = t1["TEAM"]
        confidence = p1
    else:
        pick = t2["TEAM"]
        confidence = p2

    results.append({
        "game_id": row["game_id"],
        "team1": t1["TEAM"],
        "seed1": int(t1["SEED"]),
        "team2": t2["TEAM"],
        "seed2": int(t2["SEED"]),
        "team1_win_prob": round(p1, 4),
        "team2_win_prob": round(p2, 4),
        "recommended_pick": pick,
        "pick_confidence": round(confidence, 4),
    })

results_df = pd.DataFrame(results).sort_values("game_id")

print("\nROUND 1 PREDICTIONS\n")
if not results_df.empty:
    print(results_df.to_string(index=False))
else:
    print("No predictions created.")

if missing:
    print("\nSTILL UNMATCHED OR SKIPPED:")
    for name in sorted(set(missing)):
        print("-", name)

output_path = data_dir / "round1_predictions.csv"
results_df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")