import pandas as pd
from pathlib import Path
import joblib

data_dir = Path("data")

artifact = joblib.load(data_dir / "best_logreg_model.joblib")
model = artifact["model"]
feature_cols = artifact["feature_cols"]

teams = pd.read_csv(data_dir / "teams_2026_clean.csv")
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

matchups["team1_clean"] = matchups["team1"].str.strip().str.lower().replace(name_map)
matchups["team2_clean"] = matchups["team2"].str.strip().str.lower().replace(name_map)

lookup = teams.set_index("TEAM_CLEAN")

results = []
missing = []

for _, row in matchups.iterrows():
    t1_key = row["team1_clean"]
    t2_key = row["team2_clean"]

    if "/" in t1_key or "/" in t2_key:
        missing.append(f"{row['team1']} vs {row['team2']}  <-- First Four placeholder")
        continue

    if t1_key not in lookup.index:
        missing.append(row["team1"])
        continue
    if t2_key not in lookup.index:
        missing.append(row["team2"])
        continue

    t1 = lookup.loc[t1_key]
    t2 = lookup.loc[t2_key]

    features = {
        "kadj_em_diff": t1["KADJ EM"] - t2["KADJ EM"],
        "kadj_o_diff": t1["KADJ O"] - t2["KADJ O"],
        "kadj_d_diff": t1["KADJ D"] - t2["KADJ D"],
        "barthag_diff": t1["BARTHAG"] - t2["BARTHAG"],
        "elite_sos_diff": t1["ELITE SOS"] - t2["ELITE SOS"],
        "talent_diff": t1["TALENT"] - t2["TALENT"],
        "exp_diff": t1["EXP"] - t2["EXP"],
        "o_rate_diff": t1["O RATE"] - t2["O RATE"],
        "d_rate_diff": t1["D RATE"] - t2["D RATE"],
        "relative_rating_diff": t1["RELATIVE RATING"] - t2["RELATIVE RATING"],
        "injury_rank_diff": t1["INJURY RANK"] - t2["INJURY RANK"],
        "roster_rank_diff": t1["ROSTER RANK"] - t2["ROSTER RANK"],
        "net_rpi_diff": t1["NET RPI"] - t2["NET RPI"],
        "resume_diff": t1["RESUME"] - t2["RESUME"],
        "wab_rank_diff": t1["WAB RANK"] - t2["WAB RANK"],
        "elo_diff": t1["ELO"] - t2["ELO"],
        "b_power_diff": t1["B POWER"] - t2["B POWER"],
    }

    X_game = pd.DataFrame([features])[feature_cols].fillna(0)
    p_team1 = float(model.predict_proba(X_game)[0, 1])
    p_team2 = 1.0 - p_team1

    if p_team1 >= p_team2:
        pick = t1["TEAM"]
        confidence = p_team1
    else:
        pick = t2["TEAM"]
        confidence = p_team2

    results.append({
        "game_id": row["game_id"],
        "team1": t1["TEAM"],
        "seed1": int(t1["SEED"]),
        "team2": t2["TEAM"],
        "seed2": int(t2["SEED"]),
        "team1_win_prob": round(p_team1, 4),
        "team2_win_prob": round(p_team2, 4),
        "recommended_pick": pick,
        "pick_confidence": round(confidence, 4),
    })

results_df = pd.DataFrame(results).sort_values("game_id")

print("\nROUND 1 BEST-MODEL PREDICTIONS\n")
if not results_df.empty:
    print(results_df.to_string(index=False))
else:
    print("No predictions created.")

if missing:
    print("\nSTILL UNMATCHED OR SKIPPED:")
    for name in sorted(set(missing)):
        print("-", name)

output_path = data_dir / "round1_predictions_best.csv"
results_df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")