import pandas as pd
from pathlib import Path

data_dir = Path("data")

matchups = pd.read_csv(data_dir / "Tournament Matchups.csv")
kenpom = pd.read_csv(data_dir / "KenPom Barttorvik.csv")
evan = pd.read_csv(data_dir / "EvanMiya.csv")
resumes = pd.read_csv(data_dir / "Resumes.csv")

# Keep only years where all main data sources exist
years = sorted(
    set(matchups["YEAR"].dropna().unique())
    & set(kenpom["YEAR"].dropna().unique())
    & set(evan["YEAR"].dropna().unique())
    & set(resumes["YEAR"].dropna().unique())
)

print("Shared years:", years)

all_games = []

for year in years:
    m = matchups[matchups["YEAR"] == year].copy()
    k = kenpom[kenpom["YEAR"] == year].copy()
    e = evan[evan["YEAR"] == year].copy()
    r = resumes[resumes["YEAR"] == year].copy()

    team_features = k[[
        "TEAM NO", "TEAM", "SEED", "KADJ EM", "KADJ O", "KADJ D", "BARTHAG", "ELITE SOS", "TALENT", "EXP"
    ]].merge(
        e[[
            "TEAM NO", "O RATE", "D RATE", "RELATIVE RATING", "INJURY RANK", "ROSTER RANK"
        ]],
        on="TEAM NO",
        how="left"
    ).merge(
        r[[
            "TEAM NO", "NET RPI", "RESUME", "WAB RANK", "ELO", "B POWER"
        ]],
        on="TEAM NO",
        how="left"
    )

    feature_lookup = team_features.set_index("TEAM NO")

    for current_round in sorted(m["CURRENT ROUND"].dropna().unique()):
        grp = m[m["CURRENT ROUND"] == current_round].copy().reset_index(drop=True)

        if len(grp) % 2 != 0:
            print(f"Skipping odd row count: year={year}, current_round={current_round}, rows={len(grp)}")
            continue

        for i in range(0, len(grp), 2):
            a = grp.iloc[i]
            b = grp.iloc[i + 1]

            team1_no = a["TEAM NO"]
            team2_no = b["TEAM NO"]

            if team1_no not in feature_lookup.index or team2_no not in feature_lookup.index:
                continue

            f1 = feature_lookup.loc[team1_no]
            f2 = feature_lookup.loc[team2_no]

            # Infer winner from ROUND vs CURRENT ROUND
            # Team that advanced should have ROUND < CURRENT ROUND
            team1_won = None
            if a["ROUND"] < current_round and b["ROUND"] == current_round:
                team1_won = 1
            elif b["ROUND"] < current_round and a["ROUND"] == current_round:
                team1_won = 0
            else:
                continue

            row = {
                "year": year,
                "current_round": current_round,
                "team1": a["TEAM"],
                "team2": b["TEAM"],
                "seed1": a["SEED"],
                "seed2": b["SEED"],
                "team1_won": team1_won,

                "kadj_em_diff": f1["KADJ EM"] - f2["KADJ EM"],
                "kadj_o_diff": f1["KADJ O"] - f2["KADJ O"],
                "kadj_d_diff": f1["KADJ D"] - f2["KADJ D"],
                "barthag_diff": f1["BARTHAG"] - f2["BARTHAG"],
                "elite_sos_diff": f1["ELITE SOS"] - f2["ELITE SOS"],
                "talent_diff": f1["TALENT"] - f2["TALENT"],
                "exp_diff": f1["EXP"] - f2["EXP"],

                "o_rate_diff": f1["O RATE"] - f2["O RATE"],
                "d_rate_diff": f1["D RATE"] - f2["D RATE"],
                "relative_rating_diff": f1["RELATIVE RATING"] - f2["RELATIVE RATING"],
                "injury_rank_diff": f1["INJURY RANK"] - f2["INJURY RANK"],
                "roster_rank_diff": f1["ROSTER RANK"] - f2["ROSTER RANK"],

                "net_rpi_diff": f1["NET RPI"] - f2["NET RPI"],
                "resume_diff": f1["RESUME"] - f2["RESUME"],
                "wab_rank_diff": f1["WAB RANK"] - f2["WAB RANK"],
                "elo_diff": f1["ELO"] - f2["ELO"],
                "b_power_diff": f1["B POWER"] - f2["B POWER"],
            }

            all_games.append(row)

historical = pd.DataFrame(all_games)

print("\nBuilt historical matchup table")
print("shape:", historical.shape)
print("\nColumns:")
print(list(historical.columns))
print("\nPreview:")
print(historical.head(20).to_string(index=False))

output_path = data_dir / "historical_matchups.csv"
historical.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")