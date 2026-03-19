import pandas as pd
import streamlit as st
from pathlib import Path
import joblib

st.set_page_config(page_title="2026 Bracket Predictor", layout="wide")

data_dir = Path("data")

rankings = pd.read_csv(data_dir / "team_rankings_2026.csv")
round1 = pd.read_csv(data_dir / "round1_predictions_best.csv")
matchups = pd.read_csv(data_dir / "round1_matchups_2026.csv")
teams = pd.read_csv(data_dir / "teams_2026_clean.csv")

artifact = joblib.load(data_dir / "best_logreg_model.joblib")
model = artifact["model"]
feature_cols = artifact["feature_cols"]

predicted_ids = set(round1["game_id"].tolist())
skipped = matchups[~matchups["game_id"].isin(predicted_ids)].copy()

st.title("🏀 2026 Men’s Bracket Predictor")
st.caption("Uses a year-aware logistic regression model trained on historical NCAA tournament matchups.")

top_team = rankings.sort_values("model_score", ascending=False).iloc[0]
safest_pick = round1.sort_values("pick_confidence", ascending=False).iloc[0]
closest_game = round1.sort_values("pick_confidence", ascending=True).iloc[0]

st.markdown("## Quick Summary")
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Top Power Team", f"{top_team['TEAM']} (Seed {int(top_team['SEED'])})")

with c2:
    st.metric(
        "Safest Round 1 Pick",
        f"{safest_pick['recommended_pick']}",
        f"{safest_pick['pick_confidence'] * 100:.1f}% confidence"
    )

with c3:
    st.metric(
        "Closest Round 1 Game",
        f"{closest_game['team1']} vs {closest_game['team2']}",
        f"{closest_game['pick_confidence'] * 100:.1f}% edge"
    )

tab1, tab2, tab3, tab4 = st.tabs([
    "Power Rankings",
    "Round of 64 Picks",
    "Matchup Predictor",
    "Skipped Games"
])

with tab1:
    st.subheader("Top Teams by Power Score")
    title_df = rankings[
        ["TEAM", "SEED", "model_score", "KADJ EM", "RELATIVE RATING", "ELO", "NET RPI"]
    ].sort_values("model_score", ascending=False).reset_index(drop=True)
    title_df.index = title_df.index + 1
    st.dataframe(title_df.head(25), width="stretch")

with tab2:
    st.subheader("Recommended Round of 64 Picks")
    display_df = round1.copy()
    display_df["team1_win_prob"] = (display_df["team1_win_prob"] * 100).round(1)
    display_df["team2_win_prob"] = (display_df["team2_win_prob"] * 100).round(1)
    display_df["pick_confidence"] = (display_df["pick_confidence"] * 100).round(1)

    display_df = display_df.rename(columns={
        "team1": "Team 1",
        "seed1": "Seed 1",
        "team2": "Team 2",
        "seed2": "Seed 2",
        "team1_win_prob": "Team 1 Win %",
        "team2_win_prob": "Team 2 Win %",
        "recommended_pick": "Recommended Pick",
        "pick_confidence": "Confidence %",
    })

    st.dataframe(display_df, width="stretch")

    st.markdown("### Most Confident Picks")
    st.dataframe(display_df.sort_values("Confidence %", ascending=False).head(12), width="stretch")

    st.markdown("### Closest Games")
    st.dataframe(display_df.sort_values("Confidence %", ascending=True).head(8), width="stretch")

with tab3:
    st.subheader("Matchup Predictor")
    st.caption("Choose any two tournament teams and the trained model will estimate who should win.")

    teams["TEAM_CLEAN"] = teams["TEAM"].str.strip().str.lower()
    teams = teams.drop_duplicates(subset=["TEAM_CLEAN"]).copy()
    lookup = teams.set_index("TEAM")

    team_options = sorted(teams["TEAM"].dropna().unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team 1", team_options, index=0)
    with col2:
        team_b = st.selectbox("Team 2", team_options, index=1)

    if team_a == team_b:
        st.warning("Choose two different teams.")
    else:
        a = lookup.loc[team_a]
        b = lookup.loc[team_b]

        features = {
            "kadj_em_diff": a["KADJ EM"] - b["KADJ EM"],
            "kadj_o_diff": a["KADJ O"] - b["KADJ O"],
            "kadj_d_diff": a["KADJ D"] - b["KADJ D"],
            "barthag_diff": a["BARTHAG"] - b["BARTHAG"],
            "elite_sos_diff": a["ELITE SOS"] - b["ELITE SOS"],
            "talent_diff": a["TALENT"] - b["TALENT"],
            "exp_diff": a["EXP"] - b["EXP"],
            "o_rate_diff": a["O RATE"] - b["O RATE"],
            "d_rate_diff": a["D RATE"] - b["D RATE"],
            "relative_rating_diff": a["RELATIVE RATING"] - b["RELATIVE RATING"],
            "injury_rank_diff": a["INJURY RANK"] - b["INJURY RANK"],
            "roster_rank_diff": a["ROSTER RANK"] - b["ROSTER RANK"],
            "net_rpi_diff": a["NET RPI"] - b["NET RPI"],
            "resume_diff": a["RESUME"] - b["RESUME"],
            "wab_rank_diff": a["WAB RANK"] - b["WAB RANK"],
            "elo_diff": a["ELO"] - b["ELO"],
            "b_power_diff": a["B POWER"] - b["B POWER"],
        }

        X_game = pd.DataFrame([features])[feature_cols].fillna(0)
        p_a = float(model.predict_proba(X_game)[0, 1])
        p_b = 1.0 - p_a

        winner = team_a if p_a >= p_b else team_b
        confidence = max(p_a, p_b) * 100

        st.success(f"{winner} is favored.")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(f"{team_a} win probability", f"{p_a * 100:.1f}%")
        with m2:
            st.metric(f"{team_b} win probability", f"{p_b * 100:.1f}%")
        with m3:
            st.metric("Model confidence", f"{confidence:.1f}%")

        comparison = pd.DataFrame({
            "Team": [team_a, team_b],
            "Seed": [int(a["SEED"]), int(b["SEED"])],
            "KADJ EM": [round(a["KADJ EM"], 3), round(b["KADJ EM"], 3)],
            "KADJ O": [round(a["KADJ O"], 3), round(b["KADJ O"], 3)],
            "KADJ D": [round(a["KADJ D"], 3), round(b["KADJ D"], 3)],
            "Relative Rating": [round(a["RELATIVE RATING"], 3), round(b["RELATIVE RATING"], 3)],
            "ELO": [a["ELO"], b["ELO"]],
            "NET RPI": [a["NET RPI"], b["NET RPI"]],
        })

        st.dataframe(comparison, width="stretch")

with tab4:
    st.subheader("Games Not Predicted Yet")
    st.caption("These are the First Four placeholder matchups you still need to resolve.")
    if skipped.empty:
        st.success("No skipped games.")
    else:
        st.dataframe(skipped, width="stretch")

st.markdown("---")
st.markdown("Workflow: update First Four winners in `data/round1_matchups_2026.csv`, rerun prediction script, refresh app.")