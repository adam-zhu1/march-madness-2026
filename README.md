# 🏀 March Madness 2026 Bracket Predictor

A data-driven NCAA men’s tournament prediction app for making smarter bracket picks for the **2026 ESPN Tournament Challenge**.

It combines historical tournament results, advanced team metrics, and a **year-aware logistic regression model** to estimate game win probabilities—then serves the results in a simple **Streamlit** UI.

---

## ✨ What you get

- **Round of 64 picks** with win probabilities + recommended pick + confidence
- **Power rankings** built from merged team metrics
- **Matchup Predictor**: pick any two 2026 tournament teams and get a model forecast
- **Training + prediction pipeline** based on historical NCAA tournament matchups

---

## ▶️ Quick start

1. Install dependencies (you likely already have these):

```bash
python3 -m pip install -U streamlit pandas joblib
```

2. Run the app:

```bash
python3 -m streamlit run app.py
```

---

## 🧭 How the app works (`app.py`)

The Streamlit app loads precomputed CSV outputs from `data/` and exposes 4 tabs:

- **Power Rankings**: top teams sorted by `model_score`
- **Round of 64 Picks**: all first-round predictions + most confident picks + closest games
- **Matchup Predictor**: interactive “Team A vs Team B” probability from the trained model
- **Skipped Games**: matchups present in `round1_matchups_2026.csv` but missing from predictions (typically First Four placeholders)

---

## 📁 Key files & outputs

- **`app.py`**: Streamlit UI + interactive matchup predictor
- **`scripts/`**: build steps for 2026 tables / matchups (project pipeline utilities)
- **`data/best_logreg_model.joblib`**: trained model artifact containing:
  - `model` (logistic regression)
  - `feature_cols` (expected feature order)
- **`data/round1_predictions_best.csv`**: precomputed Round of 64 predictions used by the UI
- **`data/round1_matchups_2026.csv`**: Round of 64 bracket matchups (includes First Four placeholders until resolved)
- **`data/teams_2026_clean.csv`**: cleaned 2026 team table with model features used for interactive matchups
- **`data/team_rankings_2026.csv`**: power rankings table used in the Rankings tab

---

## 🧠 Model + features

The current approach is a **year-aware logistic regression** trained on historical NCAA tournament matchups.

It predicts games using **feature differences** (Team A minus Team B), including:

- `kadj_em_diff`, `kadj_o_diff`, `kadj_d_diff`
- `barthag_diff`, `elite_sos_diff`
- `talent_diff`, `exp_diff`
- `o_rate_diff`, `d_rate_diff`
- `relative_rating_diff`
- `injury_rank_diff`, `roster_rank_diff`
- `net_rpi_diff`, `resume_diff`, `wab_rank_diff`
- `elo_diff`, `b_power_diff`

---

## 🔁 Updating predictions (First Four → Round of 64)

If the app shows games in **Skipped Games**, you need to resolve the First Four placeholders and regenerate Round 1 predictions:

1. Update winners / resolved teams in `data/round1_matchups_2026.csv`
2. Rerun the prediction pipeline script(s) to regenerate `data/round1_predictions_best.csv`
3. Refresh the Streamlit app

---

## 📊 Data sources (high level)

The pipeline consumes CSV datasets containing team ratings and tournament history, such as:

- KenPom / Barttorvik style ratings
- EvanMiya ratings
- Resume / NET / ELO / B Power data
- Tournament matchup history and seeds

Examples referenced by the pipeline:

- `data/KenPom Barttorvik.csv`
- `data/EvanMiya.csv`
- `data/Resumes.csv`
- `data/Tournament Matchups.csv`
- `data/round1_matchups_2026.csv`

---

## ⚠️ Notes

- This is a **probabilistic model**, not a guarantee. Upsets are the fun part.
- If you change the underlying feature columns or input tables, you must retrain/re-export `best_logreg_model.joblib` so `feature_cols` matches what the app computes.
