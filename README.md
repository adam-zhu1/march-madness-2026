# 🏀 March Madness 2026 Bracket Predictor

A data-driven NCAA men’s tournament prediction app built to help generate smarter bracket picks for the **2026 ESPN Tournament Challenge**.

This project combines historical tournament results, advanced team ratings, and a trained logistic regression model to estimate game win probabilities and help make bracket decisions one matchup at a time.

---

## ✨ Features

- **2026 Round of 64 predictions**
- **Matchup Predictor** for any two tournament teams
- **Power rankings** based on merged team metrics
- **Historical training pipeline** built from past NCAA tournament data
- **Year-aware logistic regression model**
- **Streamlit app UI** for easy bracket decision-making

---

## 📊 Data Used

This project uses a collection of CSV datasets including:

- KenPom / Barttorvik style ratings
- EvanMiya ratings
- Resume / NET / ELO / B Power data
- Tournament matchup history
- Team and seed results
- Public picks and simulation files where available

Main files used in the final pipeline include:

- `data/KenPom Barttorvik.csv`
- `data/EvanMiya.csv`
- `data/Resumes.csv`
- `data/Tournament Matchups.csv`
- `data/round1_matchups_2026.csv`

---

## 🧠 Modeling Approach

The project started with a hand-built weighted score model, then moved to a stronger **historical logistic regression approach**.

### Current workflow

1. Build a clean 2026 tournament team table  
2. Build historical tournament matchup rows  
3. Compute feature differences between two teams  
4. Train a logistic regression model on past tournaments  
5. Predict 2026 Round of 64 games  
6. Serve everything in a local Streamlit app  

### Features used in the model

- `kadj_em_diff`
- `kadj_o_diff`
- `kadj_d_diff`
- `barthag_diff`
- `elite_sos_diff`
- `talent_diff`
- `exp_diff`
- `o_rate_diff`
- `d_rate_diff`
- `relative_rating_diff`
- `injury_rank_diff`
- `roster_rank_diff`
- `net_rpi_diff`
- `resume_diff`
- `wab_rank_diff`
- `elo_diff`
- `b_power_diff`

---

## 📈 Best Model Results

The best year-aware logistic regression model achieved:

- **2025 holdout accuracy:** `0.7937`
- **2025 holdout log loss:** `0.4313`

This model is used for the current app predictions.

---

## ▶️ Running the App

Use this command in the terminal:

```bash
python3 -m streamlit run app.py
```

March Madness 2026 App/
├── app.py
├── requirements.txt
├── data/
│   ├── best_logreg_model.joblib
│   ├── best_logreg_coefficients.csv
│   ├── historical_matchups.csv
│   ├── round1_matchups_2026.csv
│   ├── round1_predictions_best.csv
│   ├── team_rankings_2026.csv
│   ├── teams_2026_clean.csv
│   └── ... other raw CSV files
├── scripts/
│   ├── build_2026_table.py
│   ├── build_historical_matchups.py
│   ├── check_data.py
│   ├── inspect_team_results.py
│   ├── predict_round1_best.py
│   ├── predict_round1_from_csv.py
│   ├── predict_round1_logreg.py
│   ├── rank_teams.py
│   ├── train_best_model.py
│   └── train_logreg_model.py
└── README.md
