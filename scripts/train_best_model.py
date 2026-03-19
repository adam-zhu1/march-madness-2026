import pandas as pd
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, log_loss

data_dir = Path("data")

df = pd.read_csv(data_dir / "historical_matchups.csv")

feature_cols = [
    "kadj_em_diff",
    "kadj_o_diff",
    "kadj_d_diff",
    "barthag_diff",
    "elite_sos_diff",
    "talent_diff",
    "exp_diff",
    "o_rate_diff",
    "d_rate_diff",
    "relative_rating_diff",
    "injury_rank_diff",
    "roster_rank_diff",
    "net_rpi_diff",
    "resume_diff",
    "wab_rank_diff",
    "elo_diff",
    "b_power_diff",
]

# Use 2011-2024 for tuning, 2025 as a true holdout year, then refit on 2011-2025
tune_df = df[df["year"] < 2025].copy()
holdout_df = df[df["year"] == 2025].copy()
final_df = df[df["year"] < 2026].copy()

X_tune = tune_df[feature_cols].fillna(0)
y_tune = tune_df["team1_won"]
groups = tune_df["year"]

X_holdout = holdout_df[feature_cols].fillna(0)
y_holdout = holdout_df["team1_won"]

X_final = final_df[feature_cols].fillna(0)
y_final = final_df["team1_won"]

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(solver="liblinear", max_iter=5000)),
])

param_grid = {
    "logreg__penalty": ["l1", "l2"],
    "logreg__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
}

cv = LeaveOneGroupOut()

search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="neg_log_loss",
    cv=cv,
    n_jobs=-1,
    refit=True,
)

search.fit(X_tune, y_tune, groups=groups)

best_model = search.best_estimator_

print("\nBest params:")
print(search.best_params_)
print(f"Best CV neg log loss: {search.best_score_:.6f}")

if len(holdout_df) > 0:
    holdout_probs = best_model.predict_proba(X_holdout)[:, 1]
    holdout_preds = (holdout_probs >= 0.5).astype(int)

    holdout_acc = accuracy_score(y_holdout, holdout_preds)
    holdout_ll = log_loss(y_holdout, holdout_probs)

    print("\n2025 holdout evaluation")
    print(f"Holdout accuracy: {holdout_acc:.4f}")
    print(f"Holdout log loss: {holdout_ll:.4f}")
else:
    print("\nNo 2025 holdout rows found.")

# Refit final model on all pre-2026 data
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        solver="liblinear",
        max_iter=5000,
        penalty=search.best_params_["logreg__penalty"],
        C=search.best_params_["logreg__C"],
    )),
])

final_model.fit(X_final, y_final)

coef_model = final_model.named_steps["logreg"]
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": coef_model.coef_[0]
}).sort_values("coefficient", ascending=False)

print("\nTop positive coefficients:")
print(coef_df.head(10).to_string(index=False))

print("\nTop negative coefficients:")
print(coef_df.tail(10).to_string(index=False))

artifact = {
    "model": final_model,
    "feature_cols": feature_cols,
    "best_params": search.best_params_,
}

joblib.dump(artifact, data_dir / "best_logreg_model.joblib")
coef_df.to_csv(data_dir / "best_logreg_coefficients.csv", index=False)

print("\nSaved model to: data/best_logreg_model.joblib")
print("Saved coefficients to: data/best_logreg_coefficients.csv")