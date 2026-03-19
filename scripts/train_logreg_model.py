import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

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

# Fill missing values from older seasons
X = df[feature_cols].fillna(0).copy()
y = df["team1_won"].copy()

# Random split for now
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)

acc = accuracy_score(y_test, test_preds)
ll = log_loss(y_test, test_probs)

print("\nModel evaluation")
print(f"Test accuracy: {acc:.4f}")
print(f"Test log loss: {ll:.4f}")

coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": model.coef_[0]
}).sort_values("coefficient", ascending=False)

print("\nTop positive coefficients:")
print(coef_df.head(10).to_string(index=False))

print("\nTop negative coefficients:")
print(coef_df.tail(10).to_string(index=False))

# Save coefficients for inspection
coef_df.to_csv(data_dir / "logreg_coefficients.csv", index=False)

print("\nSaved coefficients to: data/logreg_coefficients.csv")