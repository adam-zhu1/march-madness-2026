import pandas as pd
from pathlib import Path

data_dir = Path("data")

sim = pd.read_csv(data_dir / "Tournament Simulation.csv")

sim_2026 = sim[sim["YEAR"] == 2026].copy()

print("Shape:", sim_2026.shape)
print("\nColumns:")
print(list(sim_2026.columns))

print("\nFirst 20 rows:")
print(sim_2026.head(20).to_string(index=False))