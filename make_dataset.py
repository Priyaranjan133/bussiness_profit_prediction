"""Generate the sample dataset (50 startups style). Replace data/startups.csv with your own file
that has the columns: R&D Spend, Administration, Marketing Spend, Profit (State is optional)."""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
n = 50
rd = rng.uniform(0, 165_000, n).round(2)
admin = rng.uniform(51_000, 182_000, n).round(2)
mkt = np.clip(rng.normal(215_000, 120_000, n), 0, 472_000).round(2)
state = rng.choice(["New York", "California", "Florida"], n)
profit = (50_000 + 0.80 * rd - 0.02 * admin + 0.03 * mkt + rng.normal(0, 9_000, n)).round(2)
df = pd.DataFrame({"R&D Spend": rd, "Administration": admin, "Marketing Spend": mkt,
                   "State": state, "Profit": profit})
df.to_csv("data/startups.csv", index=False)
print(df.head(), df.shape)
