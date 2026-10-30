"""Synthetic data with creditcard.csv's schema (Time, V1-V28, Amount, Class).
ONLY for smoke tests when you don't have the real Kaggle file."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42); n = 30000
df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
df.insert(0, "Time", np.sort(rng.integers(0, 172800, n)))
df["Amount"] = rng.lognormal(3.2, 1.3, n).round(2)
df["Class"] = (rng.random(n) < 0.005).astype(int)
f = df.Class == 1
shift = np.zeros(28); shift[[0, 1, 2, 3, 9, 10, 11, 13]] = [-3, 2.5, -3, 2.5, -3, 2, -3, -3]
df.loc[f, [f"V{i}" for i in range(1, 29)]] = rng.normal(shift, 1.6, (f.sum(), 28))
df.loc[f, "Amount"] = rng.lognormal(4.5, 1.5, f.sum()).round(2)
df.to_csv("data/creditcard.csv", index=False); print("fraud rate", df.Class.mean())
