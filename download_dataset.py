"""
Download German Credit Data from UCI and save it as data/german_credit.csv.
Run this once before opening the main notebook.

Usage (from the Source Code folder):
    python download_dataset.py
"""

import subprocess
import sys
import os

# Install ucimlrepo if not already installed
subprocess.check_call([sys.executable, "-m", "pip", "install", "ucimlrepo", "--quiet"])

from ucimlrepo import fetch_ucirepo
import pandas as pd

print("Fetching German Credit Data from UCI...")
german = fetch_ucirepo(id=144)

X = german.data.features
y = german.data.targets

df = pd.concat([X, y], axis=1)
df.columns = [*X.columns, "target"]

# Recode target: 1 = Good credit (0), 2 = Bad credit (1)
df["target"] = df["target"].map({1: 0, 2: 1})

# Save to data/ subfolder
os.makedirs("data", exist_ok=True)
out_path = os.path.join("data", "german_credit.csv")
df.to_csv(out_path, index=False)

print(f"Saved to: {out_path}")
print(f"Shape: {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts()}")
print("\nFirst 3 rows:")
print(df.head(3))
