import pandas as pd
import os

# --- File paths
bond_path = "data/india_10y_yield.csv"
macro_path = "data/india_macro.csv"

# --- Check if files exist
if not os.path.exists(bond_path):
    raise FileNotFoundError(f"File not found: {bond_path}")
if not os.path.exists(macro_path):
    raise FileNotFoundError(f"File not found: {macro_path}")

# --- Load data
bond_df = pd.read_csv(bond_path)
macro_df = pd.read_csv(macro_path)

# --- Preview
print("Loaded bond yield data:")
print(bond_df.head())

print("\nLoaded macroeconomic data:")
print(macro_df.head())
