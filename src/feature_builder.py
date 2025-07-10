import pandas as pd
import os

# --- Load datasets
bond_df = pd.read_csv("data/india_10y_yield.csv")
macro_df = pd.read_csv("data/india_macro.csv")

# --- Merge on 'Date'
df = pd.merge(bond_df, macro_df, on='Date', how='inner')

# --- Sort by Date
df = df.sort_values("Date")

# --- Feature Engineering

# Lagged Yield (previous month)
df["Yield_Lag1"] = df["Yield"].shift(1)

# Monthly Change
df["Delta_Yield"] = df["Yield"] - df["Yield_Lag1"]

# 3-month Moving Average
df["MA_Yield_3"] = df["Yield"].rolling(window=3).mean()

# Direction Label (1 = yield goes up next month, 0 = stays or drops)
df["Yield_Next"] = df["Yield"].shift(-1)
df["Direction"] = (df["Yield_Next"] > df["Yield"]).astype(int)

# Drop rows with any missing values
df = df.dropna()

# --- Save final dataset
os.makedirs("data", exist_ok=True)
df.to_csv("data/model_data.csv", index=False)
print("[✔] Saved processed dataset to data/model_data.csv")
print(df.tail())
