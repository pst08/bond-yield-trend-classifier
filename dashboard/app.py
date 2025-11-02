import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bond Yield Trend Classifier", layout="centered")

st.title("Bond Yield Trend Classifier")
st.markdown("Predict whether the 10Y bond yield will rise next month.")

#Load data
df = pd.read_csv("data/model_data.csv")
model = joblib.load("models/random_forest.pkl")

st.subheader("Latest Macroeconomic Data")
st.dataframe(df.tail(5)[["Date", "Yield", "CPI", "RepoRate"]])

st.subheader("Yield Over Time")
fig, ax = plt.subplots()
df.plot(x="Date", y="Yield", ax=ax, figsize=(10, 4))
st.pyplot(fig)

st.subheader("Feature Importance")
st.image("output/feature_importance_rf.png", caption="Random Forest Feature Importance", use_column_width=True)

st.subheader("Predict Next Month's Yield Direction")

latest = df.iloc[-1]
features = ["Yield_Lag1", "Delta_Yield", "MA_Yield_3", "CPI", "RepoRate"]
latest_input = latest[features].values.reshape(1, -1)
pred = model.predict(latest_input)[0]

if pred == 1:
    st.success("Prediction: Yield is expected to **increase** next month.")
else:
    st.warning("Prediction: Yield is expected to **stay the same or decrease**.")
