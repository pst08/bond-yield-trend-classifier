Bond Yield Trend Classifier

A machine learning pipeline that predicts the next-month direction of India's 10-Year Government Bond yield using macroeconomic indicators like CPI and Repo Rate.

This project combines time-series features, financial reasoning, and ML modeling — complete with a Streamlit dashboard for interactive insights.

Project Structure :-

bond-yield-trend-classifier/
├── data/ # Raw & processed CSV files
├── src/
│ ├── data_loader.py # Downloads or loads yield + macro data
│ └── feature_builder.py
├── models/
│ └── train_model.py # Model training + evaluation
├── output/ # Plots (confusion matrix, feature importance)
├── dashboard/
│ └── app.py # Streamlit dashboard
└── README.md

How It Works :-

> Data:

10Y Indian bond yield (Investing.com / Yahoo)

Macroeconomic indicators (CPI, Repo Rate via TradingEconomics or CSV)

> Feature Engineering:

Lagged yields, delta, moving average, CPI, and repo rate

> Models:

Logistic Regression: 79% accuracy

Random Forest: 100% on small test set (subject to real-world validation)

> Output:

Visuals: Confusion matrix & feature importance

Dashboard: Predicts next-month yield direction and shows macro trends

Why This Matters?

Bond yields are influenced by inflation, interest rates, and economic expectations.
This project simulates how institutions like LSEG or RBI might monitor macro-driven yield movements using machine learning.

Sample Prediction :-

"Given the latest CPI, Repo Rate, and trend data, the yield is predicted to increase next month."

Run the Dashboard :-

# From root folder

streamlit run dashboard/app.py

Setup (if you want to include this) :-

conda create -n bondai python=3.11
conda activate bondai
pip install -r requirements.txt

Future Enhancements :-

Add CSV upload for user predictions
Incorporate LSTM for forecasting
Deploy to Streamlit Cloud
Improve class balance with SMOTE or weighted loss

Author :-
Preethy Sunny Thomas
CS Undergrad | Machine Learning Enthusiast
github.com/pst08
