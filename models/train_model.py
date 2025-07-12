import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# --- Load the processed data
df = pd.read_csv("data/model_data.csv")

# --- Define features and target
features = ["Yield_Lag1", "Delta_Yield", "MA_Yield_3", "CPI", "RepoRate"]
X = df[features]
y = df["Direction"]

# --- Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# --- Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# --- Evaluate both
print("\nLogistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# --- Confusion Matrix (RF)
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
os.makedirs("output", exist_ok=True)
plt.savefig("output/confusion_matrix_rf.png")
plt.close()

# --- Feature Importance Plot (RF)
importances = pd.Series(rf.feature_importances_, index=features)
importances.sort_values().plot(kind='barh')
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("output/feature_importance_rf.png")
plt.close()

# --- Save models
joblib.dump(lr, "models/logistic_regression.pkl")
joblib.dump(rf, "models/random_forest.pkl")

print("\n[✔] Models trained and saved.")
print("[📊] Evaluation plots saved to output/")
