
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# Project folders
Path("results").mkdir(exist_ok=True)

# Load the downloaded AI4I dataset
df = pd.read_csv("ai4i2020.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst five rows:")
print(df.head())

# Remove columns that are identifiers or can leak the target
df = df.drop(
    columns=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"],
    errors="ignore"
)

# Target: machine failure
target = "Machine failure"

if target not in df.columns:
    raise ValueError(
        "Target column not found. Check the dataset column names."
    )

# Convert categorical Type column to numeric one-hot columns
X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nRandom Forest Results")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

# Save confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=[0, 1],
    display_labels=["Normal", "Failure"]
)
plt.title("AI4I Dataset - Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

# Save feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

importance.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("results/feature_importance.png")
plt.close()

# Save model and feature columns
joblib.dump(
    {"model": model, "features": X.columns.tolist()},
    "results/failure_model.joblib"
)

pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": predictions
}).to_csv("results/predictions.csv", index=False)

print("\nSaved model and evaluation files in the results folder.")
print("Dataset: AI4I 2020 Predictive Maintenance (synthetic data).")

# Graph for actual failure cases
failed_cases = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": predictions
})

actual_failures = failed_cases[failed_cases["Actual"] == 1]

correct_detected = (
    (actual_failures["Predicted"] == 1).sum()
)
missed_failures = (
    (actual_failures["Predicted"] == 0).sum()
)

plt.figure(figsize=(6, 4))
plt.bar(
    ["Correctly Detected", "Missed Failures"],
    [correct_detected, missed_failures]
)
plt.title("Actual Failure Cases Analysis")
plt.ylabel("Number of Cases")
plt.tight_layout()
plt.savefig("results/failed_cases_graph.png")
plt.close()

print("Correctly detected failures:", correct_detected)
print("Missed failures:", missed_failures)
print("Saved model and evaluation files in the results folder.")
