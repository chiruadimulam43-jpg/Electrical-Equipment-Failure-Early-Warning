
import os
import time
import warnings

import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

warnings.filterwarnings("ignore")

# 1. Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "ai4i2020.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Electrical Equipment Failure Early Warning")
print("-" * 45)

# 2. Load dataset
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        "Place ai4i2020.csv in the same folder as train_model.py"
    )

df = pd.read_csv(DATA_FILE)
print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)

# 3. Prepare target
target = "Machine failure"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found.")

y = pd.to_numeric(df[target], errors="coerce")

# Exclude target and failure-mode labels to prevent leakage
exclude = [
    "UDI", "Product ID", "Machine failure",
    "TWF", "HDF", "PWF", "OSF", "RNF"
]

X = df.drop(columns=exclude, errors="ignore").copy()

# 4. Convert features into numeric values
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.replace([float("inf"), float("-inf")], float("nan"))
X = X.fillna(X.median()).fillna(0)

# Remove rows with invalid target labels
valid = y.isin([0, 1])
X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].astype(int).reset_index(drop=True)

print("Cleaned dataset shape:", X.shape)
print("Features:", list(X.columns))
print("Failure cases:", int((y == 1).sum()))
print("Normal cases:", int((y == 0).sum()))
print("Missing feature values:", int(X.isna().sum().sum()))

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 6. Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

start = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start

print("\nModel training completed!")
print("Training time:", round(training_time, 2), "seconds")

# 7. Predictions and evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)

print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("ROC-AUC:", round(roc_auc, 4))
print("Average Precision:", round(avg_precision, 4))

report_text = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Failure", "Failure"],
    zero_division=0
)

print("\nClassification Report:\n")
print(report_text)

report_dict = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Failure", "Failure"],
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report_dict).transpose().to_csv(
    os.path.join(OUTPUT_DIR, "classification_report.csv")
)

with open(
    os.path.join(OUTPUT_DIR, "classification_report.txt"),
    "w",
    encoding="utf-8"
) as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
    f.write(report_text)

# 8. Save predictions
pd.DataFrame({
    "Actual_Failure": y_test.to_numpy(),
    "Predicted_Failure": y_pred,
    "Failure_Probability": y_prob
}).to_csv(
    os.path.join(OUTPUT_DIR, "test_predictions.csv"),
    index=False
)

# 9. Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Failure", "Failure"],
    yticklabels=["No Failure", "Failure"]
)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# 10. Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

plt.figure(figsize=(9, 6))
importance.plot(kind="barh")
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "feature_importance.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

importance.sort_values(ascending=False).to_csv(
    os.path.join(OUTPUT_DIR, "feature_importance.csv"),
    header=["Importance"]
)

# 11. ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "--", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "roc_curve.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# 12. Precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve (AP = {avg_precision:.3f})")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "precision_recall_curve.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# 13. Save metrics
pd.DataFrame([{
    "Model": "Random Forest",
    "Accuracy_Percent": round(accuracy * 100, 2),
    "ROC_AUC": round(roc_auc, 4),
    "Average_Precision": round(avg_precision, 4),
    "Training_Time_Seconds": round(training_time, 2),
    "Training_Samples": len(X_train),
    "Testing_Samples": len(X_test)
}]).to_csv(
    os.path.join(OUTPUT_DIR, "model_metrics.csv"),
    index=False
)

# 14. Save model
joblib.dump(
    {
        "model": model,
        "features": list(X.columns),
        "threshold": 0.5
    },
    os.path.join(OUTPUT_DIR, "electrical_failure_model.pkl")
)

print("\nAll files saved successfully!")
print("Results folder:", OUTPUT_DIR)

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)

print("\nProcess completed successfully!")
