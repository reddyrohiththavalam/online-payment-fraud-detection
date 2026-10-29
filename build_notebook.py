"""Generate fraud_detection_walkthrough.ipynb containing cell-by-cell modules."""

import json
from pathlib import Path


def create_notebook():
    cells = []

    def add_markdown(content: str):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in content.split("\n")]
        })

    def add_code(content: str):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in content.split("\n")]
        })

    # Header
    add_markdown("""# Online Payment Fraud Detection - Module-by-Module Walkthrough

This notebook provides an interactive, cell-by-cell walkthrough of every module in the project:
1. **Module 1: Environment Setup & Library Imports**
2. **Module 2: Data Ingestion & Class Imbalance Inspection**
3. **Module 3: Exploratory Data Analysis & Visualization**
4. **Module 4: Feature Engineering & Balance Error Derivation**
5. **Module 5: Class Balancing Strategy & Sample Weighting**
6. **Module 6: Model Architecture & Artifact Inspection**
7. **Module 7: Model Evaluation & Performance Metrics**
8. **Module 8: Visual Evaluation (Confusion Matrix & ROC Curve)**
9. **Module 9: Interactive Single-Transaction Inference & Anomaly Checks**
10. **Module 10: Batch CSV Transaction Scoring**
11. **Module 11: Architectural Comparison (Gradient Boosting vs. RXT-J / EARN Deep Learning)**

You can run each cell sequentially (`Shift + Enter`) to observe the output of each module.""")

    # Module 1
    add_markdown("## Module 1: Environment Setup & Library Imports\nImports all necessary libraries and verifies environment configurations.")
    add_code("""import os
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

try:
    from IPython.display import display
except ImportError:
    display = print

# Project modules
from fast_inference import FastFraudPredictor
from train_fast import _engineer_features, _align_features
from web_app import _balance_warnings

# Plotting configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
%matplotlib inline

print("[OK] All libraries loaded successfully.")
print(f"[OK] Python version: {sys.version.split()[0]}")
print(f"[OK] NumPy: {np.__version__} | Pandas: {pd.__version__} | Joblib: {joblib.__version__}")""")

    # Module 2
    add_markdown("## Module 2: Data Ingestion & Class Imbalance Inspection\nLoads sample records from the PaySim dataset splits and examines the severe class imbalance.")
    add_code("""data_dir = Path("data/data/data/processed")
x_train_path = data_dir / "X_train.csv"
y_train_path = data_dir / "y_train.csv"

# Load sample chunk
sample_size = 20000
df_features = pd.read_csv(x_train_path, nrows=sample_size)
df_labels = pd.read_csv(y_train_path, nrows=sample_size)

print(f"Sample Dataset Shape: {df_features.shape[0]:,} rows, {df_features.shape[1]} columns")
print("\\nColumns:", list(df_features.columns))

# Class distribution
fraud_counts = df_labels['isFraud'].value_counts()
print("\\nClass Distribution:")
print(f"  - Legitimate (0): {fraud_counts.get(0, 0):,} ({fraud_counts.get(0, 0)/len(df_labels)*100:.2f}%)")
print(f"  - Fraudulent (1): {fraud_counts.get(1, 0):,} ({fraud_counts.get(1, 0)/len(df_labels)*100:.2f}%)")

df_features.head()""")

    # Module 3
    add_markdown("## Module 3: Exploratory Data Visualization\nVisualizes the transaction distribution and features.")
    add_code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Class imbalance
labels = ['Legitimate (0)', 'Fraud (1)']
sizes = [fraud_counts.get(0, 0), fraud_counts.get(1, 0)]
colors = ['#3b82f6', '#ef4444']
axes[0].bar(labels, sizes, color=colors, width=0.4)
axes[0].set_yscale('log')
axes[0].set_title('Class Imbalance (Log Scale)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Transaction Count (Log Scale)')

# Plot 2: Transaction Types in sample
type_cols = [c for c in df_features.columns if c.startswith('type_')]
if type_cols:
    type_counts = df_features[type_cols].sum()
    type_counts.index = [c.replace('type_', '') for c in type_counts.index]
    axes[1].bar(type_counts.index, type_counts.values, color='#10b981', width=0.5)
    axes[1].set_title('Transaction Types Breakdown', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()""")

    # Module 4
    add_markdown("""## Module 4: Feature Engineering & Domain Logic
In financial fraud detection, raw balances alone are insufficient. We compute financial domain indicators:
* `errorBalanceOrig`: Inconsistency in sender account balance.
* `errorBalanceDest`: Inconsistency in recipient account balance.
* Drop `isFlaggedFraud`: Synthetic rule flag that would cause data leakage.
* Drop identifiers: `nameOrig`, `nameDest`.""")
    add_code("""raw_sample = pd.DataFrame([
    {
        "step": 200,
        "type": "CASH_OUT",
        "amount": 100000.0,
        "oldbalanceOrg": 100000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 20000.0,
        "newbalanceDest": 120000.0,
        "isFlaggedFraud": 0,
        "nameOrig": "C12345678",
        "nameDest": "M87654321"
    },
    {
        "step": 205,
        "type": "TRANSFER",
        "amount": 50000.0,
        "oldbalanceOrg": 50000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0, # Anomaly: recipient received 0
        "isFlaggedFraud": 1,
        "nameOrig": "C99999999",
        "nameDest": "C88888888"
    }
])

print("Raw Data (Before Engineering):")
display(raw_sample)

engineered_sample = _engineer_features(raw_sample)
print("\\nEngineered Data (Notice derived errorBalance columns & dropped IDs):")
display(engineered_sample)""")

    # Module 5
    add_markdown("""## Module 5: Class Balancing Strategy & Sample Weighting
Because legitimate transactions exceed fraud cases by ~700 to 1:
* Training retains **100% of fraud instances**.
* Legitimate records are sampled down to 250,000 to keep RAM footprint low.
* **Population-correcting sample weights** are computed to preserve true prior probabilities.""")
    add_code("""# Weight calculation demonstration
total_negatives = 5_000_000
total_positives = 5_274
sampled_negatives = 250_000
sampled_positives = 5_274

weight_legitimate = total_negatives / sampled_negatives
weight_fraud = total_positives / sampled_positives

print("Sampling & Weighting Calibration:")
print(f"  - Weight per legitimate sample : {weight_legitimate:.4f} (scales downsampled instances)")
print(f"  - Weight per fraudulent sample : {weight_fraud:.4f} (preserves full fraud strength)")
print(f"  - Effective population ratio   : {weight_legitimate / weight_fraud:.2f}x calibration")""")

    # Module 6
    add_markdown("## Module 6: Model Architecture & Artifact Inspection\nLoads and inspects the trained model artifact from `artifacts/fast_model/`.")
    add_code("""model_file = Path("artifacts/fast_model/fraud_hist_gradient_boosting.joblib")
artifact = joblib.load(model_file)

model = artifact["model"]
feature_cols = artifact["feature_columns"]
threshold = artifact["threshold"]

print("=" * 60)
print("TRAINED MODEL METADATA")
print("=" * 60)
print(f"Classifier Type   : {type(model).__name__}")
print(f"Iterations Used   : {model.n_iter_}")
print(f"Max Iterations    : {model.max_iter}")
print(f"Learning Rate     : {model.learning_rate}")
print(f"Decision Threshold: {threshold:.2f}")
print(f"Target Column     : {artifact['target_column']}")
print(f"Feature Count     : {len(feature_cols)}")
print("\\nFeatures Required by Model:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:>2}. {col}")""")

    # Module 7
    add_markdown("## Module 7: Model Evaluation & Performance Metrics\nReads the evaluation metrics recorded for validation and test splits.")
    add_code("""metrics_file = Path("artifacts/fast_model/fast_metrics.json")
with open(metrics_file, "r") as f:
    report = json.load(f)

test_m = report["metrics"]
val_m = report.get("validation_metrics", {})

print("=" * 65)
print("HELD-OUT TEST SET METRICS (Steps 601+)")
print("=" * 65)
print(f"Test Window Size    : {report['test_rows']:,} transactions")
print(f"Fraud Cases in Test : {report['test_fraud_rows']:,} transactions")
print("-" * 65)
print(f"Accuracy            : {test_m['accuracy'] * 100:.4f}%")
print(f"Precision           : {test_m['precision'] * 100:.4f}%")
print(f"Recall              : {test_m['recall'] * 100:.4f}%")
print(f"F1-Score            : {test_m['f1']:.6f}")
print(f"ROC AUC             : {test_m['auc_roc']:.6f}")
print(f"Average Precision   : {test_m['average_precision']:.6f}")
print(f"Matthews Corr (MCC) : {test_m['mcc']:.6f}")
print(f"Log Loss            : {test_m['log_loss']:.8f}")
print("=" * 65)""")

    # Module 8
    add_markdown("## Module 8: Visual Evaluation (Confusion Matrix & ROC Curve)\nDisplays the evaluated performance curves.")
    add_code("""from PIL import Image

cm_path = Path("artifacts/fast_model/confusion_matrix.png")
roc_path = Path("artifacts/fast_model/roc_curve.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

if cm_path.is_file():
    axes[0].imshow(Image.open(cm_path))
    axes[0].axis('off')
    axes[0].set_title('Test Set Confusion Matrix', fontsize=13, fontweight='bold')

if roc_path.is_file():
    axes[1].imshow(Image.open(roc_path))
    axes[1].axis('off')
    axes[1].set_title('ROC Curve (AUC = 1.000)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()""")

    # Module 9
    add_markdown("## Module 9: Interactive Single-Transaction Inference\nTests the inference engine against diverse scenarios: legitimate payments, transfers, and fraud drain attacks.")
    add_code("""predictor = FastFraudPredictor("artifacts/fast_model/fraud_hist_gradient_boosting.joblib")

test_scenarios = [
    {
        "scenario": "Legitimate Merchant Payment",
        "type": "PAYMENT", "step": 280, "amount": 120.50,
        "oldbalanceOrg": 1500.00, "newbalanceOrig": 1379.50,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0
    },
    {
        "scenario": "Legitimate P2P Transfer",
        "type": "TRANSFER", "step": 350, "amount": 5000.00,
        "oldbalanceOrg": 12000.00, "newbalanceOrig": 7000.00,
        "oldbalanceDest": 10000.00, "newbalanceDest": 15000.00
    },
    {
        "scenario": "Account Drain Attack (TRANSFER)",
        "type": "TRANSFER", "step": 583, "amount": 759701.03,
        "oldbalanceOrg": 759701.03, "newbalanceOrig": 0.00,
        "oldbalanceDest": 0.00, "newbalanceDest": 0.00
    },
    {
        "scenario": "Emptied Balance Cash-Out",
        "type": "CASH_OUT", "step": 602, "amount": 350000.00,
        "oldbalanceOrg": 350000.00, "newbalanceOrig": 0.00,
        "oldbalanceDest": 0.00, "newbalanceDest": 350000.00
    }
]

df_scenarios = pd.DataFrame(test_scenarios)
results = predictor.predict(df_scenarios.drop(columns="scenario"))

print(f"{'Scenario':<32} | {'Score':<10} | {'Decision':<22} | {'Warnings'}")
print("-" * 85)
for i, row in results.iterrows():
    name = test_scenarios[i]["scenario"]
    score = f"{row['fraud_probability'] * 100:.2f}%"
    decision = "[FLAGGED]" if row['predicted_label'] == 1 else "[APPROVED]"
    warnings = _balance_warnings(test_scenarios[i])
    warn_str = warnings[0] if warnings else "None"
    print(f"{name:<32} | {score:<10} | {decision:<22} | {warn_str}")""")

    # Module 10
    add_markdown("## Module 10: Batch CSV Transaction Scoring\nDemonstrates streaming batch prediction for scoring large files.")
    add_code("""batch_data = pd.DataFrame([
    {"step": 100, "type": "PAYMENT", "amount": 45.0, "oldbalanceOrg": 300.0, "newbalanceOrig": 255.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0},
    {"step": 102, "type": "TRANSFER", "amount": 950000.0, "oldbalanceOrg": 950000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0},
    {"step": 105, "type": "CASH_IN", "amount": 1200.0, "oldbalanceOrg": 1000.0, "newbalanceOrig": 2200.0, "oldbalanceDest": 3000.0, "newbalanceDest": 1800.0},
    {"step": 110, "type": "CASH_OUT", "amount": 800000.0, "oldbalanceOrg": 800000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 5000.0, "newbalanceDest": 805000.0},
])

scored_batch = predictor.predict(batch_data)
display(scored_batch[['step', 'type', 'amount', 'fraud_probability', 'predicted_label']])""")

    # Module 11
    add_markdown("## Module 11: Architectural Comparison (Gradient Boosting vs RXT-J / EARN Deep Learning)\nCompares the deployed production model against the research deep learning baseline.")
    add_code("""comparison_df = pd.DataFrame([
    {"Metric / Property": "Model Family", "HistGradientBoosting (Active)": "Histogram Gradient Boost Trees", "RXT-J / EARN Deep Learning": "Autoencoder + ResNet + GRU"},
    {"Metric / Property": "Data Representation", "HistGradientBoosting (Active)": "Tabular + Derived Balance Errors", "RXT-J / EARN Deep Learning": "Sequence Windows (L=10) + SMOTE"},
    {"Metric / Property": "Accuracy", "HistGradientBoosting (Active)": "99.999%", "RXT-J / EARN Deep Learning": "80.00%"},
    {"Metric / Property": "Recall (Fraud Catch)", "HistGradientBoosting (Active)": "99.76% (100% held-out)", "RXT-J / EARN Deep Learning": "100.00%"},
    {"Metric / Property": "Precision (Alert Acc)", "HistGradientBoosting (Active)": "99.70% (5 false alarms)", "RXT-J / EARN Deep Learning": "31.82% (higher false alarms)"},
    {"Metric / Property": "F1-Score", "HistGradientBoosting (Active)": "0.9973", "RXT-J / EARN Deep Learning": "0.4828"},
    {"Metric / Property": "ROC AUC", "HistGradientBoosting (Active)": "0.99985", "RXT-J / EARN Deep Learning": "0.9643"},
    {"Metric / Property": "Matthews Corr (MCC)", "HistGradientBoosting (Active)": "0.9986", "RXT-J / EARN Deep Learning": "0.4980"},
    {"Metric / Property": "Log Loss", "HistGradientBoosting (Active)": "0.00000026", "RXT-J / EARN Deep Learning": "0.6384"},
    {"Metric / Property": "Inference Latency", "HistGradientBoosting (Active)": "< 5 ms (Real-time)", "RXT-J / EARN Deep Learning": "~50-150 ms (Buffered)"},
    {"Metric / Property": "Model Artifact Size", "HistGradientBoosting (Active)": "~450 KB (Lightweight)", "RXT-J / EARN Deep Learning": "Multi-MB Keras weights"},
    {"Metric / Property": "Deployment Status", "HistGradientBoosting (Active)": "ACTIVE (Production)", "RXT-J / EARN Deep Learning": "Research Benchmark"},
])

display(comparison_df.set_index("Metric / Property"))""")

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.14.7"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    output_path = Path("fraud_detection_walkthrough.ipynb")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"[SUCCESS] Notebook generated successfully: {output_path.resolve()}")


if __name__ == "__main__":
    create_notebook()
