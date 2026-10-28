"""Display and compute evaluation metrics for the trained fraud detection model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from fast_inference import FastFraudPredictor


def display_saved_metrics(metrics_path: Path) -> None:
    """Read and format saved metrics from artifacts."""
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics file not found at: {metrics_path.resolve()}")

    with open(metrics_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    metrics = data.get("metrics", {})
    val_metrics = data.get("validation_metrics", {})

    print("=" * 65)
    print("      ONLINE PAYMENT FRAUD DETECTION - MODEL METRICS")
    print("=" * 65)
    print(f"Model Architecture  : {data.get('model', 'HistGradientBoostingClassifier')}")
    print(f"Artifact Location   : {data.get('model_file', 'artifacts/fast_model')}")
    print(f"Decision Threshold  : {metrics.get('threshold', 0.5):.2f}")
    print("-" * 65)
    print(f"Held-Out Test Window: Steps {data.get('test_step_range', [601, 'End'])[0]}+")
    print(f"Total Test Samples  : {data.get('test_rows', 0):,} transactions")
    print(f"Fraud Cases in Test : {data.get('test_fraud_rows', 0):,} transactions")
    print("-" * 65)
    print("HELD-OUT TEST SET PERFORMANCE:")
    print(f"  - Accuracy        : {metrics.get('accuracy', 0.0) * 100:.4f}%")
    print(f"  - Precision       : {metrics.get('precision', 0.0) * 100:.4f}%")
    print(f"  - Recall          : {metrics.get('recall', 0.0) * 100:.4f}%")
    print(f"  - F1-Score        : {metrics.get('f1', 0.0):.6f}")
    print(f"  - ROC AUC         : {metrics.get('auc_roc', 0.0):.6f}")
    print(f"  - Avg Precision   : {metrics.get('average_precision', 0.0):.6f}")
    print(f"  - Matthews Corr   : {metrics.get('mcc', 0.0):.6f}")
    print(f"  - Log Loss        : {metrics.get('log_loss', 0.0):.8f}")
    print("-" * 65)
    if val_metrics:
        print("VALIDATION SET PERFORMANCE (Steps 551-600):")
        print(f"  - Validation Acc  : {val_metrics.get('accuracy', 0.0) * 100:.4f}%")
        print(f"  - Validation F1   : {val_metrics.get('f1', 0.0):.6f}")
        print(f"  - Validation AUC  : {val_metrics.get('auc_roc', 0.0):.6f}")
        print("=" * 65)


def run_full_evaluation(
    model_path: Path,
    data_dir: Path,
    chunksize: int = 100_000,
) -> None:
    """Evaluate the model in streaming batches over all test samples."""
    print("=" * 65)
    print("      RUNNING FULL TEST EVALUATION (1.27M TRANSACTIONS)")
    print("=" * 65)
    predictor = FastFraudPredictor(model_path)
    x_test_path = data_dir / "X_test.csv"
    y_test_path = data_dir / "y_test.csv"

    if not x_test_path.is_file() or not y_test_path.is_file():
        raise FileNotFoundError(f"Test data files missing in {data_dir.resolve()}")

    y_trues: list[np.ndarray] = []
    y_preds: list[np.ndarray] = []
    y_probs: list[np.ndarray] = []

    print(f"Streaming data from {data_dir.name} in chunks of {chunksize:,} rows...")
    for i, (chunk_x, chunk_y) in enumerate(
        zip(
            pd.read_csv(x_test_path, chunksize=chunksize),
            pd.read_csv(y_test_path, chunksize=chunksize),
            strict=True,
        )
    ):
        pred = predictor.predict(chunk_x)
        y_trues.append(chunk_y.iloc[:, 0].to_numpy())
        y_preds.append(pred["predicted_label"].to_numpy())
        y_probs.append(pred["fraud_probability"].to_numpy())
        print(f"  Scored batch {i + 1} ({len(chunk_x):,} rows)...")

    y_true = np.concatenate(y_trues)
    y_pred = np.concatenate(y_preds)
    y_prob = np.concatenate(y_probs)

    matrix = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print("\nCONFUSION MATRIX:")
    print(f"  {'':15} Predicted Benign  Predicted Fraud")
    print(f"  Actual Benign : {matrix[0, 0]:>16,} {matrix[0, 1]:>16,}")
    print(f"  Actual Fraud  : {matrix[1, 0]:>16,} {matrix[1, 1]:>16,}")

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=["Benign (0)", "Fraud (1)"], digits=5))
    print(f"ROC AUC Score: {auc:.6f}")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-file",
        default="artifacts/fast_model/fast_metrics.json",
        help="Path to saved fast_metrics.json report.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run live evaluation across all 1.27M test transactions.",
    )
    parser.add_argument(
        "--model",
        default="artifacts/fast_model/fraud_hist_gradient_boosting.joblib",
        help="Path to trained model artifact for full evaluation.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/data/data/processed",
        help="Directory containing X_test.csv and y_test.csv.",
    )
    args = parser.parse_args()

    if args.full:
        run_full_evaluation(Path(args.model), Path(args.data_dir))
    else:
        display_saved_metrics(Path(args.metrics_file))


if __name__ == "__main__":
    main()
