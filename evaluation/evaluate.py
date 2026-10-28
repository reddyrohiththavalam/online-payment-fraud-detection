"""Binary classification metrics and fraud-detection plots."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class Evaluator:
    """Compute requested binary metrics and save ROC/confusion-matrix plots."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        probabilities: np.ndarray,
        output_dir: str | Path,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        raw_labels = np.asarray(y_true).reshape(-1)
        scores = np.asarray(probabilities).reshape(-1).astype(float)
        if len(raw_labels) != len(scores) or len(raw_labels) == 0:
            raise ValueError("y_true and probabilities must have the same non-zero length.")
        if not np.isfinite(raw_labels).all() or set(np.unique(raw_labels)) != {0, 1}:
            raise ValueError("Evaluation requires both binary labels 0 and 1.")
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Predicted probabilities must be finite values between 0 and 1.")
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be a finite value between 0 and 1.")

        labels = raw_labels.astype(int)
        predictions = (scores >= threshold).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "mcc": float(matthews_corrcoef(labels, predictions)),
            "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
            "auc_roc": float(roc_auc_score(labels, scores)),
        }
        Evaluator.save_plots(labels, scores, predictions, output_dir)
        return metrics

    @staticmethod
    def save_plots(
        y_true: np.ndarray,
        probabilities: np.ndarray,
        predictions: np.ndarray,
        output_dir: str | Path,
    ) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
        auc = roc_auc_score(y_true, probabilities)
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.plot(false_positive_rate, true_positive_rate, label=f"ROC AUC = {auc:.4f}")
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        axis.set(xlabel="False positive rate", ylabel="True positive rate", title="Fraud ROC curve")
        axis.legend(loc="lower right")
        figure.tight_layout()
        figure.savefig(path / "roc_curve.png", dpi=150)
        plt.close(figure)

        matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
        figure, axis = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign", "Fraud"],
            yticklabels=["Benign", "Fraud"],
            ax=axis,
        )
        axis.set(xlabel="Predicted label", ylabel="True label", title="Confusion matrix")
        figure.tight_layout()
        figure.savefig(path / "confusion_matrix.png", dpi=150)
        plt.close(figure)
