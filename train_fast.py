"""Train and evaluate a memory-conscious tree model on prepared PaySim splits."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from evaluation.evaluate import Evaluator


LOGGER = logging.getLogger("fraud_detection.fast_train")
CHUNK_SIZE = 200_000
TARGET = "isFraud"
VALIDATION_TRAIN_MAX_STEP = 550
TRAIN_MAX_STEP = 600
VALIDATION_MIN_STEP = 551
VALIDATION_MAX_STEP = 600
TEST_MIN_STEP = 601


def _count_classes(path: Path, chunksize: int = CHUNK_SIZE) -> tuple[int, int]:
    negative_count = 0
    positive_count = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        labels = chunk.iloc[:, 0]
        negative_count += int((labels == 0).sum())
        positive_count += int((labels == 1).sum())
    if negative_count == 0 or positive_count == 0:
        raise ValueError(f"{path} must contain both fraud labels 0 and 1.")
    return negative_count, positive_count


def _engineer_features(features: pd.DataFrame) -> pd.DataFrame:
    drop_columns = [
        column
        for column in ("isFlaggedFraud", "isFraud", "nameOrig", "nameDest")
        if column in features.columns
    ]
    if drop_columns:
        features = features.drop(columns=drop_columns)
    if "type" in features.columns:
        type_features = pd.get_dummies(features.pop("type"), prefix="type", dtype=np.int8)
        features = pd.concat([features, type_features], axis=1)
    if {"newbalanceOrig", "amount", "oldbalanceOrg"}.issubset(features.columns):
        features["errorBalanceOrig"] = (
            features["newbalanceOrig"] + features["amount"] - features["oldbalanceOrg"]
        )
    if {"oldbalanceDest", "amount", "newbalanceDest"}.issubset(features.columns):
        features["errorBalanceDest"] = (
            features["oldbalanceDest"] + features["amount"] - features["newbalanceDest"]
        )
    return features


def _align_features(
    features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    engineered = _engineer_features(features.copy())
    expected_type_columns = [
        column for column in feature_columns if column.startswith("type_")
    ]
    if expected_type_columns and not any(
        column == "type" or column.startswith("type_")
        for column in features.columns
    ):
        raise ValueError(
            "Prediction data must include the transaction 'type' column or its "
            "one-hot encoded type features."
        )
    optional_features = {
        column
        for column in feature_columns
        if column.startswith("type_")
        or column in {"errorBalanceOrig", "errorBalanceDest"}
    }
    missing = [
        column
        for column in feature_columns
        if column not in engineered.columns and column not in optional_features
    ]
    if missing:
        raise ValueError(
            "Prediction data is missing required model features: " + ", ".join(missing)
        )
    aligned = engineered.reindex(columns=feature_columns, fill_value=0)
    try:
        aligned = aligned.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Prediction features must be numeric or known transaction types.") from error
    values = aligned.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Prediction features must not contain missing or non-finite values.")
    return aligned


def _sample_training_rows(
    feature_path: Path,
    label_path: Path,
    max_negative_samples: int,
    random_state: int,
    chunksize: int = CHUNK_SIZE,
    max_step: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, int, int, int, int]:
    if max_step is None:
        total_negatives, total_positives = _count_classes(label_path, chunksize)
    else:
        total_negatives = 0
        total_positives = 0
        for features, label_frame in zip(
            pd.read_csv(feature_path, chunksize=chunksize),
            pd.read_csv(label_path, chunksize=chunksize),
            strict=True,
        ):
            labels = label_frame.iloc[:, 0].to_numpy(dtype=np.int8)
            if len(features) != len(labels):
                raise ValueError("Feature and label chunk sizes do not match.")
            if "step" not in features:
                raise ValueError("Step-based training requires the PaySim 'step' feature.")
            selected = labels[features["step"].to_numpy() <= max_step]
            total_negatives += int((selected == 0).sum())
            total_positives += int((selected == 1).sum())
        if total_negatives == 0 or total_positives == 0:
            raise ValueError("The training time window must contain both fraud labels.")

    sample_limit = min(max_negative_samples, total_negatives)
    rng = np.random.default_rng(random_state)
    sampled_features: list[pd.DataFrame] = []
    sampled_labels: list[np.ndarray] = []
    negative_sample_count = 0
    positive_sample_count = 0

    feature_chunks: Iterator[pd.DataFrame] = pd.read_csv(
        feature_path,
        chunksize=chunksize,
    )
    label_chunks: Iterator[pd.DataFrame] = pd.read_csv(label_path, chunksize=chunksize)
    for features, label_frame in zip(feature_chunks, label_chunks, strict=True):
        labels = label_frame.iloc[:, 0].to_numpy(dtype=np.int8)
        if len(features) != len(labels):
            raise ValueError("Feature and label chunk sizes do not match.")
        if max_step is not None:
            if "step" not in features:
                raise ValueError("Step-based training requires the PaySim 'step' feature.")
            in_window = features["step"].to_numpy() <= max_step
        else:
            in_window = np.ones(len(labels), dtype=bool)

        positive_indices = np.flatnonzero(in_window & (labels == 1))
        negative_indices = np.flatnonzero(in_window & (labels == 0))
        remaining_negatives = sample_limit - negative_sample_count
        negative_quota = min(
            len(negative_indices),
            int(round(sample_limit * len(negative_indices) / total_negatives)),
            remaining_negatives,
        )
        if negative_quota:
            selected_negatives = rng.choice(
                negative_indices,
                size=negative_quota,
                replace=False,
            )
        else:
            selected_negatives = np.empty(0, dtype=np.int64)
        selected = np.concatenate((positive_indices, selected_negatives))
        if not len(selected):
            continue

        sampled_features.append(features.iloc[selected].copy())
        sampled_labels.append(labels[selected].copy())
        negative_sample_count += len(selected_negatives)
        positive_sample_count += len(positive_indices)

    if negative_sample_count == 0 or positive_sample_count == 0:
        raise ValueError("The sampled training data must contain both fraud labels.")
    training_features = _engineer_features(pd.concat(sampled_features, ignore_index=True))
    training_labels = np.concatenate(sampled_labels)
    if set(np.unique(training_labels)) != {0, 1}:
        raise ValueError("Training labels must be binary values 0 and 1.")
    return (
        training_features,
        training_labels,
        negative_sample_count,
        positive_sample_count,
        total_negatives,
        total_positives,
    )


def _evaluate_test_split(
    model: HistGradientBoostingClassifier,
    feature_path: Path,
    label_path: Path,
    chunksize: int = CHUNK_SIZE,
    output_dir: Path | None = None,
    min_step: int | None = None,
    max_step: int | None = None,
) -> tuple[dict[str, float], int, int]:
    actual_labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for features, label_frame in zip(
        pd.read_csv(feature_path, chunksize=chunksize),
        pd.read_csv(label_path, chunksize=chunksize),
        strict=True,
    ):
        labels = label_frame.iloc[:, 0].to_numpy(dtype=np.int8)
        if len(features) != len(labels):
            raise ValueError("Test feature and label chunk sizes do not match.")
        if min_step is not None or max_step is not None:
            if "step" not in features:
                raise ValueError("Step-based evaluation requires the PaySim 'step' feature.")
            in_window = np.ones(len(features), dtype=bool)
            if min_step is not None:
                in_window &= features["step"].to_numpy() >= min_step
            if max_step is not None:
                in_window &= features["step"].to_numpy() <= max_step
            selected = np.flatnonzero(in_window)
            features = features.iloc[selected]
            labels = labels[selected]
            if not len(selected):
                continue
        engineered = _align_features(features, model.feature_names_in_.tolist())
        actual_labels.append(labels)
        probabilities.append(model.predict_proba(engineered)[:, 1])

    if not actual_labels:
        raise ValueError("The requested evaluation time window contains no transactions.")
    y_true = np.concatenate(actual_labels)
    y_score = np.concatenate(probabilities)
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("The test split must contain both fraud labels 0 and 1.")
    predictions = (y_score >= 0.5).astype(np.int8)
    if output_dir is not None:
        Evaluator.save_plots(y_true, y_score, predictions, output_dir)
    metrics = {
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "log_loss": float(log_loss(y_true, y_score, labels=[0, 1])),
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }
    return metrics, len(y_true), int(y_true.sum())


def _fit_model(
    args: argparse.Namespace,
    data_dir: Path,
    max_step: int,
    training_phase: str,
) -> tuple[HistGradientBoostingClassifier, pd.DataFrame, dict[str, int]]:
    (
        train_features,
        train_labels,
        sampled_negatives,
        sampled_positives,
        training_negatives,
        training_positives,
    ) = _sample_training_rows(
        data_dir / "X_train.csv",
        data_dir / "y_train.csv",
        args.max_negative_samples,
        args.seed,
        max_step=max_step,
    )
    sample_weights = np.where(
        train_labels == 0,
        training_negatives / sampled_negatives,
        training_positives / sampled_positives,
    )
    model = HistGradientBoostingClassifier(
        learning_rate=args.learning_rate,
        max_iter=args.max_iterations,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=args.seed,
    )
    LOGGER.info(
        "Fitting %s model on %s sampled rows (%s fraud, %s legitimate) from steps 1-%s.",
        training_phase,
        len(train_labels),
        sampled_positives,
        sampled_negatives,
        max_step,
    )
    model.fit(train_features, train_labels, sample_weight=sample_weights)
    training_counts = {
        "window_rows": training_negatives + training_positives,
        "sample_rows": len(train_labels),
        "sample_fraud_rows": sampled_positives,
        "sample_legitimate_rows": sampled_negatives,
    }
    return model, train_features, training_counts


def train_and_evaluate(args: argparse.Namespace) -> dict[str, object]:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_model, validation_features, _ = _fit_model(
        args,
        data_dir,
        VALIDATION_TRAIN_MAX_STEP,
        "validation",
    )
    validation_metrics, validation_rows, validation_frauds = _evaluate_test_split(
        validation_model,
        data_dir / "X_train.csv",
        data_dir / "y_train.csv",
        min_step=VALIDATION_MIN_STEP,
        max_step=VALIDATION_MAX_STEP,
    )
    del validation_model, validation_features

    model, train_features, training_counts = _fit_model(
        args,
        data_dir,
        TRAIN_MAX_STEP,
        "final",
    )
    negative_count, positive_count = _count_classes(data_dir / "y_train.csv")
    metrics, test_rows, test_frauds = _evaluate_test_split(
        model,
        data_dir / "X_test.csv",
        data_dir / "y_test.csv",
        output_dir=output_dir,
        min_step=TEST_MIN_STEP,
    )
    model_path = output_dir / "fraud_hist_gradient_boosting.joblib"
    feature_columns = train_features.columns.tolist()
    derived_features = {"errorBalanceOrig", "errorBalanceDest"}
    form_defaults = {
        column: float(train_features[column].median())
        for column in feature_columns
        if not column.startswith("type_") and column not in derived_features
    }
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "form_defaults": form_defaults,
            "target_column": TARGET,
            "threshold": metrics["threshold"],
            "test_metrics": metrics,
            "test_rows": test_rows,
            "test_fraud_rows": test_frauds,
        },
        model_path,
    )
    report: dict[str, object] = {
        "model": "HistGradientBoostingClassifier",
        "data_directory": str(data_dir),
        "training_split_rows": negative_count + positive_count,
        "training_window_rows": training_counts["window_rows"],
        "training_sample_rows": training_counts["sample_rows"],
        "training_sample_fraud_rows": training_counts["sample_fraud_rows"],
        "training_sample_legitimate_rows": training_counts["sample_legitimate_rows"],
        "training_sample_max_legitimate_rows": args.max_negative_samples,
        "validation_training_step_range": [1, VALIDATION_TRAIN_MAX_STEP],
        "training_step_range": [1, TRAIN_MAX_STEP],
        "validation_step_range": [VALIDATION_MIN_STEP, VALIDATION_MAX_STEP],
        "validation_rows": validation_rows,
        "validation_fraud_rows": validation_frauds,
        "validation_metrics": validation_metrics,
        "test_step_range": [TEST_MIN_STEP, None],
        "test_rows": test_rows,
        "test_fraud_rows": test_frauds,
        "iterations_used": int(model.n_iter_),
        "max_iterations": args.max_iterations,
        "max_leaf_nodes": args.max_leaf_nodes,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "model_file": str(model_path),
    }
    (output_dir / "fast_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    LOGGER.info(
        "Time-separated test metrics (steps >= %s): %s",
        TEST_MIN_STEP,
        json.dumps(metrics, sort_keys=True),
    )
    LOGGER.info("Model and report saved under %s", output_dir.resolve())
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/data/data/processed",
        help="Directory containing X_train.csv, y_train.csv, X_test.csv, and y_test.csv.",
    )
    parser.add_argument("--output-dir", default="artifacts/fast_model")
    parser.add_argument("--max-negative-samples", type=int, default=250_000)
    parser.add_argument("--max-iterations", type=int, default=150)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.max_negative_samples < 1:
        parser.error("--max-negative-samples must be positive.")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be positive.")
    if args.max_leaf_nodes < 2:
        parser.error("--max-leaf-nodes must be at least 2.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    train_and_evaluate(parse_args())


if __name__ == "__main__":
    main()
