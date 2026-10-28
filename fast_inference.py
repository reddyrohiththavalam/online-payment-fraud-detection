"""Inference helper for the memory-conscious tree-based fraud model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from train_fast import _align_features


class FastFraudPredictor:
    """Load a trusted fast-model artifact and score raw or prepared transactions."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Fast model file not found: {self.model_path}")
        artifact: dict[str, Any] = joblib.load(self.model_path)
        try:
            self.model = artifact["model"]
            self.feature_columns = list(artifact["feature_columns"])
            self.form_defaults = {
                str(name): float(value)
                for name, value in artifact.get("form_defaults", {}).items()
            }
            self.test_metrics = dict(artifact.get("test_metrics", {}))
            self.test_rows = int(artifact.get("test_rows", 0))
            self.test_fraud_rows = int(artifact.get("test_fraud_rows", 0))
            self.target_column = str(artifact["target_column"])
            self.threshold = float(artifact["threshold"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid fast-model artifact: {self.model_path}") from error
        if not self.feature_columns:
            raise ValueError("Fast-model artifact has no feature schema.")
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("Fast-model artifact has an invalid decision threshold.")

    @property
    def form_fields(self) -> list[dict[str, Any]]:
        derived = set()
        if {"oldbalanceOrg", "newbalanceOrig", "amount"}.issubset(self.feature_columns):
            derived.add("errorBalanceOrig")
        if {"oldbalanceDest", "newbalanceDest", "amount"}.issubset(self.feature_columns):
            derived.add("errorBalanceDest")

        fields: list[dict[str, Any]] = []
        if any(column.startswith("type_") for column in self.feature_columns):
            categories = [
                column.removeprefix("type_")
                for column in self.feature_columns
                if column.startswith("type_")
            ]
            fields.append(
                {
                    "name": "type",
                    "label": "Transaction type",
                    "kind": "select",
                    "options": categories,
                    "help": "Choose the PaySim transaction category.",
                }
            )

        field_details = {
            "step": {
                "label": "Time step",
                "help": "PaySim hour index (1 to 743).",
                "minimum": 1,
                "maximum": 743,
                "placeholder": "For example, 300",
                "input_step": 1,
            },
            "amount": {
                "label": "Transaction amount",
                "help": "Amount transferred or paid; enter a non-negative value.",
                "minimum": 0,
                "placeholder": "For example, 1250.00",
                "input_step": "any",
            },
            "oldbalanceOrg": {
                "label": "Sender balance before",
                "help": "Origin account balance before the transaction.",
                "minimum": 0,
                "placeholder": "Enter the starting sender balance",
                "input_step": "any",
            },
            "newbalanceOrig": {
                "label": "Sender balance after",
                "help": "Origin account balance after the transaction.",
                "minimum": 0,
                "placeholder": "Enter the ending sender balance",
                "input_step": "any",
            },
            "oldbalanceDest": {
                "label": "Recipient balance before",
                "help": "Destination account balance before the transaction.",
                "minimum": 0,
                "placeholder": "Enter the starting recipient balance",
                "input_step": "any",
            },
            "newbalanceDest": {
                "label": "Recipient balance after",
                "help": "Destination account balance after the transaction.",
                "minimum": 0,
                "placeholder": "Enter the ending recipient balance",
                "input_step": "any",
            },
        }
        for column in self.feature_columns:
            if (
                column in derived
                or column.startswith("type_")
                or column == "type"
            ):
                continue
            details = field_details.get(
                column,
                {
                    "label": column.replace("_", " ").capitalize(),
                    "help": "Enter the value recorded for this transaction.",
                    "minimum": 0,
                    "placeholder": "Enter a value",
                    "input_step": "any",
                },
            )
            typical_value = self.form_defaults.get(column)
            if typical_value is not None:
                details = {
                    **details,
                    "placeholder": f"{details['placeholder']} (typical: {typical_value:.6g})",
                }
            fields.append(
                {
                    "name": column,
                    **details,
                    "kind": "number",
                    "default": "",
                }
            )
        return fields

    def predict(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Return input transactions with fraud probability and thresholded label."""
        if transactions.empty:
            raise ValueError("No transactions were supplied for prediction.")
        features = _align_features(transactions, self.feature_columns)
        probabilities = self.model.predict_proba(features)[:, 1]
        if (
            len(probabilities) != len(transactions)
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
        ):
            raise RuntimeError("Classifier returned invalid fraud probabilities.")
        result = transactions.copy()
        result = result.drop(columns=[self.target_column], errors="ignore")
        result["fraud_probability"] = probabilities
        result["predicted_label"] = (probabilities >= self.threshold).astype(np.int8)
        return result
