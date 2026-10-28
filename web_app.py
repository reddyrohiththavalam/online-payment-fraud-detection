"""Local browser interface for interactive transaction fraud scoring."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from fast_inference import FastFraudPredictor


LOGGER = logging.getLogger("fraud_detection.web")


def _balance_warnings(row: dict[str, Any]) -> list[str]:
    transaction_type = row.get("type")
    if transaction_type not in {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}:
        return []

    amount = float(row["amount"])
    old_origin = float(row["oldbalanceOrg"])
    new_origin = float(row["newbalanceOrig"])
    expected_origin = (
        old_origin + amount
        if transaction_type == "CASH_IN"
        else max(old_origin - amount, 0)
    )
    warnings = []
    if not np.isclose(new_origin, expected_origin, rtol=0, atol=0.01):
        warnings.append(
            f"For {transaction_type}, the sender's after-balance usually is "
            f"{expected_origin:,.2f} from these inputs; please verify the balances."
        )

    if transaction_type == "CASH_IN":
        old_destination = float(row["oldbalanceDest"])
        new_destination = float(row["newbalanceDest"])
        expected_destination = max(old_destination - amount, 0)
        if not np.isclose(new_destination, expected_destination, rtol=0, atol=0.01):
            warnings.append(
                "For CASH_IN, the recipient-side balance usually decreases by "
                f"the amount (to {expected_destination:,.2f} here); please verify "
                "the balances."
            )
    return warnings


def _parse_input(
    raw_data: dict[str, Any],
    fields: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate and extract transaction inputs according to model field definitions."""
    row: dict[str, Any] = {}
    display_values: dict[str, str] = {}
    for field in fields:
        name = field["name"]
        raw_val = raw_data.get(name)
        raw_value = str(raw_val).strip() if raw_val is not None else ""
        if not raw_value:
            raise ValueError(f"{field['label']} is required.")
        if field["kind"] == "number":
            try:
                value = float(raw_value)
            except ValueError as error:
                raise ValueError(f"{field['label']} must be a number.") from error
            if not np.isfinite(value):
                raise ValueError(f"{field['label']} must be finite.")
            minimum = field.get("minimum")
            maximum = field.get("maximum")
            if minimum is not None and value < minimum:
                raise ValueError(
                    f"{field['label']} must be at least {minimum}."
                )
            if maximum is not None and value > maximum:
                raise ValueError(
                    f"{field['label']} must be no more than {maximum}."
                )
            if field.get("input_step") == 1 and not value.is_integer():
                raise ValueError(f"{field['label']} must be a whole number.")
        else:
            if raw_value not in field.get("options", []):
                raise ValueError(f"Select a valid {field['label']} value.")
            value = raw_value
        row[name] = value
        display_values[name] = raw_value
    return row, display_values


def _score_row(row: dict[str, Any], predictor: Any) -> dict[str, Any]:
    """Score a parsed transaction row and format probabilities and threshold results."""
    prediction = predictor.predict(pd.DataFrame([row])).iloc[0]
    probability = float(prediction["fraud_probability"])
    label = int(prediction["predicted_label"])
    percent = probability * 100
    if percent == 0:
        probability_display = "0%"
    elif percent < 1e-10:
        probability_display = f"{percent:.6g}%"
    else:
        decimal_places = max(0, 5 - math.floor(math.log10(percent)))
        formatted_percent = f"{percent:.{decimal_places}f}"
        probability_display = (
            f"{formatted_percent.rstrip('0').rstrip('.')}%"
            if "." in formatted_percent
            else f"{formatted_percent}%"
        )
    return {
        "probability": probability,
        "percent_display": probability_display,
        "label": label,
        "balance_warnings": _balance_warnings(row),
        "decision": (
            "Flagged for review"
            if label
            else "Not flagged at the model threshold"
        ),
    }


def create_app(
    predictor: FastFraudPredictor | Any | None = None,
    fast_model: str | Path | None = None,
) -> Flask:
    """Create the local web app, optionally injecting a predictor in tests."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    app.config["SECRET_KEY"] = "local-fraud-demo"

    if predictor is None:
        project_dir = Path(__file__).resolve().parent
        default_fast_model = (
            Path(fast_model)
            if fast_model is not None
            else (
                project_dir
                / "artifacts"
                / "fast_model"
                / "fraud_hist_gradient_boosting.joblib"
            )
        )
        predictor = FastFraudPredictor(default_fast_model)

    fields = predictor.form_fields
    model_description = "Fast PaySim tree model"
    threshold_percent = predictor.threshold * 100

    @app.route("/", methods=["GET", "POST"])
    def index():
        result = None
        error = None
        values = {
            field["name"]: field.get("default", "")
            for field in fields
        }

        if request.method == "POST":
            try:
                row, parsed_values = _parse_input(request.form.to_dict(), fields)
                values.update(parsed_values)
                result = _score_row(row, predictor)
            except ValueError as exception:
                error = str(exception)
                LOGGER.info("Rejected invalid web form input: %s", error)
                for field in fields:
                    name = field["name"]
                    if name in request.form:
                        values[name] = request.form[name].strip()

        return (
            render_template(
                "index.html",
                fields=fields,
                values=values,
                result=result,
                error=error,
                model_description=model_description,
                threshold_percent=threshold_percent,
                test_metrics=predictor.test_metrics,
                test_rows=predictor.test_rows,
                test_fraud_rows=predictor.test_fraud_rows,
            ),
            400 if error else 200,
        )

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return (
                jsonify(
                    {
                        "status": "error",
                        "error": "Invalid request body: expected a JSON object with transaction fields.",
                    }
                ),
                400,
            )
        try:
            row, _ = _parse_input(data, fields)
            result = _score_row(row, predictor)
            return (
                jsonify(
                    {
                        "status": "success",
                        "fraud_probability": result["probability"],
                        "fraud_probability_percent": result["percent_display"],
                        "predicted_label": result["label"],
                        "is_fraud": bool(result["label"]),
                        "decision": result["decision"],
                        "threshold": predictor.threshold,
                        "balance_warnings": result["balance_warnings"],
                    }
                ),
                200,
            )
        except ValueError as exception:
            return jsonify({"status": "error", "error": str(exception)}), 400

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy", "model": model_description}), 200

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast-model",
        help="Fast tree-model artifact.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")

    app = create_app(fast_model=args.fast_model)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
