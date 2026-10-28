import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from fast_inference import FastFraudPredictor
from predict import main as predict_main
from train_fast import _engineer_features, train_and_evaluate
from web_app import _balance_warnings, create_app


class FastTrainingTests(unittest.TestCase):
    def test_cash_in_balance_warnings_identify_inconsistent_values(self):
        transaction = {
            "type": "CASH_IN",
            "amount": 799999,
            "oldbalanceOrg": 800000,
            "newbalanceOrig": 89999999,
            "oldbalanceDest": 89999,
            "newbalanceDest": 6888,
        }

        warnings = _balance_warnings(transaction)

        self.assertEqual(len(warnings), 2)
        self.assertIn("1,599,999.00", warnings[0])
        self.assertIn("recipient-side balance", warnings[1])
        consistent_transaction = {
            **transaction,
            "newbalanceOrig": 1599999,
            "newbalanceDest": 0,
        }
        self.assertEqual(_balance_warnings(consistent_transaction), [])

    def test_feature_engineering_drops_rule_flag_and_adds_balance_errors(self):
        features = pd.DataFrame(
            {
                "amount": [10.0],
                "oldbalanceOrg": [25.0],
                "newbalanceOrig": [15.0],
                "oldbalanceDest": [5.0],
                "newbalanceDest": [15.0],
                "isFlaggedFraud": [1],
            }
        )

        result = _engineer_features(features)

        self.assertNotIn("isFlaggedFraud", result.columns)
        self.assertEqual(result.loc[0, "errorBalanceOrig"], 0.0)
        self.assertEqual(result.loc[0, "errorBalanceDest"], 0.0)

    def test_fast_training_saves_model_and_held_out_metrics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "processed"
            output_dir = root / "fast_model"
            data_dir.mkdir()
            rng = np.random.default_rng(42)

            def create_split(name: str, rows: int) -> None:
                labels = np.tile([0, 0, 0, 0, 1], rows // 5)
                fraud = labels.astype(bool)
                amount = rng.uniform(1, 100, rows)
                amount[fraud] += 200
                if name == "train":
                    steps = np.where(np.arange(rows) % 4 == 0, 575, 300)
                else:
                    steps = np.where(np.arange(rows) % 2 == 0, 650, 300)
                features = pd.DataFrame(
                    {
                        "step": steps,
                        "amount": amount,
                        "oldbalanceOrg": rng.uniform(0, 500, rows),
                        "newbalanceOrig": rng.uniform(0, 500, rows),
                        "oldbalanceDest": rng.uniform(0, 500, rows),
                        "newbalanceDest": rng.uniform(0, 500, rows),
                        "isFlaggedFraud": np.zeros(rows, dtype=np.int8),
                        "type_TRANSFER": fraud.astype(np.int8),
                    }
                )
                features.to_csv(data_dir / f"X_{name}.csv", index=False)
                pd.DataFrame({"isFraud": labels}).to_csv(
                    data_dir / f"y_{name}.csv",
                    index=False,
                )

            create_split("train", 200)
            create_split("test", 100)
            args = type(
                "Args",
                (),
                {
                    "data_dir": str(data_dir),
                    "output_dir": str(output_dir),
                    "max_negative_samples": 80,
                    "max_iterations": 5,
                    "max_leaf_nodes": 7,
                    "learning_rate": 0.1,
                    "seed": 42,
                },
            )()

            report = train_and_evaluate(args)

            self.assertEqual(report["training_split_rows"], 200)
            self.assertEqual(report["validation_training_step_range"], [1, 550])
            self.assertEqual(report["training_step_range"], [1, 600])
            self.assertEqual(report["validation_step_range"], [551, 600])
            self.assertEqual(report["validation_rows"], 50)
            self.assertEqual(report["test_step_range"], [601, None])
            self.assertEqual(report["test_rows"], 50)
            self.assertEqual(report["test_fraud_rows"], 10)
            self.assertTrue((output_dir / "fraud_hist_gradient_boosting.joblib").is_file())
            self.assertTrue((output_dir / "roc_curve.png").is_file())
            self.assertTrue((output_dir / "confusion_matrix.png").is_file())
            metrics = json.loads(
                (output_dir / "fast_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["metrics"]["threshold"], 0.5)
            self.assertIn("average_precision", metrics["metrics"])
            self.assertTrue(np.isfinite(metrics["metrics"]["auc_roc"]))

            predictor = FastFraudPredictor(
                output_dir / "fraud_hist_gradient_boosting.joblib"
            )
            self.assertEqual(predictor.test_rows, 50)
            self.assertEqual(predictor.test_fraud_rows, 10)
            form_fields = {
                field["name"]: field
                for field in predictor.form_fields
                if field["kind"] == "number"
            }
            self.assertEqual(form_fields["amount"]["default"], "")
            self.assertIn("typical:", form_fields["amount"]["placeholder"])
            transactions = pd.DataFrame(
                {
                    "step": [3, 5],
                    "type": ["TRANSFER", "PAYMENT"],
                    "amount": [500.0, 20.0],
                    "nameOrig": ["C1", "C2"],
                    "oldbalanceOrg": [500.0, 50.0],
                    "newbalanceOrig": [0.0, 30.0],
                    "nameDest": ["C3", "M4"],
                    "oldbalanceDest": [0.0, 100.0],
                    "newbalanceDest": [500.0, 120.0],
                    "isFlaggedFraud": [0, 0],
                }
            )
            predictions = predictor.predict(transactions)
            self.assertEqual(len(predictions), 2)
            self.assertTrue(predictions["fraud_probability"].between(0, 1).all())
            self.assertTrue(predictions["predicted_label"].isin([0, 1]).all())
            self.assertIn("nameOrig", predictions.columns)
            with self.assertRaisesRegex(ValueError, "missing required model features"):
                predictor.predict(transactions.drop(columns="amount"))

            app = create_app(predictor=predictor)
            app.testing = True
            client = app.test_client()
            page = client.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn(b"Fast PaySim tree model", page.data)
            self.assertIn(b'value="" disabled selected', page.data)
            self.assertIn(b"time-separated PaySim test set", page.data)
            response = client.post(
                "/",
                data={
                    "type": "TRANSFER",
                    "step": "3",
                    "amount": "500",
                    "oldbalanceOrg": "500",
                    "newbalanceOrig": "0",
                    "oldbalanceDest": "0",
                    "newbalanceDest": "500",
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Estimated PaySim fraud score", response.data)
            self.assertIn(b"The model flags transactions at or above 50.0%", response.data)
            self.assertIn(b"simulated PaySim data", response.data)
            self.assertIn(b"account identities", response.data)
            self.assertIn(b"transaction history", response.data)

            class LowProbabilityPredictor:
                form_fields = [{"name": "amount", "label": "amount", "kind": "number", "default": "1"}]
                threshold = 0.5
                test_metrics = {}
                test_rows = 0
                test_fraud_rows = 0

                @staticmethod
                def predict(_transactions):
                    return pd.DataFrame(
                        {
                            "fraud_probability": [0.0000008199543116603912],
                            "predicted_label": [0],
                        }
                    )

            low_probability_app = create_app(predictor=LowProbabilityPredictor())
            low_probability_app.testing = True
            low_probability_page = low_probability_app.test_client().post(
                "/",
                data={"amount": "1"},
            )
            self.assertEqual(low_probability_page.status_code, 200)
            self.assertIn(b"0.0000819954%", low_probability_page.data)

            input_path = root / "raw_transactions.csv"
            output_path = root / "scored_transactions.csv"
            transactions.to_csv(input_path, index=False)
            with patch(
                "sys.argv",
                [
                    "predict.py",
                    "--model",
                    str(output_dir / "fraud_hist_gradient_boosting.joblib"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--chunksize",
                    "1",
                ],
            ):
                predict_main()
            scored = pd.read_csv(output_path)
            self.assertEqual(len(scored), 2)
            self.assertIn("fraud_probability", scored.columns)


if __name__ == "__main__":
    unittest.main()
