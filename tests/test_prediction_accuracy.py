import unittest
import pandas as pd
from fast_inference import FastFraudPredictor
from web_app import create_app


class PredictionAccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = FastFraudPredictor("artifacts/fast_model/fraud_hist_gradient_boosting.joblib")
        cls.app = create_app(predictor=cls.predictor)
        cls.app.testing = True
        cls.client = cls.app.test_client()

    def test_legitimate_payment_prediction(self):
        sample = pd.DataFrame([{
            "type": "PAYMENT",
            "step": 280,
            "amount": 120.50,
            "oldbalanceOrg": 1500.00,
            "newbalanceOrig": 1379.50,
            "oldbalanceDest": 0.00,
            "newbalanceDest": 0.00,
        }])
        pred = self.predictor.predict(sample).iloc[0]
        self.assertEqual(pred["predicted_label"], 0)
        self.assertLess(pred["fraud_probability"], 0.01)

    def test_legitimate_transfer_prediction(self):
        sample = pd.DataFrame([{
            "type": "TRANSFER",
            "step": 350,
            "amount": 5000.00,
            "oldbalanceOrg": 12000.00,
            "newbalanceOrig": 7000.00,
            "oldbalanceDest": 10000.00,
            "newbalanceDest": 15000.00,
        }])
        pred = self.predictor.predict(sample).iloc[0]
        self.assertEqual(pred["predicted_label"], 0)
        self.assertLess(pred["fraud_probability"], 0.05)

    def test_fraudulent_transfer_drain_prediction(self):
        sample = pd.DataFrame([{
            "type": "TRANSFER",
            "step": 583,
            "amount": 759701.03,
            "oldbalanceOrg": 759701.03,
            "newbalanceOrig": 0.00,
            "oldbalanceDest": 0.00,
            "newbalanceDest": 0.00,
        }])
        pred = self.predictor.predict(sample).iloc[0]
        self.assertEqual(pred["predicted_label"], 1)
        self.assertGreater(pred["fraud_probability"], 0.90)

    def test_fraudulent_cash_out_prediction(self):
        sample = pd.DataFrame([{
            "type": "CASH_OUT",
            "step": 602,
            "amount": 350000.00,
            "oldbalanceOrg": 350000.00,
            "newbalanceOrig": 0.00,
            "oldbalanceDest": 0.00,
            "newbalanceDest": 350000.00,
        }])
        pred = self.predictor.predict(sample).iloc[0]
        self.assertEqual(pred["predicted_label"], 1)
        self.assertGreater(pred["fraud_probability"], 0.90)

    def test_api_predict_legitimate_json(self):
        payload = {
            "type": "PAYMENT",
            "step": 280,
            "amount": 120.50,
            "oldbalanceOrg": 1500.00,
            "newbalanceOrig": 1379.50,
            "oldbalanceDest": 0.00,
            "newbalanceDest": 0.00,
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["predicted_label"], 0)
        self.assertFalse(data["is_fraud"])
        self.assertLess(data["fraud_probability"], 0.01)
        self.assertEqual(data["decision"], "Not flagged at the model threshold")

    def test_api_predict_fraud_json(self):
        payload = {
            "type": "TRANSFER",
            "step": 583,
            "amount": 759701.03,
            "oldbalanceOrg": 759701.03,
            "newbalanceOrig": 0.00,
            "oldbalanceDest": 0.00,
            "newbalanceDest": 0.00,
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["predicted_label"], 1)
        self.assertTrue(data["is_fraud"])
        self.assertGreater(data["fraud_probability"], 0.90)
        self.assertEqual(data["decision"], "Flagged for review")

    def test_api_predict_validation_errors(self):
        # Missing payload
        res = self.client.post("/api/predict", data="not json", content_type="text/plain")
        self.assertEqual(res.status_code, 400)

        # Missing required field
        res = self.client.post("/api/predict", json={"type": "TRANSFER", "step": 1})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

        # Invalid numeric field
        res = self.client.post(
            "/api/predict",
            json={
                "type": "TRANSFER",
                "step": "invalid_step",
                "amount": 100,
                "oldbalanceOrg": 100,
                "newbalanceOrig": 0,
                "oldbalanceDest": 0,
                "newbalanceDest": 100,
            },
        )
        self.assertEqual(res.status_code, 400)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")


if __name__ == "__main__":
    unittest.main()

