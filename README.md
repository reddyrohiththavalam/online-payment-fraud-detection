# Online Payment Fraud Detection

The project trains a fast `HistGradientBoostingClassifier` using the supplied
preprocessed PaySim train/test splits. Fraud rows are all retained, legitimate
rows are sampled to limit memory use, and sample weights correct for the
downsampling. The rule-generated `isFlaggedFraud` column is excluded.

## Setup

From the project directory on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train and evaluate

The default input directory is `data\data\data\processed`:

```powershell
.\.venv\Scripts\python.exe train.py
```

Model selection trains on PaySim steps 1-550, with all fraud cases retained
and up to 250,000 legitimate cases sampled with population-correcting sample
weights. Steps 551-600 are used for validation. After validation, the final
model is refit on training-split rows from steps 1-600. Reported test results
use only untouched test-split rows at steps 601-743, later transactions than
the final model saw during training. The model and report are saved to
`artifacts\fast_model`:

- `fraud_hist_gradient_boosting.joblib`: trained model and feature schema
- `fast_metrics.json`: test metrics and training/test row counts
- `roc_curve.png` and `confusion_matrix.png`: held-out test plots

The current time-separated test window contains 20,848 transactions, including
304 fraud cases. The present model made no errors in that window. This is
evidence about this synthetic PaySim time block only—not proof of perfect
accuracy, calibrated real-world risk, or performance on another payment system.

Use `.\.venv\Scripts\python.exe train.py --help` to see training options.

## Model Evaluation Metrics

Display the full model evaluation report (Accuracy, Precision, Recall, F1, ROC AUC, MCC, Log Loss):

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

To run a live batch evaluation across all 1.27 million test transactions:

```powershell
.\.venv\Scripts\python.exe evaluate.py --full
```

## Score a CSV

The predictor accepts raw PaySim data (including its `type` column) or the
prepared one-hot feature format. Large CSVs are scored in chunks:

```powershell
.\.venv\Scripts\python.exe predict.py --input data\transactions.csv --output artifacts\fast_model\predictions.csv
```

By default, the model is loaded from
`artifacts\fast_model\fraud_hist_gradient_boosting.joblib`. Override it with
`--model`; adjust the batch size with `--chunksize`.

## Run the browser app

```powershell
.\.venv\Scripts\python.exe web_app.py
```

Open `http://127.0.0.1:5000`. Use `--fast-model` to load a model artifact from a
different path.

The web app includes **⚡ Quick Test Presets** to quickly test scenarios:
- **Legitimate Payment**: Routine merchant payment with low fraud probability.
- **Legitimate Transfer**: Normal peer-to-peer transfer with preserved destination credit.
- **Fraud Transfer (Drain)**: Typical PaySim fraud where origin account balance is completely drained.
- **Fraud Cash-Out**: Fraudulent cash-out with emptied balance.

You can also enter values manually. The browser app warns when sender balances do
not follow the usual PaySim balance change for the selected type.

### REST API Endpoint

The server exposes a JSON scoring endpoint:

```powershell
curl -X POST http://127.0.0.1:5000/api/predict `
  -H "Content-Type: application/json" `
  -d '{\"type\": \"TRANSFER\", \"step\": 583, \"amount\": 759701.03, \"oldbalanceOrg\": 759701.03, \"newbalanceOrig\": 0, \"oldbalanceDest\": 0, \"newbalanceDest\": 0}'
```

Response:
```json
{
  "status": "success",
  "is_fraud": true,
  "predicted_label": 1,
  "fraud_probability": 1.0,
  "fraud_probability_percent": "100%",
  "decision": "Flagged for review",
  "threshold": 0.5,
  "balance_warnings": []
}
```

Health check is available at `GET /health`.

## Tests

Run the full automated test suite (including feature engineering, balance warnings, model persistence, prediction accuracy, and REST API tests):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Only load joblib artifacts from trusted sources. Keep real transaction data
private and protect input files and generated artifacts appropriately.
