"""Score transactions with a trained fraud model bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fast_inference import FastFraudPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="artifacts/fast_model/fraud_hist_gradient_boosting.joblib",
        help="Fast tree-model artifact.",
    )
    parser.add_argument("--input", required=True, help="Input transaction CSV.")
    parser.add_argument("--output", default="artifacts/fast_model/predictions.csv")
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Rows per batch when scoring the CSV.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"Input CSV does not exist: {input_path}")
    if args.chunksize < 1:
        parser.error("--chunksize must be positive.")

    predictor = FastFraudPredictor(args.model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    first_chunk = True
    for transactions in pd.read_csv(input_path, chunksize=args.chunksize):
        predictions = predictor.predict(transactions)
        predictions.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        total_rows += len(predictions)
        first_chunk = False
    if first_chunk:
        raise ValueError(f"Input CSV contains no transactions: {input_path}")
    print(
        f"Scored {total_rows} transactions; predictions saved to "
        f"{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
