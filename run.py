"""Reproduces our final AIcrowd submission.

Usage (from the repo root, with x_train.csv, y_train.csv, x_test.csv in data/):
    python run.py

Pipeline:
    1. load the csv files (course helper)
    2. preprocessing fitted on the full training set (src/preprocess.py)
    3. regularized logistic regression (implementations.py)
    4. decision threshold maximizing F1 on the training predictions
    5. predictions in {-1, 1} written to submission.csv
The hyperparameters below were chosen by 5-fold cross-validation
(see src/cross_validation.py and the report).
"""

import numpy as np

from src.helpers import load_csv_data, create_csv_submission
from src.special_function import read_column_names
from src.preprocess import fit_preprocess, transform
from implementations import reg_logistic_regression
from src.cross_validation import best_threshold

DATA_PATH = "data"
PREP_CONFIG = {}      # DEFAULT_CONFIG of src/preprocess.py
LAMBDA = 1e-4         # L2 regularization
GAMMA = 0.5           # step size
MAX_ITERS = 300       # gradient steps


def main():
    """Runs the full pipeline and writes submission.csv."""
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_PATH)
    names = read_column_names(f"{DATA_PATH}/x_train.csv")

    # preprocessing statistics learned on the training set only
    stats = fit_preprocess(x_train, names, config=PREP_CONFIG)
    tx_train = transform(x_train, stats)
    tx_test = transform(x_test, stats)
    print("features:", tx_train.shape[1])

    # labels {-1, 1} -> {0, 1} for logistic regression
    y01 = (y_train + 1) / 2
    initial_w = np.zeros(tx_train.shape[1])
    w, loss = reg_logistic_regression(y01, tx_train, LAMBDA, initial_w, MAX_ITERS, GAMMA)
    threshold, f1_train = best_threshold(tx_train @ w, y_train)
    print(f"train loss {loss:.4f}, threshold {threshold:.3f}, train F1 {f1_train:.4f}")

    y_pred = np.where(tx_test @ w >= threshold, 1, -1)
    create_csv_submission(test_ids, y_pred, "submission.csv")
    print("predicted positive rate:", np.mean(y_pred == 1))


if __name__ == "__main__":
    main()
