# ML Project 1 – Heart disease prediction (EPFL CS-433, Fall 2026)

Binary classification of coronary heart disease / heart attack (MICHD) from the
BRFSS 2015 health survey (328,135 training people, 321 survey columns).
Metric: F1 score. Only NumPy (+ standard library) is used.

## Reproduce the submission

1. Put `x_train.csv`, `y_train.csv`, `x_test.csv` in `data/`.
2. From the repo root: `python run.py` (a few minutes, mostly spent parsing the csv files; writes `submission.csv`).

## Files

| File | Content |
| --- | --- |
| `run.py` | Full pipeline that reproduces our best submission |
| `implementations.py` | The 6 required methods (GD, SGD, least squares, ridge, logistic, regularized logistic) |
| `src/helpers.py` | Course helpers: `load_csv_data`, `create_csv_submission` |
| `src/special_function.py` | Decoders for the 20 columns that pack a unit inside the code (e.g. `205` = 5 per week) |
| `src/preprocess.py` | `fit_preprocess` / `transform`: column cleaning, value mapping, NaN filter, one-hot, log1p, imputation, standardization |
| `src/models.py` | Class-weighted logistic regression (tested in CV, not used in the final model) |
| `src/cross_validation.py` | K-fold cross-validation with F1; preprocessing, model and threshold fitted on training folds only |
| `src/implementations.py` | Same as `implementations.py` (imported by the notebook) |
| `data_columns.csv` | Our metadata per column: value mapping, valid range, categorical flag, special flag |
| `projet.ipynb` | Exploration and first baseline experiments |
| `report/report.pdf` | 2-page report |

## Cross-validation

`src/cross_validation.cross_validate` runs the whole pipeline 5 times
(5-fold, seed 1). Inside each fold the preprocessing statistics (medians,
mean, std, kept columns, categories), the weights and the decision threshold
are learned on the 4 training folds and the F1 is measured once on the
held-out fold. Results are in the report.
