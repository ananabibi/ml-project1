"""K-fold cross-validation with F1, without data leakage.

Inside each fold, the preprocessing statistics, the model and the decision
threshold are all fitted on the training part only; the held-out fold is
used once, for scoring.
"""

import numpy as np

from src.preprocess import fit_preprocess, transform


def build_k_indices(n, k_fold, seed=1):
    """
    Random split of the sample indices into k folds.

    Args:
        n (int): number of samples
        k_fold (int): number of folds
        seed (int): random seed (reproducibility)

    Returns:
        k_indices (np.array): shape (k_fold, n // k_fold)
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    size = n // k_fold
    return np.array([idx[k * size:(k + 1) * size] for k in range(k_fold)])


def f1_score(y_true, y_pred):
    """
    F1 score for labels in {-1, 1}.

    Args:
        y_true (np.array): true labels, shape (N,)
        y_pred (np.array): predicted labels, shape (N,)

    Returns:
        f1 (float): 2 TP / (2 TP + FP + FN)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def best_threshold(scores, y_true, n_grid=200):
    """
    Threshold on the model scores that maximizes F1 on the given data.

    Args:
        scores (np.array): real-valued model outputs, shape (N,)
        y_true (np.array): labels in {-1, 1}, shape (N,)
        n_grid (int): number of candidate thresholds (score quantiles)

    Returns:
        threshold (float): best threshold
        f1 (float): F1 obtained with it
    """
    grid = np.quantile(scores, np.linspace(0.5, 0.995, n_grid))
    f1s = [f1_score(y_true, np.where(scores >= t, 1, -1)) for t in grid]
    k = int(np.argmax(f1s))
    return grid[k], f1s[k]


def cross_validate(x, y, names, fit_model, k_fold=5, seed=1, prep_config=None):
    """
    K-fold cross-validation of a full pipeline (preprocessing + model +
    threshold), everything fitted on the training folds only.

    Args:
        x (np.array): raw data, shape (N, D)
        y (np.array): labels in {-1, 1}, shape (N,)
        names (list): column names of x
        fit_model (callable): fit_model(tx, y01) -> w, scores are tx @ w
        k_fold (int): number of folds
        seed (int): random seed of the split
        prep_config (dict): options passed to fit_preprocess

    Returns:
        f1s (np.array): F1 on each held-out fold, shape (k_fold,)
        thresholds (np.array): threshold chosen in each fold, shape (k_fold,)
    """
    k_indices = build_k_indices(len(y), k_fold, seed)
    f1s, thresholds = [], []
    for k in range(k_fold):
        te = k_indices[k]
        tr = np.concatenate([k_indices[i] for i in range(k_fold) if i != k])
        stats = fit_preprocess(x[tr], names, config=prep_config)
        tx_tr, tx_te = transform(x[tr], stats), transform(x[te], stats)
        w = fit_model(tx_tr, (y[tr] + 1) / 2)
        # threshold chosen on the training folds, never on the held-out fold
        t, _ = best_threshold(tx_tr @ w, y[tr])
        f1s.append(f1_score(y[te], np.where(tx_te @ w >= t, 1, -1)))
        thresholds.append(t)
    return np.array(f1s), np.array(thresholds)
