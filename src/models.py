"""Extra model used for the final submission: class-weighted regularized
logistic regression. It extends reg_logistic_regression from
implementations.py with a weight per class, because only 8.8% of the
samples are positive (heart disease)."""

import numpy as np


def sigmoid(t):
    """
    Numerically stable logistic function.

    Args:
        t (np.array): any shape

    Returns:
        np.array: 1 / (1 + exp(-t)), same shape as t
    """
    return 0.5 * (1.0 + np.tanh(0.5 * t))


def weighted_logistic_regression(y, tx, lambda_, max_iters=300, gamma=0.5,
                                 pos_weight=1.0, initial_w=None):
    """
    Gradient descent on the class-weighted negative log-likelihood
    + lambda * ||w||^2.

    Args:
        y (np.array): labels in {0, 1}, shape (N,)
        tx (np.array): features with bias column, shape (N, D)
        lambda_ (float): L2 regularization strength
        max_iters (int): number of gradient steps
        gamma (float): step size
        pos_weight (float): weight of positive samples (negatives weigh 1)
        initial_w (np.array): starting weights, shape (D,), zeros if None

    Returns:
        w (np.array): last weights, shape (D,)
        loss (float): weighted mean NLL, without the penalty term
    """
    w = np.zeros(tx.shape[1]) if initial_w is None else initial_w.copy()
    sample_w = np.where(y == 1, pos_weight, 1.0)
    sample_w = sample_w / sample_w.mean()  # keep the gradient scale stable
    for _ in range(max_iters):
        err = sample_w * (sigmoid(tx @ w) - y)
        grad = tx.T @ err / len(y) + 2 * lambda_ * w
        w = w - gamma * grad
    z = tx @ w
    loss = np.mean(sample_w * (np.logaddexp(0, z) - y * z))
    return w, loss
