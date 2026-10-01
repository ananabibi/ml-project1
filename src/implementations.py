import numpy as np


# ---------- helper functions ----------
def compute_mse(y, tx, w):
    """MSE loss with the factor 1/2 (as in the lecture): 1/(2N) * ||y - tx w||^2."""
    e = y - tx @ w
    return e @ e / (2 * len(y))


def compute_mse_gradient(y, tx, w):
    """Gradient of the linear MSE: -1/N * tx^T e."""
    e = y - tx @ w
    return -tx.T @ e / len(y)


def sigmoid(t):
    """Logistic function."""
    return 1.0 / (1.0 + np.exp(-t))


def compute_logistic_loss(y, tx, w):
    """Negative log-likelihood (y in {0,1}), averaged over N."""
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    """Gradient of the logistic loss: 1/N * tx^T (sigmoid(tx w) - y)."""
    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


# ---------- the 6 required functions ----------
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * compute_mse_gradient(y, tx, w)
    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using SGD with mini-batch size 1."""
    w = initial_w
    for _ in range(max_iters):
        n = np.random.randint(len(y))                      # pick one datapoint
        w = w - gamma * compute_mse_gradient(y[n:n+1], tx[n:n+1], w)
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """Least squares regression using the normal equations."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)               # solve, never invert
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations."""
    N, D = tx.shape
    lambda_prime = 2 * N * lambda_
    w = np.linalg.solve(tx.T @ tx + lambda_prime * np.eye(D), tx.T @ y)
    return w, compute_mse(y, tx, w)                        # loss WITHOUT penalty


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0,1})."""
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * compute_logistic_gradient(y, tx, w)
    return w, compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using GD, penalty lambda_ * ||w||^2."""
    w = initial_w
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    return w, compute_logistic_loss(y, tx, w)              # loss WITHOUT penalty