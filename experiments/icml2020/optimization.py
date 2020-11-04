"""Logistic regression model"""

import numpy as np
from strategic import best_response


def sigmoid(z):
    """Evaluate sigmoid function"""
    return 1 / (1 + np.exp(-z))


def evaluate_loss(X, Y, theta, lam, strat_features=[], epsilon=0):
    """Evaluate L2-regularized logistic regression loss function. For epsilon>0 it returns the performative loss.

    Parameters
    ----------
        X: np.array
            training data matrix
        Y: np.array
            labels
        theta: np.array
            parameter vector
        lam: float
            regulariaation parameter, lam>0
        strat_features: list
            list of features that can be manipulated strategically, other features remain fixed
        epsilon: float
            sensitivity parameter, quantifying the strength of performative effects  

    Returns
    -------
        loss: float
            logistic loss value  
    """

    n = X.shape[0]

    # compute strategically manipulated data
    if epsilon > 0:
        X_perf = best_response(X, theta, epsilon, strat_features)
    else:
        X_perf = np.copy(X)

    # compute log likelihood
    t1 = 1.0/n * np.sum(-1.0 * np.multiply(Y, X_perf @ theta) +
                        np.log(1 + np.exp(X_perf @ theta)))

    # add regularization (without considering the bias)
    t2 = lam / 2.0 * np.linalg.norm(theta[:-1]) ** 2
    loss = t1 + t2

    return loss


def logistic_regression(X_orig, Y_orig, lam, method, tol=1e-7, theta_init=None):
    """Training of an L2-regularized logistic regression model.

    Parameters
    ----------
        X_orig: np.array
            training data matrix
        Y_orig: np.array
            labels
        lam: float
            regulariation parameter, lam>0
        method: string
            optimization method: 'Exact' for returning the exact solution and 'GD' for performing a single gradient descent step on the parameter vector
        tol: float
            stopping criterion for exact minimization
        theta_init: np.array
            initial parameter vector. If None procedure is initialized at zero

    Returns
    -------
        theta: np.array
            updated parameter vector
        loss_list: list
            loss values furing training for reporting
        smoothness: float
            smoothness parameter of the logistic loss function given the current training data matrix
    """

    # assumes that the last coordinate is the bias term
    X = np.copy(X_orig)
    Y = np.copy(Y_orig)
    n, d = X.shape

    # compute smoothness of the logistic loss
    smoothness = np.sum(np.square(np.linalg.norm(X, axis=1))) / (4.0 * n)

    if method == 'Exact':
        eta_init = 1 / (smoothness + lam)  # true smoothness

    elif method == 'GD':
        assert(theta_init is not None)
        eta_init = 2 / (smoothness + 2 * lam)

    else:
        print('method must be Exact or GD')
        raise ValueError

    if theta_init is not None:
        theta = np.copy(theta_init)
    else:
        theta = np.zeros(d)

    # evaluate initial loss
    prev_loss = evaluate_loss(X, Y, theta, lam)

    loss_list = [prev_loss]
    is_gd = False
    i = 0
    gap = 1e30

    eta = eta_init

    while gap > tol and not is_gd:

        # take gradients
        exp_tx = np.exp(X @ theta)
        c = exp_tx / (1 + exp_tx) - Y
        gradient = 1.0/n * \
            np.sum(X * c[:, np.newaxis], axis=0) + \
            lam * np.append(theta[:-1], 0)

        new_theta = theta - eta * gradient

        # compute new loss
        t1 = 1.0/n * np.sum(-1 * np.multiply(Y, X @ new_theta) +
                            np.log(1 + np.exp(X @ new_theta)))
        t2 = lam / 2 * np.linalg.norm(new_theta[:-1])
        loss = t1 + t2

        # do backtracking line search
        if loss > prev_loss and method == 'Exact':
            eta = eta * .1
            gap = 1e30
            continue
        else:
            eta = eta_init

        theta = np.copy(new_theta)

        loss_list.append(loss)
        gap = prev_loss - loss
        prev_loss = loss

        if method == 'GD':
            is_gd = True

        i += 1

    return theta, loss_list, smoothness
