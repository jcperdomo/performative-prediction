"""Logistic regression model"""

import numpy as np
from strategic import best_response


def evaluate_loss(X, Y, theta, lam, eps=0, strat_features=None):
    """Evaluate L2-regularized logistic regression loss function. for epsilon>0 it returns the performative loss.

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
            list of features that can be manipulated strategically. other features remain fixed.
        epsilon: float
            sensitivity parameter, quantifying the strength of performative effects.    

    Returns
    -------
        loss: float
            logistic loss value  
    """

    # if eps>0 we evaluate the perfromative loss
    if eps > 0:
        X_strat = best_response(X, theta, eps, strat_features)
    else:
        X_strat = X

    n = X_strat.shape[0]
    t1 = 1.0/n * np.sum(-1.0 * np.multiply(Y, X_strat @
                                           theta) + np.log(1 + np.exp(X_strat @ theta)))
    t2 = lam / 2.0 * np.linalg.norm(theta[:-1])**2
    loss = t1 + t2

    return loss


def run_gd(X, Y, lam, n_steps=1, tol=1e-7, theta_init=None):
    """Gradient descent with line search procedure for training a logistic regression model.

    Parameters
    ----------
        X: np.array
            training data matrix
        Y: np.array
            labels
        lam: float
            regulariation parameter, lam>0
        n_steps: int
            number of gradient descent update steps. if distance between iterates is smaller than tol remaining steps are skipped.
        tol: float
            stopping criterion
        theta_init: np.array
            initial parameter vector. If None procedure is initialized at zero.

    Returns
    -------
        theta: np.array
            updated parameter vector
        loss_list: list
            loss values furing training for reporting
        smoothness: float
            smoothness parameter of the logistic loss function given the current training data matrix
    """

    n, d = X.shape

    # compute smoothness of the logistic loss
    smoothness = np.sum(np.square(np.linalg.norm(X, axis=1))) / (4.0 * n) + lam

    eta_init = 1.0 / smoothness

    # intialize model
    if theta_init is not None:
        theta = np.copy(theta_init)
    else:
        theta = np.zeros(d)

    # evaluate initial loss
    prev_loss = evaluate_loss(X, Y, theta, lam)
    loss_list = [prev_loss]
    theta_list = [theta]

    gap = 1e30
    eta = eta_init
    count = 0

    while count < n_steps:

        if gap < tol:
            break

        exp_tx = np.exp(X @ theta)
        c = exp_tx / (1 + exp_tx) - Y
        gradient = 1.0/n * \
            np.sum(X * c[:, np.newaxis], axis=0) + \
            lam * np.append(theta[:-1], 0)

        theta_new = theta - eta * gradient

        if eta < 1e-10:
            break

        # compute new loss
        loss = evaluate_loss(X, Y, theta_new, lam)

        # do backtracking line search
        if loss > prev_loss:
            eta = eta * .1
            gap = 1e30
            continue
        else:
            eta = eta_init

        theta = np.copy(theta_new)

        loss_list.append(loss)
        theta_list.append(theta)

        gap = prev_loss - loss
        prev_loss = loss

        count = count + 1

    return theta_list, loss_list
