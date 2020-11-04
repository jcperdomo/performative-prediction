"""Strategic manipulation of features"""

import numpy as np


def best_response(X, theta, epsilon, strat_features):
    """Best response function for agents given classifier theta. Assumes linear utilities and quadratic costs.

    Parameters
    ----------
        X: np.array
            training data matrix
        theta: np.array
            deployed parameter vector
        epsilon: float
            sensitivity parameter, strength of performative effects
        strat_features: list
            list of features that can be manipulated strategically, other features remain fixed

    Returns
    -------
        X_strat: np.array
            modified training data matrix after each agents best responds to the classifier  
    """

    n = X.shape[0]

    X_strat = np.copy(X)

    for i in range(n):
        # move everything by epsilon in the direction towards better classification
        theta_strat = theta[strat_features]
        X_strat[i, strat_features] += -epsilon * theta_strat

    return X_strat
