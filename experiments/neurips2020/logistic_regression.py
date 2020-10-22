"""
logistic regression
----------------------
utility function for evaluating and training an L2-regularized logistic regression classifier
the objective is stated explicitly in Appendix A.2
"""

import numpy as np

# evaluate logistic regression loss
def evaluate_loss(X, Y, theta, lam, eps=0, strat_features=None):
    
    # if eps>0 we evaluate the perfromative loss
    if eps > 0:
        X_strat = best_response(X, theta, eps, strat_features) 
    else:
        X_strat = X
        
    n = X_strat.shape[0]
    t1 = 1.0/n * np.sum( -1.0 * np.multiply(Y, X_strat @ theta) + np.log(1 + np.exp(X_strat @ theta)))
    t2 = lam / 2.0 * np.linalg.norm(theta[:-1])**2
    loss = t1 + t2
    
    return loss

# gradient descent with the line search procedure for training a logistic regression model 
def run_gd(X, Y, lam, n_steps = 1, tol=1e-7, theta_init=None):
        
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
    prev_loss  = evaluate_loss(X, Y, theta, lam)
    loss_list  = [prev_loss]
    theta_list = [theta]
    
    gap   = 1e30
    eta   = eta_init
    count = 0
    
    while count < n_steps:
        
        if gap < tol:
            break
            
        exp_tx   = np.exp(X @ theta)
        c        = exp_tx / (1 + exp_tx) - Y
        gradient = 1.0/n * np.sum(X * c[:, np.newaxis], axis=0) + lam * np.append(theta[:-1], 0)
                
        theta_new = theta - eta * gradient
        
        if eta<1e-10:
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
        
        gap       = prev_loss - loss
        prev_loss = loss
            
        count = count + 1
           
    return theta_list, loss_list
