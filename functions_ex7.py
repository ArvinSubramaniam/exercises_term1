""" Function from Ex. 7"""

import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def calculate_primal_objective(y, X, w, lambda_):
    """compute the full cost (the primal objective), that is loss plus regularizer.
    X: the full dataset matrix, shape = (num_examples, num_features)
    y: the corresponding +1 or -1 labels, shape = (num_examples)
    w: shape = (num_features)
    """
    # ***************************************************
    prod = y * (X.dot(w))
    l = np.clip(1 - prod, 0, float("inf"))
    cost = np.sum(l) + (lambda_/2.)*np.linalg.norm(w)
    return cost

def calculate_accuracy(y, X, w):
    """compute the training accuracy on the training set (can be called for test set as well).
    X: the full dataset matrix, shape = (num_examples, num_features)
    y: the corresponding +1 or -1 labels, shape = (num_examples)
    w: shape = (num_features)
    """
    # ***************************************************
    return np.mean(y == np.sign(X.dot(w))) * 100.

def calculate_stochastic_gradient(y, X, w, lambda_, n, num_examples):
    """compute the stochastic gradient of loss plus regularizer.
    X: the dataset matrix, shape = (num_examples, num_features)
    y: the corresponding +1 or -1 labels, shape = (num_examples)
    w: shape = (num_features)
    n: the index of the (one) datapoint we have sampled
    num_examples: N
    """
    # ***************************************************
    x_n, y_n = X[n], y[n]
    N = num_examples
    z = y_n * x_n.dot(w)
    if z <= 1:
        grad = -N * y_n * x_n #Factor of N because the cost function is a sum instead of an average, unlike previous ones
    if z > 0:
        grad = 0.
    return grad
    # ***************************************************
    
def sgd_for_svm_demo(y, X):
    """To run SGD for svm"""
    max_iter = 100000
    gamma = 1
    lambda_ = 0.01

    num_examples, num_features = X.shape
    w = np.zeros(num_features)
    
    for it in range(max_iter):
        # n = sample one data point uniformly at random data from x
        n = random.randint(0,num_examples-1)
        
        grad = calculate_stochastic_gradient(y, X, w, lambda_, n, num_examples)
        w -= gamma/(it+1) * grad
        
        if it % 10000 == 0:
            cost = calculate_primal_objective(y, X, w, lambda_)
            print("iteration={i}, cost={c}".format(i=it, c=cost))
    
    print("training accuracy = {l}".format(l=calculate_accuracy(y, X, w)))

def calculate_coordinate_update(y, X, lambda_, alpha, w, n):
    """compute a coordinate update (closed form) for coordinate n.
    X: the dataset matrix, shape = (num_examples, num_features)
    y: the corresponding +1 or -1 labels, shape = (num_examples)
    w: shape = (num_features)
    n: the coordinate to be updated
    """
    # ***************************************************
    # ***************************************************
    x_n, y_n = X[n], y[n]
    old_alpha_n = np.copy(alpha[n])
    w = (1/lambda_)*x_n.T*y_n*old_alpha_n#Solving for lambda
    k = x_n.dot(x_n.T)#Kernel matrix
    q = k*(y_n**2)
    alpha = lambda_*(1/q)*np.ones((y.shape[0],1))#Solve for alpha
    return w, alpha

def calculate_dual_objective(y, X, w, alpha, lambda_):
    """calculate the objective for the dual problem."""
    # ***************************************************
    Y = np.diag(y)
    obj = np.sum(alpha) + (alpha.T.dot(Y)).dot(X.dot(w)) + (lambda_/2)*np.linalg.norm(w)
    return obj

def coordinate_descent_for_svm_demo(y, X):
    """Performing dual optimization"""
    max_iter = 100000
    lambda_ = 0.01

    num_examples, num_features = X.shape
    w = np.zeros(num_features)
    alpha = np.zeros(num_examples)
    
    for it in range(max_iter):
        # n = sample one data point uniformly at random data from x
        n = random.randint(0,num_examples-1)
        
        w, alpha = calculate_coordinate_update(y, X, lambda_, alpha, w, n)
            
        if it % 10000 == 0:
            # primal objective
            primal_value = calculate_primal_objective(y, X, w, lambda_)
            # dual objective
            dual_value = calculate_dual_objective(y, X, w, alpha, lambda_)
            # primal dual gap
            duality_gap = primal_value - dual_value
            print('iteration=%i, primal:%.5f, dual:%.5f, gap:%.5f'%(
                    it, primal_value, dual_value, duality_gap))
    print("training accuracy = {l}".format(l=calculate_accuracy(y, X, w)))
    
