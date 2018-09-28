# -*- coding: utf-8 -*-
# gradient_methods.py
"""N-D Optimization 2 (Gradient Descent Methods).
Bailey Smith
March 2 2017
"""
import numpy as np
from scipy import linalg as la
from scipy.optimize import fmin_cg

def steepestDescent(f, Df, x0, step=1, tol=.0001, maxiters=50):
    """Use the Method of Steepest Descent to find the minimizer x of the convex
    function f:Rn -> R.

    Parameters:
        f (function Rn -> R): The objective function.
        Df (function Rn -> Rn): The gradient of the objective function f.
        x0 ((n,) ndarray): An initial guess for x.
        step (float): The initial step size.
        tol (float): The convergence tolerance.

    Returns:
        x ((n,) ndarray): The minimizer of f.

    while inside for loop? reset step to 1 each time
    """
    i = 0
    alpha = step
    while i < maxiters:
        
        if f(x0 - step*Df(x0)) < f(x0):
            x = x0 - step * Df(x0)
            x0 = x
            step = alpha
        else:
            step *= .5
            i -=1
        if step < tol:
            break
        i += 1
    return x

def conjugateGradient(b, x0, Q, tol=.0001):
    """Use the Conjugate Gradient Method to find the solution to the linear
    system Qx = b.

    Parameters:
        b  ((n, ) ndarray)
        x0 ((n, ) ndarray): An initial guess for x.
        Q  ((n,n) ndarray): A positive-definite square matrix.
        tol (float): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The solution to the linear systm Qx = b, according
            to the Conjugate Gradient Method.
    """
    r = Q.dot(x0) - b
    d = - r
    while la.norm(r) > tol:
        alpha = la.norm(r)**2/(d.T).dot(Q).dot(d)
        x = x0 + alpha*d
        rk = r + alpha*Q.dot(d)
        Bk = la.norm(rk)**2/la.norm(r)**2
        dk = -rk + Bk*d
        d = dk
        r = rk
        x0 = x
    return x

def lin_reg(filename="linregression.txt"):
    """Use conjugateGradient() to solve the linear regression problem with
    the data from linregression.txt.
    Return the solution x*.
    """
    data = np.loadtxt(filename)
    b =  np.copy(data[:,0])
    data[:,0] = 1
    # print data, b
    A = data
    x0 = data[0,:]/data[0,:]
    return conjugateGradient(A.T.dot(b), x0, A.T.dot(A))

def log_reg(filename="logregression.txt"):
    """Use scipy.optimize.fmin_cg() to find the maximum likelihood estimate
    for the data in logregression.txt.
    """
    data = np.loadtxt(filename)
    y = np.copy(data[:,0])
    data[:,0] = 1
    x = data
    def objective(b):
        #Return -1*l(b[0], b[1]), where l is the log likelihood.
        return (np.log(1+np.exp(x.dot(b))) - y*(x.dot(b))).sum()

    guess = np.array([1., 1., 1., 1.])
    # b = fmin_cg(objective, guess)
    return fmin_cg(objective, guess)

