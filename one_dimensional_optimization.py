# one_dimensional_optimization.py
"""1-D Optimization.
Bailey Smith
15 February 2017
"""
import numpy as np
import time

def golden_section(f, a, b, niter=10):
    """Find the minimizer of the unimodal function f on the interval [a,b]
    using the golden section search method.

    Inputs:
        f (function): unimodal scalar-valued function on R.
        a (float): left bound of the interval of interest.
        b (float): right bound of the interval of interest.
        niter (int): number of iterations to compute.

    Returns:
        the approximated minimizer (the midpoint of the final interval).
    """
    row = .5*(3-np.sqrt(5))
    for i in xrange(niter):
        a2 = a + row*(b-a)
        b2 = a + (1-row)*(b-a)
        if f(a2) < f(b2):
            b = b2
        else:
            a = a2
    return (a+b)/2.


def bisection(df, a, b, niter=10):
    """Find the minimizer of the unimodal function with derivative df on the
    interval [a,b] using the bisection algorithm.

    Inputs:
        df (function): derivative of a unimodal scalar-valued function on R.
        a (float): left bound of the interval of interest.
        b (float): right bound of the interval of interest.
        niter (int): number of iterations to compute.
    """
    for i in xrange(niter):
        mid = (a+b)/2.
        if df(mid) > 0:
            b = mid
        else:
            a = mid

    print "Bisection method converges faster"
    return (a+b)/2.

def newton1d(f, df, ddf, x, niter=10):
    """Minimize the scalar function f with derivative df and second derivative
    df using Newton's method.

    Parameters
        f (function): A twice-differentiable scalar-valued function on R.
        df (function): The first derivative of f.
        ddf (function): The second derivative of f.
        x (float): The initial guess.
        niter (int): number of iterations to compute.

    Returns:
        The approximated minimizer.
    """
    for i in xrange(niter):
        x_new = x - df(x)/ddf(x)
        x = x_new
    return x


def secant1d(f, df, x0, x1, niter=10):
    """Minimize the scalar function f using the secant method.

    Inputs:
        f (function): A differentiable scalar-valued function on R.
        df (function): The first derivative of f.
        x0 (float): A first initial guess.
        x1 (float): A second initial guess.
        niter (int): number of iterations to compute.

    Returns:
        The approximated minimizer.
    """
    for i in xrange(niter):
        x_new = x1 - df(x1)*(x1 - x0)/(df(x1)-df(x0))
        x0 = x1
        x1 = x_new
    return x_new


def backtracking(f, slope, x, p, a=1, rho=.9, c=10e-4):
    """Do a backtracking line search to satisfy the Wolfe Conditions.
    Return the step length.

    Inputs:
        f (function): A scalar-valued function on R.
        slope (float): The derivative of f at x.
        x (float): The current approximation to the minimizer.
        p (float): The current search direction.
        a (float): Initial step length (set to 1 in Newton and quasi-Newton
            methods).
        rho (float): Parameter in (0,1).
        c (float): Parameter in (0,1).

    Returns:
        The computed step size.
    """
    while f(x + a*p) > f(x) + c*a*slope*p:
        a = rho*a
    return a

if __name__ == '__main__':
    f = lambda x: np.exp(x) - 4*x
    df = lambda x: np.exp(x) - 4
    f = lambda x: x**2
    start = time.time()
    print golden_section(f, 0, 3)
    time1 = time.time() - start

    start = time.time()
    print bisection(df, 0, 3, niter=10)
    time2 = time.time() - start

    print time1, time2

    f = lambda x: x**2 + np.sin(5*x)
    df = lambda x: 2*x + 5*np.cos(5*x)
    ddf = lambda x: 2 - 25*np.sin(5*x)
    x = 0
    print newton1d(f, df, ddf, x, niter=10)

    f = lambda x: x**2 + np.sin(x) + np.sin(10*x)
    df = lambda x: 2*x + np.cos(x) + 10*np.cos(10*x)
    x0, x1 = 0, -1
    print secant1d(f, df, x0, x1, niter=10)
