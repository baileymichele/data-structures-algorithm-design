# gaussian_quadrature.py
"""Gaussian Quadrature.
Bailey Smith
"""

import numpy as np
from math import sqrt
from scipy.stats import norm
from scipy import sparse as sp
from scipy.integrate import quad
from matplotlib import pyplot as plt

def shift(f, a, b, plot=False):
    """Shift the function f on [a, b] to a new function g on [-1, 1] such that
    the integral of f from a to b is equal to the integral of g from -1 to 1.

    Inputs:
        f (function): a scalar-valued function on the reals.
        a (int): the left endpoint of the interval of integration.
        b (int): the right endpoint of the interval of integration.
        plot (bool): if True, plot f over [a,b] and g over [-1,1] in separate
            subplots.

    Returns:
        The new, shifted function.
    """
    g = lambda u: f((b-a)/2.*u + (a+b)/2.)
    if plot == True:
        domain1 = np.linspace(a,b,100)
        domain2 = np.linspace(-1,1,100)

        plt.subplot(121)
        plt.plot(domain1, f(domain1))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("f")

        plt.subplot(122)
        plt.plot(domain2,g(domain2))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("g")

        plt.show()
    return g


def estimate_integral(f, a, b, points, weights):
    """Estimate the value of the integral of the function f over [a,b].

    Inputs:
        f (function): a scalar-valued function on the reals.
        a (int): the left endpoint of the interval of integration.
        b (int): the right endpoint of the interval of integration.
        points ((n,) ndarray): an array of n sample points.
        weights ((n,) ndarray): an array of n weights.

    Returns:
        The approximate integral of f over [a,b].
    """
    g = shift(f,a,b)
    return ((b-a)/2.)*weights.dot(g(points))

def construct_jacobi(gamma, alpha, beta):
    """Construct the Jacobi matrix."""
    a = []
    b = []
    for i in xrange(len(gamma)):
        a.append(-beta[i]/alpha[i])
        if i != len(gamma)-1:
            b.append((gamma[i+1]/float((alpha[i]*alpha[i+1])))**.5)
    print a, b
    return sp.diags([a,b,b],[0,-1,1]).todense()#could use np.diags


def points_and_weights(n):
    """Calculate the points and weights for a quadrature over [a,b] with n
    points.

    Returns:
        points ((n,) ndarray): an array of n sample points.
        weights ((n,) ndarray): an array of n weights.
    """
    alpha = np.zeros(n)
    beta = np.zeros(n)
    gamma = np.zeros(n)
    for i in xrange(1,n+1):
        gamma[i-1] = (i-1)/float((i))
        alpha[i-1] = (2*i-1)/float((i))
    jacobi = construct_jacobi(gamma,alpha,beta)
    points, eigvec = np.linalg.eigh(jacobi)
    weights = 2*np.power(eigvec[0,:], 2)
    return points, weights


def gaussian_quadrature(f, a, b, n):
    """Using the functions from the previous problems, integrate the function
    'f' over the domain [a,b] using 'n' points in the quadrature.
    """
    x,w = points_and_weights(n)
    return estimate_integral(f, a, b, x, w)


def normal_cdf(x):
    """Use scipy.integrate.quad() to compute the CDF of the standard normal
    distribution at the point 'x'. That is, compute P(X <= x), where X is a
    normally distributed random variable with mean = 0 and std deviation = 1.
    """
    f = lambda t: (1/(sqrt(2*np.pi)))*np.exp(-t**2/2.)
    return quad(f,np.NINF,x)[0]


