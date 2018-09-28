# -*- coding: utf-8 -*-
# cvxopt_intro.py
"""CVXOPT
Bailey Smith
January 2 2017
"""

import numpy as np
from scipy import linalg as la
from cvxopt import matrix, solvers


def conv_opt():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      -x - 2y          <= -3
                    -2x - 10y - 3z   <= -10
                    -x               <= 0
                    -y               <= 0
                    -z               <= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    c = np.array([2., 1., 3.])
    G = np.array([[-1., -2., 0.],[-2., -10., -3.],[-1., 0., 0.],[0., -1., 0.], [0., 0., -1.]])
    h = np.array([-3., -10., 0., 0., 0.])

    #Convert to CVXOPT matrix type
    c = matrix(c)
    G = matrix(G)
    h = matrix(h)

    sol = solvers.lp(c, G, h)

    return np.ravel(sol['x']), sol['primal objective']


def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray), without any slack variables u
        The optimal value (sol['primal objective'])

    np.zeros_like(A).astype(float)
    """
    n = np.shape(A)
    zero = np.zeros(n[1]).astype(float)
    ones = np.ones_like(zero)

    I = np.eye(n[1])
    top = np.hstack((-I, I))
    bottom = np.hstack((-I,-I))
    I = np.vstack((top,bottom))

    z = np.zeros_like(A).astype(float)
    a = np.hstack((z,A))

    z2 = np.zeros(2*n[1]).astype(float)
    # z2 = np.reshape(z2,(2*n[1],1))

    # print "z2\n", z2, "\nb\n", b

    c = np.array(np.hstack((ones, zero)))
    G = np.array(np.vstack((I,a,-a)))
    h = np.array(np.append(np.append(z2,b),-b))

    # print "c\n", c, "\nG\n", G, "\nh\n", h, "\n"

    c = matrix(c)
    G = matrix(G)
    h = matrix(h)

    sol = solvers.lp(c, G, h)
    # print "sol['x']", np.ravel(sol['x']), type(np.ravel(sol['x'])), "\nn", n
    # print "\n\n", "sol['primal objective']",sol['primal objective']
    return np.ravel(sol['x'])[n[1]:], sol['primal objective']

def transport_prob():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    c = matrix([4., 7., 6., 8., 8., 9])
    G = -1*np.eye(6)
    G = np.append(G, [[0.,1.,0.,1.,0.,1.],[0.,-1.,0.,-1.,0.,-1.]], axis=0)
    G = matrix(G)
    h = np.zeros(6)
    h = np.append(h,[8,-8], axis=0)
    h = matrix(h)
    A = matrix(np.array([[1.,1.,0.,0.,0.,0.],
                         [0.,0.,1.,1.,0.,0.],
                         [0.,0.,0.,0.,1.,1.],
                         [1.,0.,1.,0.,1.,0.]]))
    b = matrix([7., 2., 4., 5.])
    sol = solvers.lp(c, G, h, A, b)

    return np.ravel(sol['x']), sol['primal objective']


def non_linear_min():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    P = matrix(np.array([[3.,2.,1.],[2.,4.,2.],[1.,2.,3.]]))
    q = matrix([3.,0.,1.])
    sol=solvers.qp(P, q)

    return np.ravel(sol['x']), sol['primal objective']

def l2Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_2
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    n = np.shape(A)

    P = matrix(2*np.eye(n[1]).astype(float))
    q = matrix(np.zeros(n[1]).astype(float))
    sol = solvers.qp(P, q, A=matrix(A.astype(float)), b=matrix(b.astype(float)))

    return np.ravel(sol['x']), sol['primal objective']



def resource_allocation():
    """Solve the allocation model problem in 'ForestData.npy'.
    Note that the first three rows of the data correspond to the first
    analysis area, the second group of three rows correspond to the second
    analysis area, and so on.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective']*-1000)
    """
    i, s, j, p, t, g, w = np.load("ForestData.npy").T
    zeros = np.zeros_like(s)


    tgw = np.vstack((np.vstack((-t[0::], -g[0::])), -w[0::]))
    I = np.eye(21).astype(float)
    h1 = np.array([-40000., -5., -70*788.])

    # print p
    c = np.array(-p)
    G = np.vstack((tgw, -I))
    h = np.append(h1, zeros)

    # print "c\n", c, "\nG\n", G, "\nh\n", h, "\n"

    c = matrix(c)
    G = matrix(G)
    h = matrix(h)
    A = matrix(np.array([[1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.]]))
    b = matrix(s[0::3])

    sol = solvers.lp(c, G, h, A, b)

    return np.ravel(sol['x']), sol['primal objective']*-1000


