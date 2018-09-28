# quasi_newton.py
"""Quasi-Newton Methods
Bailey Smith
February 23 2017
"""

import time
import numpy as np
import scipy.linalg as la
import scipy.optimize as so
from scipy.optimize import leastsq
from matplotlib import pyplot as plt

def newton_ND(J, H, x0, niter=10, tol=1e-5):
    """
    Perform Newton's method in N dimensions.

    Inputs:
        J (function): Jacobian of the function f for which we are finding roots.
        H (function): Hessian of f.
        x0 (float): The initial guess.
        niter (int): Max number of iterations to compute.
        tol (float): Stopping criteria for iterations.

    Returns:
        The approximated root and the number of iterations it took.
    """
    for i in xrange(niter):
        x = x0 - la.solve(H(x0),J(x0))
        error = la.norm(x-x0)
        x0 = x
        if error < tol:
            break
    return (x, i+1)#+1 because indexed from 0

def broyden_ND(J, H, x0, niter=20, tol=1e-5):
    """
    Perform Broyden's method in N dimensions.

    Inputs:
        J (function): Jacobian of the function f for which we are finding roots.
        H (function): Hessian of f.
        x0 (float): The initial guess.
        niter (int): Number of iterations to compute.
        tol (float): Stopping criteria for iterations.

    Returns:
        The approximated root and the number of iterations it took.
    """

    A = H(x0)
    error = 1
    for i in xrange(niter):
        if error < tol:
            break
        x = x0 - la.solve(A,J(x0).T)

        sk = x-x0
        yk = J(x).T - J(x0).T
        Ak = A + ((yk - A.dot(sk))/(sk.T.dot(sk))).dot(sk.T)

        error = la.norm(x-x0)
        #reassign A and x0
        A = Ak
        x0 = x

    return (x, i+1)


def BFGS(J, H, x0, niter=10, tol=1e-6):
    """
    Perform BFGS in N dimensions.

    Inputs:
        J (function): Jacobian of objective function.
        H (function): Hessian of objective function.
        x0 (float): The initial guess.
        niter (int): Number of iterations to compute.
        tol (float): Stopping criteria for iterations.

    Returns:
        The approximated root and the number of iterations it took.
    """

    A = H(x0)
    error = 1
    for i in xrange(niter):
        if error < tol:
            break
        x = x0 - la.solve(A,J(x0).T)

        sk = x-x0
        yk = J(x).T - J(x0).T
        Ak = A + (yk.dot(yk.T))/(yk.T.dot(sk)) - (A.dot(sk.dot(sk.T)).dot(A))/(sk.T).dot(A).dot(sk)

        error = la.norm(x-x0)
        #reassign A and x0
        A = Ak
        x0 = x

    return (x, i+1)


def compare_performance():
    """
    Compare the performance of Newton's, Broyden's, and modified Broyden's
    methods on the following functions:
        f(x,y) = 0.26(x^2 + y^2) - 0.48xy
        f(x,y) = sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1
    """
    J1 = lambda x: np.array([.52*x[0]-.48*x[1],.52*x[1]-.48*x[0]])
    H1 = lambda x: np.array([[.52,-.48],[-.48,.52]])
    x1 = (-2,1)

    J2 = lambda x: np.array([np.cos(x[0]+x[1])+2*(x[0]-x[1])-1.5, np.cos(x[0]+x[1])-2*(x[0]-x[1])+2.5])
    H2 = lambda x: np.array([[-np.sin(x[0]+x[1])+2,-np.sin(x[0]+x[1])-2],[-np.sin(x[0]+x[1])-2,-np.sin(x[0]+x[1])+2]])
    x2 = (2,3)

    start = time.time()
    newton =  newton_ND(J1, H1, x1)
    newton_time = time.time() - start

    start = time.time()
    B = broyden_ND(J1, H1, x1)
    B_time = time.time() - start

    start = time.time()
    bfgs = BFGS(J1, H1, x1)
    bfgs_time = time.time() - start

    print "f(x,y) = 0.26(x^2 + y^2) - 0.48xy"
    print "\nNewon's Method:\n", "Converged in", newton[1], "iterations\n", "Total time:", newton_time, "\n", "Time per iteration:", newton_time/newton[1]
    print "\nBroyden's Method:\n", "Converged in", B[1], "iterations\n", "Total time:", B_time, "\n", "Time per iteration:",B_time/B[1]
    print "\nBFGS Method:\n", "Converged in", bfgs[1], "iterations\n", "Total time:", bfgs_time, "\n", "Time per iteration:", bfgs_time/bfgs[1]

    start = time.time()
    newton =  newton_ND(J2, H2, x2)
    newton_time = time.time() - start

    start = time.time()
    B = broyden_ND(J2, H2, x2)
    B_time = time.time() - start

    start = time.time()
    bfgs = BFGS(J2, H2, x2)
    bfgs_time = time.time() - start

    print "\n\n\nf(x,y) = sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1"
    print "\nNewon's Method:\n", "Converged in", newton[1], "iterations\n", "Total time:", newton_time, "\n", "Time per iteration:", newton_time/newton[1]
    print "\nBroyden's Method:\n", "Converged in", B[1], "iterations\n", "Total time:", B_time, "\n", "Time per iteration:",B_time/B[1]
    print "\nBFGS Method:\n", "Converged in", bfgs[1], "iterations\n", "Total time:", bfgs_time, "\n", "Time per iteration:", bfgs_time/bfgs[1]


def gauss_newton(J, r, x0, niter=10):
    """
    Solve a nonlinear least squares problem with Gauss-Newton method.

    Inputs:
        J (function): Jacobian of the objective function.
        r (function): Residual vector.
        x0 (float): The initial guess.
        niter (int): Number of iterations to compute.

    Returns:
        The approximated root.
    """

    for i in xrange(niter):
        x = x0 - la.solve(J(x0).T.dot(J(x0)),J(x0).T.dot(r(x0)))
        x0 = x
    return x

t = np.arange(10)
y = 3*np.sin(0.5*t)+ 0.5*np.random.randn(10)
def model(x, t):
    return x[0]*np.sin(x[1]*t)
def residual(x):
    return model(x, t) - y
def jac(x):
    ans = np.empty((10,2))
    ans[:,0] = np.sin(x[1]*t)
    ans[:,1] = x[0]*t*np.cos(x[1]*t)
    return ans

def lstsq_comp():
    """
    Compare the least squares regression with 8 years of population data and 16
    years of population data.
    """
    years1 = np.arange(8)
    pop1 = np.array([3.929, 5.308, 7.240, 9.638, 12.866,
                 17.069, 23.192, 31.443])

    years2 = np.arange(16)
    pop2 = np.array([3.929, 5.308, 7.240, 9.638, 12.866,
                 17.069, 23.192, 31.443, 38.558, 50.156,
                 62.948, 75.996, 91.972, 105.711, 122.775,
                 131.669])

    def model1(x,t):
        return x[0]*np.exp(x[1]*(t+x[2]))
    def residual1(x):
        return model1(x, years1) - pop1
    x0 = [.9,.3,4.7]
    x1 = [208,.3,-13]


    def model2(x,t):
        return x[0]/(1+np.exp(-x[1]*(t+x[2])))
    def residual2(x):
        return model2(x,years2) - pop2
    # x0 = []

    plt.plot(years1,pop1,'*')
    plt.plot(years2,model1(x0,years2), label='First Model')
    plt.plot(years2,pop2,'*')
    plt.plot(years2,model2(x1,years2),label='Second Model')
    plt.xlabel("Years in tens")
    plt.ylabel("Population")
    plt.legend(loc="upper left")
    plt.show()

if __name__ == '__main__':
    J = lambda x : np.array([x[0]**2 - 2*x[1],3*x[1]])
    H = lambda x : np.array([[2*x[0],-2],[0,3]])
    x0 = (3,5)
    print newton_ND(J, H, x0, 50)

    J = lambda x: np.array([200*(x[1]-x[0]**2)*-2*x[0] - 2*(1-x[0]),200*(x[1]-x[0]**2)])
    H = lambda x: np.array([[-400*x[1]+1200*x[0]**2+2,-400*x[0]],[-400*x[0],200]])
    x0 = (-2,2)
    print newton_ND(J, H, x0)

    J = lambda x: np.array([.52*x[0]-.48*x[1],.52*x[1]-.48*x[0]])
    H = lambda x: np.array([[.52,-.48],[-.48,.52]])
    x0 = (-2,1)
    print broyden_ND(J, H, x0)

    x0 = np.array([3,2])
    J = lambda x: np.array([np.exp(x[0]-1)+2*(x[0]-x[1]),-np.exp(1-x[1])-2*(x[0]-x[1])])
    H = lambda x: np.array([[np.exp(x[0]-1)+2,-2],[-2,np.exp(1-x[1])+2]])
    print broyden_ND(J, H, x0,80)
    print BFGS(J, H, x0)
    print newton_ND(J, H, x0)

    print compare_performance()

    x0 = np.array([2.5,.6])
    print gauss_newton(jac, residual, x0)
    print leastsq(residual, x0)[0]

    print lstsq_comp()
