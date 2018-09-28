"""Interior Point 1 (Linear Programs)
Bailey Smith
April 5 2017
"""

import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt


# Auxiliary Functions ---------------------------------------------------------
def startingPoint(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A.dot(A.T))
    x = A.T.dot(B.dot(b))
    lam = B.dot(A.dot(c))
    mu = c - A.T.dot(lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use linear program generator to test interior point method.
def randomLP(m):
    """Generate a 'square' linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add slack variables.
    Inputs:
        m -- positive integer: the number of desired constraints
             and the dimension of space in which to optimize.
    Outputs:
        A -- array of shape (m,n).
        b -- array of shape (m,).
        c -- array of shape (n,).
        x -- the solution to the LP.
    """
    n = m
    A = np.random.random((m,n))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(n)*10
    b = A.dot(x)
    c = A.sum(axis=0)/float(n)
    return A, b, -c, x

# This random linear program generator is more general than the first.
def randomLP2(m,n):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Inputs:
        m -- positive integer >= n, number of desired constraints
        n -- dimension of space in which to optimize
    Outputs:
        A -- array of shape (m,n+m)
        b -- array of shape (m,)
        c -- array of shape (n+m,), with m trailing 0s
        v -- the solution to the LP
    """
    A = np.random.random((m,n))*20 - 10
    A[A[:,-1]<0] *= -1
    v = np.random.random(n)*10
    k = n
    b = np.zeros(m)
    b[:k] = A[:k,:].dot(v)
    b[k:] = A[k:,:].dot(v) + np.random.random(m-k)*10
    c = np.zeros(n+m)
    c[:n] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(m)))
    return A, b, -c, v


def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    # Define F (prob 1)
    def F(x,lam, mu):
        one = A.T.dot(lam) + mu - c
        two = A.dot(x) - b
        three = np.diag(mu).dot(x)
        return np.hstack((np.hstack((one,two)),three))
    # initialize
    m,n = np.shape(A)
    x, lam, mu = startingPoint(A,b,c)
    sigma = 1./10

    for i in xrange(niter):
        V = x.T.dot(mu)/float(n)
        M = np.diag(mu)
        X = np.diag(x)

        # Define DF
        one = np.vstack((np.zeros((n,n)), np.vstack((A, M))))
        two = np.vstack((A.T, np.vstack((np.zeros((m,m)), np.zeros((n,m))))))
        three = np.vstack((np.eye(n),np.vstack((np.zeros((m,n)),X))))
        Df = np.hstack((one,np.hstack((two,three))))

        # Determine direction (21.2 prob 2)
        e_ = np.ones_like(mu)
        plus = np.hstack((np.hstack((np.zeros_like(x),np.zeros_like(lam))),sigma*V*(e_)))
        s_direction = la.solve(Df,-1*F(x,lam,mu) + plus)

        delta_x = s_direction[:n]
        delta_lam = s_direction[n:n+m]
        delta_mu = s_direction[n+m:]

        # determine step size
        neg = delta_mu < 0
        negx = delta_x < 0

        if sum(neg) > 0:
            alpha_max = min(1, np.min((-1*mu/delta_mu)[neg]))
        elif sum(neg) == 0:
            alpha_max = 1

        if sum(negx) > 0:
            delta_max = min(1, np.min((-1*x/delta_x)[negx]))
        elif sum(negx) == 0:
            delta_max = 1


        Alpha = min(1.,.95*alpha_max)
        Delta = min(1.,.95*delta_max)

        # iteration
        x1 = x + Delta*delta_x
        mu =  mu + Alpha*delta_mu
        lam = lam + Alpha*delta_lam

        if la.norm(x1-x)<tol:
            break
        x = x1

    return x, c.T.dot(x)


def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
    data = np.loadtxt(filename)

    m = data.shape[0]
    n = data.shape[1] - 1
    c = np.zeros(3*m + 2*(n + 1))
    c[:m] = 1
    y = np.empty(2*m)
    y[::2] = -data[:, 0]
    y[1::2] = data[:, 0]
    x = data[:, 1:]

    A = np.ones((2*m, 3*m + 2*(n + 1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m+n] = -x
    A[1::2, m:m+n] = x
    A[::2, m+n:m+2*n] = x
    A[1::2, m+n:m+2*n] = -x
    A[::2, m+2*n] = -1
    A[1::2, m+2*n+1] = -1
    A[:, m+2*n+2:] = -np.eye(2*m, 2*m)

    sol = interiorPoint(A, y, c, niter=10)[0]
    print sol

    beta = sol[m:m+n] - sol[m+n:m+2*n]
    b = sol[m+2*n] - sol[m+2*n+1]

    slope, intercept = linregress(data[:,1], data[:,0])[:2]
    domain = np.linspace(0,10,200)
    plt.subplot(211)
    plt.plot(data[:,1],beta*data[:,1]+b)
    plt.plot(data[:,1],data[:,0], "k.")
    plt.title("Least Absolute Deviations")

    plt.subplot(212)
    plt.plot(domain, domain*slope + intercept, "g")
    plt.plot(data[:,1],data[:,0], "k.")
    plt.title("Least Squares")
    plt.show()

