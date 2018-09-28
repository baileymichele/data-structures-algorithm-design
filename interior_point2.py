"""Interior Point II (Quadratic Optimization)."""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from cvxopt import matrix, solvers
from scipy.sparse import spdiags
from mpl_toolkits.mplot3d import axes3d


def startingPoint(Q, c, A, b, guess):
    """
    Obtain an appropriate initial point for solving the QP
    .5 x^T Qx + x^T c s.t. Ax >= b.
    Inputs:
        Q -- symmetric positive semidefinite matrix shape (n,n)
        c -- array of length n
        A -- constraint matrix shape (m,n)
        b -- array of length m
        guess -- a tuple of arrays (x, y, mu) of lengths n, m, and m, resp.
    Returns:
        a tuple of arrays (x0, y0, l0) of lengths n, m, and m, resp.
    """
    m,n = A.shape
    x0, y0, l0 = guess

    # Initialize linear system
    N = np.zeros((n+m+m, n+m+m))
    N[:n,:n] = Q
    N[:n, n+m:] = -A.T
    N[n:n+m, :n] = A
    N[n:n+m, n:n+m] = -np.eye(m)
    N[n+m:, n:n+m] = np.diag(l0)
    N[n+m:, n+m:] = np.diag(y0)
    rhs = np.empty(n+m+m)
    rhs[:n] = -(Q.dot(x0) - A.T.dot(l0)+c)
    rhs[n:n+m] = -(A.dot(x0) - y0 - b)
    rhs[n+m:] = -(y0*l0)

    sol = la.solve(N, rhs)
    dx = sol[:n]
    dy = sol[n:n+m]
    dl = sol[n+m:]

    y0 = np.maximum(1, np.abs(y0 + dy))
    l0 = np.maximum(1, np.abs(l0+dl))

    return x0, y0, l0


def qInteriorPoint(Q, c, A, b, guess, niter=20, tol=1e-16, verbose=False):
    """Solve the Quadratic program min .5 x^T Q x +  c^T x, Ax >= b
    using an Interior Point method.

    Parameters:
        Q ((n,n) ndarray): Positive semidefinite objective matrix.
        c ((n, ) ndarray): linear objective vector.
        A ((m,n) ndarray): Inequality constraint matrix.
        b ((m, ) ndarray): Inequality constraint vector.
        guess (3-tuple of arrays of lengths n, m, and m): Initial guesses for
            the solution x and lagrange multipliers y and eta, respectively.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    # Define F (prob 1)
    def F(x,y, mu):
        one = Q.dot(x) - A.T.dot(mu) + c
        two = A.dot(x) - y - b
        three = np.diag(y).dot(np.diag(mu)).dot(np.ones_like(y))
        return np.hstack((np.hstack((one,two)),three))
    # initialize
    m,n = np.shape(A)
    x, y, mu = startingPoint(Q, c, A, b, guess)
    y = A.dot(x) - b
    sigma = 1./10
    tau = .95

    for i in xrange(niter):
        V = y.T.dot(mu)/float(m)
        # print V
        M = np.diag(mu)
        X = np.diag(x)
        Y = np.diag(y)

        # Define DF
        one = np.vstack((Q, np.vstack((A, np.zeros((m,n))))))
        two = np.vstack((np.zeros((n,m)), np.vstack((-np.eye(m), np.diag(mu)))))
        three = np.vstack((-A.T,np.vstack((np.zeros((m,m)),np.diag(y)))))
        Df = np.hstack((one,np.hstack((two,three))))

        # Determine direction (21.2 prob 2)
        e_ = np.ones_like(mu)
        plus = np.hstack((np.hstack((np.zeros_like(x),np.zeros_like(y))),sigma*V*(e_)))
        s_direction = la.solve(Df,-1*F(x,y,mu) + plus)

        delta_x = s_direction[:n]
        delta_y = s_direction[n:n+m]
        delta_mu = s_direction[n+m:]

        # determine step size
        neg = delta_mu < 0
        negy = delta_y < 0

        if sum(neg) > 0:
            alpha_max = min(1, np.min((-1*mu/delta_mu)[neg]))
        elif sum(neg) == 0:
            alpha_max = 1

        if sum(negy) > 0:
            delta_max = min(1, np.min((-1*y/delta_y)[negy]))
        elif sum(negy) == 0:
            delta_max = 1


        Beta = min(1., tau*alpha_max)
        Delta = min(1., tau*delta_max)
        Alpha = min(Beta, Delta)

        # iteration
        x1 = x + Delta*delta_x
        mu +=  Alpha*delta_mu
        y += Alpha*delta_y

        if la.norm(x1-x)<tol:
            break
        x = x1
    return x, c.T.dot(x)


def laplacian(n):
    """Construct the discrete Dirichlet energy matrix H for an n x n grid."""
    data = -1*np.ones((5, n**2))
    data[2,:] = 4
    data[1, n-1::n] = 0
    data[3, ::n] = 0
    diags = np.array([-n, -1, 0, 1, n])
    return spdiags(data, diags, n**2, n**2).toarray()


def circus(n=15):
    """Solve the circus tent problem for grid size length 'n'.
    Display the resulting figure.
    """
    H = laplacian(n)
    #c is a vector whose entries are all equal to  -(n - 1)^-2, and n is the side length of the grid
    c = np.ones(n**2)#n?
    c *= -(n-1)**(-2)
    A = np.eye(n**2)

    # Create the tent pole configuration.
    L = np.zeros((n,n))
    L[n//2-1:n//2+1,n//2-1:n//2+1] = .5
    m = [n//6-1, n//6, int(5*(n/6.))-1, int(5*(n/6.))]
    mask1, mask2 = np.meshgrid(m, m)
    L[mask1, mask2] = .3
    L = L.ravel()

    # Set initial guesses.
    x = np.ones((n,n)).ravel()
    y = np.ones(n**2)
    mu = np.ones(n**2)

    # Calculate the solution.
    z = qInteriorPoint(H, c, A, L, (x,y,mu))[0].reshape((n,n))

    # Plot the solution.
    domain = np.arange(n)
    X, Y = np.meshgrid(domain, domain)
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, z, rstride=1, cstride=1, color='r')
    plt.show()

def portfolio(filename="portfolio.txt"):
    """Markowitz Portfolio Optimization

    Parameters:
        filename (str): The name of the portfolio data file.

    Returns:
        (ndarray) The optimal portfolio with short selling.
        (ndarray) The optimal portfolio without short selling.
    """
    data = np.loadtxt(filename)
    data = data[:,1:]
    n,m = np.shape(data)
    R = 1.13

    Q = np.cov(data.T)
    p = np.zeros(m)

    G = -1*np.eye(m)
    h = np.zeros(m)

    mu = np.mean(data, axis=0)
    A = np.vstack((np.ones(m),mu))
    b = np.array([1,R])
    # print Q,p
    sol1 = solvers.qp(matrix(Q), matrix(p), matrix(G), matrix(h), A=matrix(A), b=matrix(b))
    sol2 = solvers.qp(matrix(Q), matrix(p), A=matrix(A), b=matrix(b))
    return np.ravel(sol2['x']), np.ravel(sol1['x'])

if __name__ == '__main__':
    q = np.array([[1,-1],[-1,2]])
    c_ = np.array([-2,-6])
    A_ = np.array([[-1,-1],[1,-2],[-2,-1],[1,0],[0,1]])
    b_ = np.array([-2,-2,-3,0,0])
    x_ = np.array([.5,.5])
    y_ = np.ones(A_.shape[0])
    mu_ = np.ones(A_.shape[0])
    point, value = qInteriorPoint(q, c_, A_, b_, (x_,y_,mu_))
    print point
    print value

