# scipy_optimize_intro.py
"""Optimization with Scipy
Bailey Smith
December 28 2016
"""
import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt
from blackbox_function import blackbox
from mpl_toolkits.mplot3d import axes3d

def min_rosenbrock():
    """Use the minimize() function in the scipy.optimize package to find the
    minimum of the Rosenbrock function (scipy.optimize.rosen) using the
    following methods:
        Nelder-Mead
        CG
        BFGS
    Use x0 = np.array([4., -2.5]) for the initial guess for each test.

    For each method, print whether it converged, and if so, print how many
        iterations it took.
    """
    x_0 = np.array([4., -2.5])
    Nelder_Mead = opt.minimize(opt.rosen, x_0, method='Nelder-Mead')
    CG = opt.minimize(opt.rosen, x_0, method='CG', jac=opt.rosen_der)
    BFGS = opt.minimize(opt.rosen, x_0, method='BFGS', jac=opt.rosen_der)

    print "Nelder-Mead Method:\n", "  Converge:", Nelder_Mead['success']
    if Nelder_Mead['success'] is True:
        print "  ", Nelder_Mead['nit']

    print "\nCG Method:\n", "  Converge:", CG['success']
    if CG['success'] is True:
        print "  ", CG['nit']

    print "\nBFGS Method:\n", "  Converge:", BFGS['success']
    if BFGS['success'] is True:
        print "  ", BFGS['nit']

    print "ANSWER", BFGS['x']


def help2(x_0, mymethod='Nelder-Mead'):
    result = opt.minimize(blackbox, x_0, method=mymethod)
    print mymethod, ":\n", "  Converge:", result['success']
    if result['success'] is True:
        print "  ", result['nit']

def min_blackbox():
    """Minimizes the function blackbox() in the blackbox_function module,
    selecting the appropriate method of scipy.optimize.minimize() for this
    problem. 

    The blackbox() function returns the length of a piecewise-linear curve
    between two fixed points: the origin, and the point (40,30).
    It accepts a one-dimensional ndarray} of length m of y-values, where m
    is the number of points of the piecewise curve excluding endpoints.
    These points are spaced evenly along the x-axis, so only the y-values
    of each point are passed into blackbox().

    Once selected a method, selects an initial point with the
    provided code.

    Plots initial curve and minimizing curve together on the same
    plot, including endpoints. Note that this will require padding your
    array of internal y-values with the y-values of the endpoints, so
    that you plot a total of 20 points for each curve.
    """
    y_initial = 30*np.random.random_sample(18)
    #Pad array with y-values of the endpoints
    y_initial = np.insert(y_initial,0,0)
    y_initial = np.append(y_initial,30)

    results = opt.minimize(blackbox, y_initial, method='Powell')
    print results
    domain = np.linspace(0,40,20)

    plt.plot(domain,y_initial, label="Initial Curve")
    plt.plot(domain,results['x'], label="Minimizing Curve")
    plt.legend(loc='lower right')


    plt.show()

#Multimin function for prob 3
def multimin(x):
    r = np.sqrt((x[0]+1)**2 + x[1]**2)
    return r**2 *(1+ np.sin(4*r)**2)

def basin_hopping():
    """
    Explores the use of basing hopping, limitations.
    Local and global minimas
    """
    x0 = np.array([-2, -2])
    result1 = opt.basinhopping(multimin, x0, stepsize=0.5, minimizer_kwargs={'method':'nelder-mead'})
    result2 = opt.basinhopping(multimin, x0, stepsize=0.2, minimizer_kwargs={'method':'nelder-mead'})

    xdomain = np.linspace(-3.5,1.5,70)
    ydomain = np.linspace(-2.5,2.5,60)
    X,Y = np.meshgrid(xdomain,ydomain)
    Z = multimin((X,Y))
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_wireframe(X, Y, Z, linewidth=.5, color='c')

    #Plot the initial point and minima by adapting the following line
    ax1.scatter(result1['x'][0], result1['x'][1], multimin(result1['x']))
    ax1.scatter(result2['x'][0], result2['x'][1], multimin(result2['x']))
    plt.show()
    print "The algorithm doesn't find the global minimum with stepsize=0.2 because it gets stuck in the local basins"
    #Explain why doesnt work with stepsize=0.2
    #return minimum value? ie z??
    return multimin(result1['x'])

def func4(x):
    return [-x[0] + x[1] + x[2], 1 + x[0]**3 - x[1]**2 + x[2]**3, -2 - x[0]**2 + x[1]**2 + x[2]**2]

def jac(x):
    return np.array([[-1,1,1],[3*x[0]**2, -2*x[1], 3*x[2]**2],[-2*x[0], 2*x[1], 2*x[2]]])

def root_finding():
    """Find the roots of the function
               [       -x + y + z     ]
    f(x,y,z) = [  1 + x^3 - y^2 + z^3 ]
               [ -2 - x^2 + y^2 + z^2 ]

    Returns the values of x,y,z as an array.
    """
    sol = opt.root(func4, [0, 0, 0], jac=jac, method='hybr')
    # print sol.x
    # print func(sol.x)
    return sol.x



def curve_fitting():
    """Uses the scipy.optimize.curve_fit() function to fit a curve to
    the data found in `convection.npy`. The first column of this file is R,
    the Rayleigh number, and the second column is Nu, the Nusselt number.

    The fitting parameters should be c and beta, as given in the convection
    equations.

    Plot the data from `convection.npy` and the curve generated by curve_fit.
    Return the values c and beta as an array.
    """
    R, v = np.load("convection.npy").T
    def func(R, c, beta):
        return c*R**beta

    popt, pcov = opt.curve_fit(func,R[4:],v[4:])

    plt.loglog(R, v, 'k.', basex=10, basey=10, lw=2, ms=6, label='Data')
    plt.plot(R[4:], func(R[4:],popt[0],popt[1]), label='Curve')
    plt.legend(loc="lower right")
    plt.show()

    return popt

if __name__ == '__main__':
    print min_rosenbrock()

    ### Choosing best Method ###
    x_0 = 30*np.random.random_sample(18)
    print help2(x_0)
    print help2(x_0,'Powell')
    print help2(x_0,'CG')
    print help2(x_0,'BFGS')
    print help2(x_0,'L-BFGS-B')
    print help2(x_0,'TNC')
    print help2(x_0,'COBYLA')
    print help2(x_0,'SLSQP')

    min_blackbox()

    print basin_hopping()

    print root_finding()

    print curve_fitting()
