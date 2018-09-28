# -*- coding: utf-8 -*-
# simplex.py
"""Simplex.
Bailey Smith
January 19 2017
"""

import numpy as np

class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        maximize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """

    def __init__(self, c, A, b):
        """

        Parameters:
            c (1xn ndarray): The coefficients of the linear objective function.
            A (mxn ndarray): The constraint coefficients matrix.
            b (1xm ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        #Check that every entry in b is >= 0
        newb = b[b<=0]
        if np.shape(newb)[0] != 0:
            raise ValueError("The given system is infeasible at the origin")

        m,n = np.shape(A)
        L1 = []
        L2 = []
        for i in xrange(n+m):
            if i < n:
                L2.append(i)
            else:
                L1.append(i)
        self.L = L1 + L2
        self.A = A
        self.c = c
        self.b = b
        self.T = np.array([])
        self.column = 1
        self.row = np.inf

    def tableau(self):
        '''Create the initial tableau as a NumPy array'''
        m,n = np.shape(self.A)
        I = np.eye(m)
        zeros = np.zeros(m)

        A_new = np.hstack((self.A, I))
        c_new = -np.append(self.c, zeros)
        top = np.append([0],np.append(c_new,[1]))
        bottom = np.hstack((np.hstack((self.b.reshape(-1,1),A_new)), np.zeros((m,1))))
        self.T = np.vstack((top,bottom))


    def find_pivot(self):
        '''Determine the pivot row and pivot column according to Bland’s Rule'''
        m,n = np.shape(self.A)
        for i in xrange(n):
            if self.T[0,i+1] < 0:
                self.column = i+1
                break
            elif i == n-1:
                return False
        ratio = np.inf

        for i in xrange(1,np.shape(self.T)[0]):
            if self.T[i,self.column] != 0:
                new_ratio = float(abs(self.T[i,0])/float(self.T[i,self.column]))
                if new_ratio < ratio and new_ratio > 0:
                    self.row = i
                    ratio = new_ratio

        if self.row == np.inf or ratio == np.inf:
            raise ValueError("Problem is unbounded")

        return True

    def check(self):
        '''Check for unboundedness and performs a single pivot operation from
            start to completion. If the problem is unbounded, raise a ValueError'''
        m,n = np.shape(self.T)
        bound = self.T[1:,self.column][self.T[1:,self.column]>0]
        if np.shape(bound)[0] == 0:
            raise ValueError("Problem is unbounded")
        else:
            #swap pivoting indices
            a, b = self.L.index(self.column-1), self.row-1
            self.L[b], self.L[a] = self.L[a], self.L[b]

            #Divide pivot row by pivot entry
            self.T[self.row,:] = self.T[self.row,:]/self.T[self.row,self.column]
            for i in xrange(m):
                if i != self.row:
                    self.T[i,:] += (self.T[self.row,:] * -self.T[i,self.column])

    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The maximum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        #
        self.tableau()
        m,n = np.shape(self.A)
        while self.find_pivot():
            self.check()

        basic = {}
        nonbasic = {}
        for i in xrange(m):
            basic.update({self.L[i]:self.T[i+1,0]})
        for j in xrange(n):
            nonbasic.update({self.L[m+j]: 0})
        return (self.T[0,0], basic, nonbasic)


def application_test(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        The minimizer of the problem (as an array).
        optimal value should be around 7453
    """
    data = np.load(filename)
    m,n = np.shape(data['A'])
    I = np.eye(n)
    solver = SimplexSolver(data['p'], np.vstack((data['A'],I)), np.hstack((data['m'],data['d'])))
    answer = solver.solve()
    print "Optimal Value:", answer[0]
    return answer[1].values()



if __name__ == '__main__':

    print 'example'
    c = np.array([3., 2])
    b = np.array([2., 5, 7])
    A = np.array([[1., -1], [3, 1], [4, 3]])
    # #
    solver = SimplexSolver(c, A, b)
    solver.tableau()
    print solver.solve()


    print application_test()
