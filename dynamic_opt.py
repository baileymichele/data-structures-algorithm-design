# dynamic_opt.py
"""Dynamic Optimization.
Bailey Smith
April 13 2017
"""

import numpy as np
from matplotlib import pyplot as plt


def graph_policy(policy, b, u):
    """Plot the utility gained over time.
    Return the total utility gained with the policy given.

    Parameters:
        policy (ndarray): Policy vector.
        b (float): Discount factor. 0 < beta < 1.
        u (function): Utility function.

    Returns:
        total_utility (float): Total utility gained from the policy given.
    """

    if np.sum(policy) != 1.0:
        raise ValueError("Policy vector should sum to 1")

    pol_beta = np.array([(b**i)*u(policy[i]) for i in xrange(len(policy))],dtype=float)
    cummulative = np.cumsum(pol_beta)
    plt.plot([i for i in xrange(len(policy))], cummulative, 'r',label='Calculated Policy')
    plt.title("Graphing the Optimal Policy")
    plt.legend(loc='upper left')
    plt.show()

    return cummulative[-1]

def consumption(N, u=lambda x: np.sqrt(x)):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        C ((N+1,N+1) ndarray): Consumption matrix.
    """
    C = np.zeros((N+1,N+1))
    w = np.linspace(0,1,N+1)
    for i in xrange(N+1):
        C[i:,i] = u(w[0:N-i+1])
    return C


def eat_cake(T, N, B, u=lambda x: np.sqrt(x)):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
    C = consumption(N, u)
    w = np.linspace(0,1,N+1)
    A = np.zeros((N+1,T+1))
    CV = np.zeros((N+1,N+1))
    A[:,T] = u(w)

    P = np.zeros((N+1,T+1))
    P[:,T] = w

    for k in xrange(T):
        for i in xrange(N+1):
            CV[:,i] = C[:,i] + B*A[i,T-k]
        for i in xrange(N+1):
            j = np.argmax(np.tril(CV, k=0)[i,:])
            P[i,T-k-1] = w[i] - w[j]
        A[:,T-k-1] = np.max(np.tril(CV, k=0),axis=1)
    return A, P



def find_policy(T, N, B, u=lambda x: np.sqrt(x)):
    """Find the most optimal route to take assuming that we start with all of
    the pieces. Show a graph of the optimal policy using graph_policy().

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        maximum_utility (float): The total utility gained from the
            optimal policy.
        optimal_policy ((N,) nd array): The matrix describing the optimal
            percentage to consume at each time.
    """
    A, P = eat_cake(T, N, B, u)
    optimal_policy = np.zeros(T+1)
    percent_left = 1.
    for t in xrange(T+1):
        row = percent_left*N
        optimal_policy[t] = P[int(row),t]
        percent_left -= P[int(row),t]
    total_utility = graph_policy(optimal_policy, B, u)
    return total_utility, optimal_policy

