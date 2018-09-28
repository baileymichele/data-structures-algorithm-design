# -*- coding: utf-8 -*-
# markov_chains.py
"""Markov Chains.
Bailey Smith
October 27 2016

Use of Markov Chains to simulate a simplified way of predicting
weather. It also implements a class that uses Markov Chains to simulate English
based on a given text file.
"""
import numpy as np
from random import random
import scipy.sparse as sp
from scipy import linalg as la

def random_markov(n):
    """Create and return a transition matrix for a random Markov chain with
    'n' states. This should be stored as an nxn NumPy array.
    """
    #create nxn random matrix, Normalize columns
    A = np.random.rand(n,n)
    for i in xrange(n):
        total = sum(A[:,i])
        A[:,i] = A[:,i]/float(total)
    return A

def forecast(days):
    """Forecast tomorrow's weather given that today is hot."""
    transition = np.array([[0.7, 0.6], [0.3, 0.4]])
    # Sample from a binomial distribution to choose a new state.
    predictions = []
    current = 0
    for i in xrange(days):
        if current == 1:
            prob = transition[1, 1]
        elif current == 0:
            prob = transition[1, 0]
        predictions.append(np.random.binomial(1, prob))
        current = predictions[i]
    return predictions

def four_state_forecast(days):
    """Run a simulation for the weather over the specified number of days,
    with mild as the starting state, using the four-state Markov chain.
    Return a list containing the day-by-day results, not including the
    starting day.

    Examples:
        >>> four_state_forecast(3)
        [0, 1, 3]
        >>> four_state_forecast(5)
        [2, 1, 2, 1, 1]
    """
    transition = np.array([[0.5, 0.3, 0.1, 0], [0.3, 0.3, 0.3, 0.3], [0.2, 0.3, 0.4, 0.5], [0, 0.1, 0.2, 0.2]])
    current = 1
    predictions = []
    for i in xrange(days):
        prob = transition[:,current]
        new = np.random.multinomial(1, prob)
        j = np.where(new == 1)#Finds the index where there is a 1 which corresponds to the current state
        current = int(j[0])
        predictions.append(current)
    return predictions

def steady_state(A, tol=1e-12, N=40):
    """Compute the steady state of the transition matrix A.

    Inputs:
        A ((n,n) ndarray): A column-stochastic transition matrix.
        tol (float): The convergence tolerance.
        N (int): The maximum number of iterations to compute.

    Raises:
        ValueError: if the iteration does not converge within N steps.

    Returns:
        x ((n,) ndarray): The steady state distribution vector of A.
    """
    n = np.shape(A)[0]
    i = 0
    e = 1
    xprev = np.random.rand(n,1)
    total = sum(xprev[:,0])
    xprev = xprev/float(total)#generate random state distribution vector
    xi = np.dot(A,xprev)
    while i <= N and e >= tol:
        i += 1
        if i > N:
            raise ValueError("Does Not Converge")
            break
        xi = np.dot(A, xprev)
        e = la.norm(xprev-xi, ord=np.inf)#calculate norm
        xprev = xi
    return xi


class SentenceGenerator(object):
    """Markov chain creator for simulating bad English.

    Attributes:
        (what attributes do you need to keep track of?)

    Example:
        >>> yoda = SentenceGenerator("Yoda.txt")
        >>> print yoda.babble()
        The dark side of loss is a path as one with you.
    """

    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line. Use sparse matrices to ensure the proccess
        will work for larger training sets.
        """
        self.filename = filename
        wordcount = set()
        lines = []
        states = ["$tart"]
        with open(filename, 'r') as myfile:
            for line in myfile:
                lines.append(line)
                for word in line.split():
                    if word not in wordcount:
                        states.append(word)
                    wordcount.add(word)#WORDS SEPARATED: does not keep duplicates
        states.append("$top")
        matrix = np.zeros((len(wordcount)+2,len(wordcount)+2))#Want a matrix the size of unique words +2
        matrix[len(wordcount)+1,len(wordcount)+1] = 1#Stop state transitions to self...index of last entry is 1 less than size

        for line in lines:#Each line is a sentence
            words = line.split()#Separate the words
            for i in xrange(len(words)):#For each word update the transition matrix
                if i == 0:#if it is the 1st word of the sentence
                    matrix[states.index(words[i]),0] += 1#Add 1 to start state
                elif i == len(words)-1:#if it is the last word in the sentence
                    matrix[states.index(words[i]),states.index(words[i-1])] += 1#Add 1 to prev word transition
                    matrix[len(wordcount)+1, states.index(words[i])] += 1#Add 1 to stop state
                else:
                    matrix[states.index(words[i]),states.index(words[i-1])] += 1#Add 1 to prev word transition
        '''Normalize columns'''
        for i in xrange(len(wordcount)+2):
            total = sum(matrix[:,i])
            if total != 0:
                matrix[:,i] = matrix[:,i]/float(total)

        self.states = states#Save the states of the transition matrix
        self.transition = matrix

    def babble(self):
        """Begin at the start sate and use the strategy from
        four_state_forecast() to transition through the Markov chain.
        Keep track of the path through the chain and the corresponding words.
        When the stop state is reached, stop transitioning and terminate the
        sentence. Return the resulting sentence as a single string.
        """
        current = 0
        predictions = []
        my_string = ""

        while current != np.shape(self.transition)[0]-1:#Transition through Markov chain until stop state
            prob = self.transition[:,current]#Probability is assigned the Start state transition
            new = np.random.multinomial(1, prob)#Takes 1 draw from multinomial with Probability of current transition state
            j = np.where(new == 1)#Finds the index where there is a 1 which corresponds to the current state
            current = int(j[0])
            predictions.append(current)#List of state at each point

        for i in predictions:#for each state
            if i < np.shape(self.transition)[0]-1:
                my_string += self.states[i] + " "
        return my_string


