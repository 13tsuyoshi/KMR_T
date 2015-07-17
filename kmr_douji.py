"""
Author:

KMR (Kandori-Mailath-Rob) Model

"""
from __future__ import division
import numpy as np
import quantecon as qe
import random
from scipy.stats import binom

def kmr_markov_matrix(p, N, epsilon):
    """
    Generate the transition probability matrix for the KMR dynamics with
    two acitons.

    """
    P=np.zeros((N+1, N+1))
    for n in range(N+1):
        if n/N > p:
            binomial_distribution=[binom.pmf(i,N,1-epsilon/2) for i in range(N+1)]
            for i in range(N+1):
                P[n][i]=binomial_distribution[i]
        if n/N < p:
            binomial_distribution=[binom.pmf(i,N,epsilon/2) for i in range(N+1)]
            for i in range(N+1):
                P[n][i]=binomial_distribution[i]
        if n/N == p:
            binomial_distribution=[binom.pmf(i,N,1/2) for i in range(N+1)]
            for i in range(N+1):
                P[n][i]=binomial_distribution[i]
    return P


class KMR(object):
    """
    Class representing the KMR dynamics with two actions.

    """
    def __init__(self, p, N, epsilon):
        P = kmr_markov_matrix(p, N, epsilon)
        self.mc = qe.MarkovChain(P)

    def simulate(self, ts_length, init=None, num_reps=None):
        """
        Simulate the dynamics

        Parameters
        ----------
        ts_length : scalar(int)
            Length of each simulation.

        init : scalar(int) or array_like(int, ndim=1),
               optional(default=None)
            Initial state(s). If None, the initial state is randomly
            drawn.

        num_reps : scalar(int), optional(default=None)
            Number of simulations. Relevant only when init is a scalar
            or None.

        Returns
        -------
        X : ndarray(int, ndim=1 or 2)
            Array containing the sample path(s), of shape (ts_length,)
            if init is a scalar (integer) or None and num_reps is None;
            of shape (k, ts_length) otherwise, where k = len(init) if
            init is an array_like, otherwise k = num_reps.

        """
        if init is None:
            #init = random.randint(0,len(P)-1)
            init = 0
        return self.mc.simulate(init=init, sample_size=ts_length)

    def compute_stationary_distribution(self):
        return self.mc.stationary_distributions[0]  