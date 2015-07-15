"""
Author:

KMR (Kandori-Mailath-Rob) Model

"""
from __future__ import division
import numpy as np
import quantecon as qe

def kmr_markov_matrix(p, N, epsilon):
    """
    Generate the transition probability matrix for the KMR dynamics with
    two acitons.

    """
    epsilon=e
    senni_gyoretu=np.zeros((n+1, n+1))
    for i in range(n+1):
        if i/n > p and i!=n:
            senni_gyoretu[i][i+1]=(1-i/n)*(1-e/2)
            senni_gyoretu[i][i-1]=i/n*(e/2)
            senni_gyoretu[i][i]=1-((1-i/n)*(1-e/2))-(i/n*e/2)
        if i/n < p and i!=0:
            senni_gyoretu[i][i+1]=(1-i/n)*(e/2)
            senni_gyoretu[i][i-1]=i/n*(1-e/2)
            senni_gyoretu[i][i]=1-((1-i/n)*(e/2))-(i/n*(1-e/2))
        if i/n == p:    
            senni_gyoretu[i][i+1]=1/2-i/2/n
            senni_gyoretu[i][i-1]=i/2/n
            senni_gyoretu[i][i]=1/2
        if i==n:
            senni_gyoretu[i][i-1]=e/2
            senni_gyoretu[i][i]=1-e/2
        if i==0:
            senni_gyoretu[i][i+1]=e/2
            senni_gyoretu[i][i]=1-e/2
    return senni_gyoretu
    


class KMR(object):
    """
    Class representing the KMR dynamics with two actions.

    """
    def __init__(self, p, N, epsilon):
        P = kmr_markov_matrix(p, N, epsilon)
        self.mc = qe.MarkovChain(P)

    def simulate(self, ts_length, init=None, num_reps=None):
        """
        Simulate the dynamics.

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
        return self.mc.simulate(ts_length, init, num_reps)

    def compute_stationary_distribution(self):
        return self.mc.stationary_distributions[0]  