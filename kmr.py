"""
Author:

KMR (Kandori-Mailath-Rob) Model

"""
import numpy as np
import quantecon as qe


def kmr_markov_matrix(p, N, epsilon):
    """
    Generate the transition probability matrix for the KMR dynamics with
    two acitons.

    """
    # KMR の遷移確率行列を返す関数を書く
    P = np.empty((N+1, N+1))  # たとえばこんな感じで始めて P の要素を埋めていく


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
        # mc.stationary_distributions の戻り値は2次元配列．
        # 各行に定常分布が入っている (一般には複数)．
        # epsilon > 0 のときは唯一，epsilon == 0 のときは複数ありえる．
        # espilon > 0 のみを想定して唯一と決め打ちするか，
        # 0か正かで分岐するかは自分で決める．
        return self.mc.stationary_distributions[0]  # これは唯一と決め打ちの場合
