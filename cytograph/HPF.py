from typing import *
import time
import tempfile
import os
from subprocess import Popen
import numpy as np
import logging
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from scipy.special import gammaln, digamma, psi
from scipy.misc import logsumexp
from numba import jit, vectorize, float32, float64, int64
import math


@vectorize([float64(float64), float32(float32)], nopython=True)
def simple_digamma(x):
    r = 0

    while (x <= 5):
        r -= 1 / x
        x += 1

    f = 1 / (x * x)

    t = f * (-1 / 12.0 + f * (1 / 120.0 + f * (-1 / 252.0 + f * (1 / 240.0 + f * (-1 / 132.0 + f * (691 / 32760.0 + f * (-1 / 12.0 + f * 3617 / 8160.0)))))))

    return r + math.log(x) - 0.5 / x + t


@jit(float64[:, :](int64[:], int64[:], float64[:, :], float64[:, :], float64[:, :], float64[:, :]), parallel=False)
def calculate_phi(u, i, gamma_shape, gamma_rate, lambda_shape, lambda_rate):
    a = simple_digamma(gamma_shape[u, :])
    b = - np.log(gamma_rate[u, :])
    c = simple_digamma(lambda_shape[i, :])
    d = - np.log(lambda_rate[i, :])
    return a + b + c + d  


def make_nonzero(a: np.ndarray) -> np.ndarray:
    """
    Make the array nonzero in place, by replacing zeros with 1e-30

    Returns:
        a	The input array
    """
    a[a == 0.0] = 1e-30
    return a


class HPF:
    """
    Bayesian Hierarchical Poisson Factorization
    Implementation of https://arxiv.org/pdf/1311.1704.pdf
    """
    def __init__(self, k: int, a: float = 0.3, b: float = 0.3, c: float = 0.3, d: float = 0.3, max_iter: int = 1000, stop_interval: int = 10) -> None:
        self.k = k
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.max_iter = max_iter
        self.stop_interval = stop_interval

        self.beta: np.ndarray = None
        self.theta: np.ndarray = None

        self.log_likelihoods: List[float] = []

    def fit(self, X: sparse.coo_matrix) -> None:
        """
        Fit an HPF model to the data matrix

        Args:
            X      Data matrix, shape (n_users, n_items)

        Remarks:
            TODO check this
            After fitting, the factor matrices beta and theta are available as self.beta of shape
            (n_users, k) and self.theta of shape (k, n_items)

        PROFILING RESULTS of input 1500 samples, 1000 genes
        make_nonzero 2.1148e-04
        digamma normalization 1.5492e+00
            phi_digamma calculation 6.5351e-01
            phi_log calculation 5.3066e-01
            y_phi calculation 4.2811e-01
        theta update 1.3353e-01
        beta update 1.4134e-01
        loglik computation 1.0144e-01
        """
        if type(X) is not sparse.coo_matrix:
            raise TypeError("Input matrix must be in sparse.coo_matrix format")

        clock = Clock()  # PROFILING
        
        # Create local variables for convenience
        (n_users, n_items) = X.shape
        (a, b, c, d) = (self.a, self.b, self.c, self.d)
        k = self.k
        # u and i are indices of the nonzero entries; y are the values of those entries
        (u, i, y) = (X.row, X.col, X.data)

        # Initialize the variational parameters with priors
        kappa_shape = np.full(n_users, a) + np.random.uniform(0, 0.1, n_users)
        kappa_rate = np.full(n_users, b + k)
        gamma_shape = np.full((n_users, k), a) + np.random.uniform(0, 0.1, (n_users, k))
        gamma_rate = np.full((n_users, k), b) + np.random.uniform(0, 0.1, (n_users, k))

        tau_shape = np.full(n_items, c) + np.random.uniform(0, 0.1, n_items)
        tau_rate = np.full(n_items, d + k)
        lambda_shape = np.full((n_items, k), c) + np.random.uniform(0, 0.1, (n_items, k))
        lambda_rate = np.full((n_items, k), d) + np.random.uniform(0, 0.1, (n_items, k))
        
        self.log_likelihoods = []
        n_iter = 0
        while True:
            n_iter += 1
            make_nonzero(gamma_shape)
            make_nonzero(gamma_rate)
            make_nonzero(lambda_shape)
            make_nonzero(lambda_rate)
            logging.debug("make_nonzero %.4e" % clock.toc())  # PROFILING

            
            # Compute y * phi only for the nonzero values, which are indexed by u and i in the sparse matrix
            # phi is calculated on log scale from expectations of the gammas, hence the digamma and log terms
            # Shape of phi will be (nnz, k)
            # TODO: digamma function can be superslow depending imput parameters!!!
            clock.tic()  # PROFILING
            phi_digamma_part = simple_digamma(gamma_shape[u, :]) + simple_digamma(lambda_shape[i, :])
            logging.debug("phi_digamma calculation %.4e" % clock.toc())  # PROFILING
            clock.tic()  # PROFILING
            phi_log_part = - np.log(gamma_rate[u, :]) - np.log(lambda_rate[i, :])
            logging.debug("phi_log calculation %.4e" % clock.toc())  # PROFILING
            phi = phi_digamma_part + phi_log_part
            
            clock.tic()
            # Multiply y by phi normalized (in log space) along the k axis
            # TODO: this normalization is one of the slowest steps, could be accelerated using numba
            y_phi = y[:, None] * np.exp(phi - logsumexp(phi, axis=1)[:, None])
            logging.debug("y_phi calculation %.4e" % clock.toc())
            
            clock.tic()  # PROFILING
            # Upate the variational parameters corresponding to theta (the users)
            # Sum of y_phi over users, for each k
            y_phi_sum_u = np.zeros((n_users, k))
            for ix in range(k):
                y_phi_sum_u[:, ix] = sparse.coo_matrix((y_phi[:, ix], (u, i)), X.shape).sum(axis=1).A.T[0]
            gamma_shape = a + y_phi_sum_u
            gamma_rate = (kappa_shape / kappa_rate)[:, None] + (lambda_shape / lambda_rate).sum(axis=0)
            kappa_rate = b + (gamma_shape / gamma_rate).sum(axis=1)
            logging.debug("theta update %.4e" % clock.toc())  # PROFILING

            clock.tic()  # PROFILING
            # Upate the variational parameters corresponding to beta (the items)
            # Sum of y_phi over items, for each k
            y_phi_sum_i = np.zeros((n_items, k))
            for ix in range(k):
                y_phi_sum_i[:, ix] = sparse.coo_matrix((y_phi[:, ix], (u, i)), X.shape).sum(axis=0).A
            lambda_shape = c + y_phi_sum_i
            lambda_rate = (tau_shape / tau_rate)[:, None] + (gamma_shape / gamma_rate).sum(axis=0)
            tau_rate = d + (lambda_shape / lambda_rate).sum(axis=1)
            logging.debug("beta update %.4e" % clock.toc())  # PROFILING

            if n_iter % self.stop_interval == 0:
                clock.tic()  # PROFILING
                # Compute the log likelihood and assess convergence
                # Expectations
                egamma = make_nonzero(gamma_shape / gamma_rate)
                elambda = make_nonzero(lambda_shape / lambda_rate)
                # Sum over k for the expectations
                # This is really a dot product but we're only computing it for the nonzeros (indexed by u and i)
                s = (egamma[u] * elambda[i]).sum(axis=1)
                # We use gammaln to compute the log factorial, hence the "y + 1"
                log_likelihood = np.sum(y * np.log(s) - s - gammaln(y + 1))
                self.log_likelihoods.append(log_likelihood)
                logging.debug("loglik computation %.4e" % clock.toc())  # PROFILING
                # Time to stop?
                if n_iter >= self.max_iter:
                    break

                # Check for convergence
                # TODO: allow for small fluctuations?
                if len(self.log_likelihoods) > 1:
                    clock.tic()  # PROFILING
                    prev_ll = self.log_likelihoods[-2]
                    diff = abs((log_likelihood - prev_ll) / prev_ll)
                    logging.info(f"Iteration {n_iter}, ll = {log_likelihood}, diff = {diff}")
                    logging.debug("convergence check %.4e" % clock.toc())
                    if diff < 0.000001:
                        break
        # End of the main fitting loop
        # Compute beta and theta, which are given by the expectations, i.e. shape / rate
        self.beta = lambda_shape / lambda_rate
        self.theta = gamma_shape / gamma_rate


class Clock:
    def __init__(self) -> None:
        self.internal = 0.

    def tic(self) -> None:
        self.internal = time.time()
    
    def toc(self) -> float:
        return time.time() - self.internal

    def reset(self) -> None:
        self.internal = 0
