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
import numexpr


def fast_log(x: np.ndarray) -> np.ndarray:
    return numexpr.evaluate('log(x)')


def fast_logprod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return numexpr.evaluate("log(a) + log(b)")


def numexpr_logsumexp(x: np.ndarray, axis: int=None) -> np.ndarray:
    if axis is None:
        return fast_log(numexpr.evaluate("sum(exp(x))"))
    else:
        return fast_log(numexpr.evaluate("sum(exp(x), axis=%i)" % axis))


def y_phi_calculation(y: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return numexpr.evaluate("y * exp(phi - logsumexp)",
                            local_dict={"y": y[:, None],
                                        "logsumexp": numexpr_logsumexp(phi, axis=1)[:, None],
                                        "phi": phi})


def update_x_r(x: np.ndarray, r: np.ndarray) -> None:
    for i in range(5):
        numexpr.evaluate('where(x<=5, r-1/x, r)', out=r)
        numexpr.evaluate('where(x<=5, x+1, x)', out=x)


def numexpr_digamma(a: np.ndarray) -> np.ndarray:
    """See https://en.wikipedia.org/wiki/Digamma_function#Computation_and_approximation
    and
    https://github.com/probml/pmtksupport/blob/master/GPstuff-2.0/dist/winCsource/digamma1.c
    or
    https://gist.github.com/miksu/223d81add9df8f878d75d39caa42873f
    """
    x = np.array(a, dtype="float64")  # to use float32 and produce speedup you would need to avoid casting later on
    r = np.zeros_like(x)
    update_x_r(x, r)
    crazy_expr = "r + log(x) - 1/(2*x) + (1/(x*x))*(-1/12.0 + (1/(x*x))*(1/120 + (1/(x*x))*(-1/252 + (1/(x*x))*(1/240 + (1/(x*x))*(-1/132 + (1/(x*x))*(691/32760 + (1/(x*x))*(-1/12 + (1/(x*x))*3617/8160)))))))"
    numexpr.evaluate(crazy_expr, out=x)  # casting="same_kind"
    return x


@jit(nopython=True)  # "float64[:,:](float64[:,:], int32[:], int32[:], int64, Tuple((int64,int64)), int64)",
def special_concatenate(y_phi, u, i, k, Xshape, axis):
    """Currently not used!!! It passes np.allclose but for some reason final result of HPF are different
    there must be some strange corner case but I cannot see it seems correct
    """
    if axis == 1:
        y_phi_sum = np.zeros((Xshape[0], k), dtype=np.float64)
        for ix in range(k):
            for n in range(len(u)):
                y_phi_sum[u[n], ix] += y_phi[n, ix]
    elif axis == 0:
        y_phi_sum = np.zeros((Xshape[1], k), dtype=np.float64)
        for ix in range(k):
            for n in range(len(i)):
                y_phi_sum[i[n], ix] += y_phi[n, ix]
    return y_phi_sum


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
    def __init__(self, k: int, a: float = 0.3, b: float = 0.3, c: float = 0.3, d: float = 0.3, max_iter: int = 1000, stop_interval: int = 10, stop_at_ll: float = 0.000001) -> None:
        self.k = k
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.max_iter = max_iter
        self.stop_interval = stop_interval
        self.stop_at_ll = stop_at_ll

        self.beta: np.ndarray = None
        self.theta: np.ndarray = None
        self.log_likelihoods: List[float] = []  # List of log likelihood for each stop_interval

        self._tau_rate: np.ndarray = None
        self._tau_shape: np.ndarray = None
        self._lambda_rate: np.ndarray = None
        self._lambda_shape: np.ndarray = None

    def fit(self, X: sparse.coo_matrix, n_threads: int=None) -> Any:
        """
        Fit an HPF model to the data matrix

        Args:
            X      Data matrix, shape (n_users, n_items)

        Remarks:
            After fitting, the factor matrices beta and theta are available as self.beta of shape
            (n_items, k) and self.theta of shape (n_users, k)
        """
        if type(X) is not sparse.coo_matrix:
            raise TypeError("Input matrix must be in sparse.coo_matrix format")

        (beta, theta) = self._fit(X, n_threads=n_threads)
        self.beta = beta
        self.theta = theta
        return self

    def _fit(self, X: sparse.coo_matrix, beta_precomputed: bool = False, n_threads: int=None) -> Tuple[np.ndarray, np.ndarray]:

        if n_threads is not None:
            previous_setting = numexpr.set_num_threads(int(n_threads))
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

        if beta_precomputed:
            tau_shape = self._tau_shape
            tau_rate = self._tau_rate
            lambda_shape = self._lambda_shape
            lambda_rate = self._lambda_rate
        else:
            tau_shape = np.full(n_items, c) + np.random.uniform(0, 0.1, n_items)
            tau_rate = np.full(n_items, d + k)
            lambda_shape = np.full((n_items, k), c) + np.random.uniform(0, 0.1, (n_items, k))
            lambda_rate = np.full((n_items, k), d) + np.random.uniform(0, 0.1, (n_items, k))

        self.log_likelihoods = []
        n_iter = 0
        # clock = Clock()
        while True:
            n_iter += 1
            # clock.tic()
            make_nonzero(gamma_shape)
            make_nonzero(gamma_rate)
            make_nonzero(lambda_shape)
            make_nonzero(lambda_rate)
            # logging.debug("make_nonzero %.4e" % clock.toc())

            # Compute y * phi only for the nonzero values, which are indexed by u and i in the sparse matrix
            # phi is calculated on log scale from expectations of the gammas, hence the digamma and log terms
            # Shape of phi will be (nnz, k)
            # clock.tic()
            phi = (numexpr_digamma(gamma_shape) - fast_log(gamma_rate))[u, :] + (numexpr_digamma(lambda_shape) - fast_log(lambda_rate))[i, :]
            # phi = (digamma(gamma_shape) - np.log(gamma_rate))[u, :] + (digamma(lambda_shape) - np.log(lambda_rate))[i, :]
            # logging.debug("phi_calc %.4e" % clock.toc())
            # Multiply y by phi normalized (in log space) along the k axis
            # clock.tic()
            y_phi = y_phi_calculation(y, phi)
            # logging.debug("y_phi_calc %.4e" % clock.toc())

            # clock.tic()
            # Upate the variational parameters corresponding to theta (the users)
            # Sum of y_phi over users, for each k
            y_phi_sum_u = np.zeros((n_users, k))
            for ix in range(k):
                y_phi_sum_u[:, ix] = sparse.coo_matrix((y_phi[:, ix], (u, i)), X.shape).sum(axis=1).A.T[0]
            # logging.debug("theta_update_p1 %.4e" % clock.toc())
            # clock.tic()
            gamma_shape = a + y_phi_sum_u
            gamma_rate = (kappa_shape / kappa_rate)[:, None] + (lambda_shape / lambda_rate).sum(axis=0)
            kappa_rate = b + (gamma_shape / gamma_rate).sum(axis=1)
            # logging.debug("theta_update_p1 %.4e" % clock.toc())

            if not beta_precomputed:
                # clock.tic()
                # Upate the variational parameters corresponding to beta (the items)
                # Sum of y_phi over items, for each k
                y_phi_sum_i = np.zeros((n_items, k))
                for ix in range(k):
                    y_phi_sum_i[:, ix] = sparse.coo_matrix((y_phi[:, ix], (u, i)), X.shape).sum(axis=0).A
                # logging.debug("beta_update_p1 %.4e" % clock.toc())
                # clock.tic()
                lambda_shape = c + y_phi_sum_i
                lambda_rate = (tau_shape / tau_rate)[:, None] + (gamma_shape / gamma_rate).sum(axis=0)
                tau_rate = d + (lambda_shape / lambda_rate).sum(axis=1)
                # logging.debug("beta_update_p2 %.4e" % clock.toc())
                
            if n_iter % self.stop_interval == 0:
                # clock.tic()
                # Compute the log likelihood and assess convergence
                # Expectations
                egamma = make_nonzero(gamma_shape / gamma_rate)
                elambda = make_nonzero(lambda_shape / lambda_rate)
                # Sum over k for the expectations: for each u,i, compute sum[k](egamma[u, k] * elambda[i, k])
                # But we're only computing it for the nonzeros (indexed by u and i)
                s = (egamma[u] * elambda[i]).sum(axis=1)
                # We use gammaln to compute the log factorial, hence the "y + 1"
                log_likelihood = np.sum(y * np.log(s) - s - gammaln(y + 1))
                self.log_likelihoods.append(log_likelihood)
                # logging.debug("compute_lik %.4e" % clock.toc())

                # Time to stop?
                if n_iter >= self.max_iter:
                    break

                # Check for convergence
                if len(self.log_likelihoods) > 1:
                    # clock.tic()
                    prev_ll = self.log_likelihoods[-2]
                    diff = abs((log_likelihood - prev_ll) / prev_ll)
                    logging.info(f"Iteration {n_iter}, ll = {log_likelihood:.0f}, diff = {diff:.6f}")
                    # logging.debug("log_lik_calc %.4e" % clock.toc())
                    if diff < self.stop_at_ll:
                        break
        # End of the main fitting loop
        # Compute beta and theta, which are given by the expectations, i.e. shape / rate
        # clock.tic()
        beta = lambda_shape / lambda_rate
        theta = gamma_shape / gamma_rate
        # logging.debug("finalize %.4e" % clock.toc())

        if not beta_precomputed:
            # Save these for future use in self.transform()
            self._tau_shape = tau_shape
            self._tau_rate = tau_rate
            self._lambda_shape = lambda_shape
            self._lambda_rate = lambda_rate

        if n_threads is not None:
            numexpr.set_num_threads(previous_setting)
        
        return (beta, theta)

    def transform(self, X: sparse.coo_matrix, n_threads: int=None) -> np.ndarray:
        """
        Transform the data matrix using an already fitted HPF model

        Args:
            X      Data matrix, shape (n_users, n_items)

        Returns:
            Factor matrix theta of shape (n_users, k)
        """
        if type(X) is not sparse.coo_matrix:
            raise TypeError("Input matrix must be in sparse.coo_matrix format")

        (beta, theta) = self._fit(X, beta_precomputed=True, n_threads=n_threads)
        return theta


"""PROFILING RESULTS of input 1500 samples, 1000 genes
make_nonzero 2.1148e-04
digamma normalization 1.5492e+00
theta update 1.3353e-01
beta update 1.4134e-01
loglik computation 1.0144e-01"""


class Clock:
    def __init__(self) -> None:
        self.internal = 0.

    def tic(self) -> None:
        self.internal = time.time()
    
    def toc(self) -> float:
        return time.time() - self.internal

    def reset(self) -> None:
        self.internal = 0
