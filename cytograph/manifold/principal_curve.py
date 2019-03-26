import numpy as np
from typing import *
import logging


class PrincipalCurve:
    """
    Ozertem et al. principal curve algorithm

    Untested
    """
    def __init__(self, tolerance: float = 0.0001, d: int = 1, h: int = 3, maxiter: int = 600) -> None:
        self.tolerance = tolerance
        self.d = d
        self.h = h
        self.maxiter = maxiter

    def fit(self, data: np.ndarray, points: np.ndarray) -> np.ndarray:
        '''Find the principal curve of a datset given a mesh of points

        In the most common use data and points are the same array. 
        (More generally points can be any grid of points)
        Note: this function runs the main loop going trough the points
        data is (dimension, points)
        '''
        data = np.copy(np.atleast_2d(data))
        n, N = data.shape
        points = np.copy(np.atleast_2d(points))
        m, M = points.shape
        if m == 1 and M == n:  # row vector
            points = np.reshape(points, (n, 1))
            m, M = points.shape
        # For every point
        for k in range(M):
            #  print "%d of %d" % (k+1, M)
            if k == int(N / 4):
                logging.info("25% complete")
            if k == int(N / 2):
                logging.info("50% complete")
            if k == int(3 * N / 4):
                logging.info("75% complete")
            points[:, k] = self._projectpoint(data, points[:, k])
        return points

    def _projectpoint(self, data: np.ndarray, point: np.ndarray) -> np.ndarray:
        '''The function that does the core of the job
        '''
        n, N = data.shape
        converged = False
        citer = 0
        while not(converged):
            probs = self._p(data, point)
            H = self._hess(data, point)
            w, v = np.linalg.eigh(H)
            index = np.argsort(w)  # arguments that sort from small to large
            # print(v.shape, list(range(n - self.d)))
            V = np.reshape(v[:, index[:(n - self.d)]], (n, n - self.d))
            ospace = np.dot(V, V.T)
            proj = np.reshape(self._ms(data, point), (n, 1)) - np.reshape(point, (n, 1))
            proj = np.dot(ospace, proj) + np.reshape(point, (n, 1))
            diff = np.linalg.norm(np.reshape(point, (n, 1)) - proj)
            point = np.reshape(proj, (n, ))
            citer = citer + 1
            if diff < self.tolerance:  # stopping condition based on distance
                converged = True
            if citer > self.maxiter:
                converged = True
                print("maximum iterations exceeded")
        return point

    def _hess(self, data: np.ndarray, x: np.ndarray) -> float:
        """A numpy-broadcasting speedup calculation of the hessian
        """
        data = np.atleast_2d(data)
        n, N = data.shape
        x = np.atleast_2d(x)
        m, M = x.shape
        if M != 1:
            x = x.T
        Sigmainv = np.identity(n) * (1 / (self.h**2))
        cs = self._c(data, x)
        us = self._u(data, x)
        Hx = np.sum(cs * ((us[:, None, :] * us) - Sigmainv[:, :, None]), 2) / N
        return Hx

    def _kern(self, x: np.ndarray) -> np.ndarray:
        "Gaussian Kernel Profile"
        return np.exp(-x / 2.0)

    def _p(self, data: np.ndarray, points: np.ndarray) -> np.ndarray:
        "Evaluate KDE on a set of points based on data"
        data = np.atleast_2d(data)
        n, N = data.shape
        points = np.atleast_2d(points)
        m, M = points.shape
        if m == 1 and M == n:  # row vector
            points = np.reshape(points, (n, 1))
            m, M = points.shape
        const = (1.0 / N) * ((self.h)**(-n)) * (2.0 * np.pi)**(-n / 2.0)
        probs = np.zeros((M,), dtype=np.float)
        for i in range(M):
            diff = (data - points[:, i, None]) / self.h
            x = np.sum(diff * diff, axis=0)
            probs[i] = np.sum(self._kern(x), axis=0) * const
        return probs

    def _u(self, data: np.ndarray, x: np.ndarray) -> np.ndarray:
        data = np.atleast_2d(data)
        n, N = data.shape
        x = np.atleast_2d(x)
        m, M = x.shape
        if M != 1:
            x = np.reshape(x, (n, 1))
        us = (x - data) / (self.h**2)
        return us

    def _c(self, data: np.ndarray, x: np.ndarray) -> np.ndarray:
        data = np.atleast_2d(data)
        n, N = data.shape
        x = np.atleast_2d(x)
        m, M = x.shape
        if M != 1:
            x = x.T
        us = self._u(data, x)
        const = (self.h**(-n)) * (2.0 * np.pi)**(-n / 2.0)
        u2 = np.sum(us * (us * (self.h**2)), axis=0)
        cs = self._kern(u2) * const
        return cs

    def _ms(self, data: np.ndarray, x: np.ndarray) -> np.ndarray:
        "Calculate the mean-shift at point x"
        data = np.atleast_2d(data)
        n, N = data.shape
        const = (1.0 / N) * ((self.h)**(-n)) * (2.0 * np.pi)**(-n / 2.0)
        x = np.atleast_2d(x)
        m, M = x.shape
        if M != 1:
            x = np.reshape(x, (n, 1))
        unprobs = self._p(data, x) / const
        diff = (data - x) / self.h
        diff2 = np.sum(diff * diff, axis=0)
        mx = np.sum(self._kern(diff2) * data, axis=1) / unprobs
        mx = np.reshape(mx, (n, 1))
        return mx

    def _grad(self, data: np.ndarray, x: np.ndarray) -> np.ndarray:
        "calculate the local gradient of the kernel density"
        "g(x) = (p(x)/h^2)*[mean-shift(x) - x]"
        data = np.atleast_2d(data)
        n, N = data.shape
        x = np.atleast_2d(x)
        m, M = x.shape
        if M != 1:
            x = np.reshape(x, (n, 1))
        probs = self._p(data, x)
        mx = np.reshape(self._ms(data, x), (n, 1))
        gx = (probs / (self.h**2)) * (mx - x)
        return gx

    def _grad2(self, data: np.ndarray, x: np.ndarray) -> np.ndarray:
        "calculate the local gradient of the kernel density"
        "g(x) = (-1/N)sum(ci*ui)"
        data = np.atleast_2d(data)
        n, N = data.shape
        x = np.atleast_2d(x)
        m, M = x.shape
        if M != 1:
            x = np.reshape(x, (n, 1))
        cs = self._c(data, x)
        us = self._u(data, x)
        gx = -np.sum(cs * us, axis=1) / N
        gx = np.reshape(gx, (n, 1))
        return gx
