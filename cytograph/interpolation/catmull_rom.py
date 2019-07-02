import numpy as np


# https://en.wikipedia.org/wiki/Centripetal_Catmull–Rom_spline
class CatmullRomSpline:
	def __init__(self, alpha: float = 0.5, n_points: int = 100) -> None:
		"""
		Args:
			alpha    The knot parameterization (0 for standard Catmull-Rom, 0.5 for centripetal, 1 for chordal)
			n_points The number of interpolation points to use for each curve segment
		"""
		self.alpha = alpha
		self.n_points = n_points
		
	def _spline(self, points: np.ndarray) -> np.ndarray:
		"""
		points is a matrix of four points, shape (4, 2), that define the Catmull-Rom spline.
		n_points is the number of interpolated points to use for this curve segment.
		"""
		p0, p1, p2, p3 = [points[i, :] for i in range(4)]

		# Calculate t0 to t4
		def tj(ti: float, pi: np.ndarray, pj: np.ndarray) -> float:
			xi, yi = pi
			xj, yj = pj
			return (((xj - xi)**2 + (yj - yi)**2)**0.5)**self.alpha + ti

		t0 = 0
		t1 = tj(t0, p0, p1)
		t2 = tj(t1, p1, p2)
		t3 = tj(t2, p2, p3)

		# Only calculate points between P1 and P2
		t = np.linspace(t1, t2, self.n_points)

		# Reshape so that we can multiply by the points P0 to P3
		# and get a point for each value of t.
		t = t.reshape(len(t), 1)
		a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
		a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
		a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
		b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
		b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3

		return (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2

	def fit_transform(self, points: np.ndarray) -> np.ndarray:
		"""
		Calculate Catmull Rom for a chain of points and return the combined curve.
		"""
		sz = points.shape[0]

		# The curve will contain an array of (x,y) points.
		curve: np.ndarray = []
		for i in range(sz - 3):
			c = self._spline(points[i:i + 4])
			curve.extend(c)

		return curve
