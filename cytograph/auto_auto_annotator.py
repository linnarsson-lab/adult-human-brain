from typing import *
import os
import logging
import numpy as np
import pandas as pd
import re
import loompy


class AutoAutoAnnotator:
	def __init__(self, n: int = 2) -> None:
		self.n = n
	
	def fit(self, dsagg: loompy.LoomConnection) -> None:
		pass
	
	def _fit_one(self, trinaries: np.ndarray) -> List[int]:
		"""
		Find the best n-tuple of markers to identify cluster #ix
		"""
		
