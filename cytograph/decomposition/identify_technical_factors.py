import numpy as np
from typing import List


def identify_technical_factors(theta: np.ndarray, batches: np.ndarray, replicates: np.ndarray) -> np.ndarray:
	"""
	Identify HPF factors that are substantially enriched or depleted in individual
	technical replicates.

	Return:
		A boolean mask array indicating the technical factors
	"""
	technical: List[int] = []
	for batch in np.unique(batches):
		repl_for_batch = replicates[batches == batch]
		max_fracs = []
		min_fracs = []
		for ix in range(theta.shape[1]):
			factor = theta[:, ix]
			repl_frac = []
			expected_frac = []
			for repl in np.unique(repl_for_batch):
				repl_frac.append(factor[(batch == batches) & (repl == replicates)].sum())
				expected_frac.append(((batch == batches) & (repl == replicates)).sum())
			repl_frac = np.array(repl_frac) / np.sum(repl_frac)
			expected_frac = np.array(expected_frac) / np.sum(expected_frac)
			max_fracs.append(np.max(repl_frac / expected_frac))  # type: ignore
			min_fracs.append(np.min(repl_frac / expected_frac))  # type: ignore
		max_fracs = np.array(max_fracs)
		min_fracs = np.array(min_fracs)
		technical += list(np.where(((min_fracs < 0.5) | (max_fracs > 2)))[0])  # type: ignore
	technical_mask = np.zeros(theta.shape[1], dtype=bool)
	if len(technical) == 0:
		return technical_mask
	technical_mask[np.unique(technical)] = True
	return technical_mask
