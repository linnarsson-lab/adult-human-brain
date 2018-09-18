import warnings
import os
import numpy as np
import cytograph as cg
from numba import jit as numba_jit
import numba
from concurrent.futures import ThreadPoolExecutor

#
# Parallelized and numba-jitted version of BallTree NN search specifically for Jensen Shannon distance
#
# Based on this code by Jake Vanderplas and David Patschke:
# https://gist.github.com/dpatschke/f5793db4c1d9cf55b3d16b2fc25c63e3 
#



#----------------------------------------------------------------------
# Distance computations

@numba.jit(nopython=True, nogil=True)
def rdist_euclidean(X1, i1, X2, i2):
	d = 0
	for k in range(X1.shape[1]):
		tmp = (X1[i1, k] - X2[i2, k])
		d += tmp * tmp
	return d


@numba.jit(nopython=True, nogil=True)
def rdist(X1, i1, X2, i2):
	return cg.jensen_shannon_distance(X1[i1, :], X2[i2, :])


@numba.jit(nopython=True, nogil=True)
def min_rdist(node_centroids, node_radius, i_node, X, j):
	d = rdist(node_centroids, i_node, X, j)
	return np.square(max(0, np.sqrt(d) - node_radius[i_node]))


#----------------------------------------------------------------------
# Heap for distances and neighbors

@numba.jit(nopython=True, nogil=True)
def heap_create(N, k):
	distances = np.full((N, k), np.finfo(np.float64).max)
	indices = np.zeros((N, k), dtype=np.int64)
	return distances, indices


def heap_sort(distances, indices):
	i = np.arange(len(distances), dtype=int)[:, None]
	j = np.argsort(distances, 1)
	return distances[i, j], indices[i, j]


@numba.jit(nopython=True, nogil=True)
def heap_push(row, val, i_val, distances, indices):
	size = distances.shape[1]

	# check if val should be in heap
	if val > distances[row, 0]:
		return

	# insert val at position zero
	distances[row, 0] = val
	indices[row, 0] = i_val

	#descend the heap, swapping values until the max heap criterion is met
	i = 0
	while True:
		ic1 = 2 * i + 1
		ic2 = ic1 + 1

		if ic1 >= size:
			break
		elif ic2 >= size:
			if distances[row, ic1] > val:
				i_swap = ic1
			else:
				break
		elif distances[row, ic1] >= distances[row, ic2]:
			if val < distances[row, ic1]:
				i_swap = ic1
			else:
				break
		else:
			if val < distances[row, ic2]:
				i_swap = ic2
			else:
				break

		distances[row, i] = distances[row, i_swap]
		indices[row, i] = indices[row, i_swap]

		i = i_swap

	distances[row, i] = val
	indices[row, i] = i_val

#----------------------------------------------------------------------
# Tools for building the tree

@numba.jit(nopython=True, nogil=True)
def _partition_indices(data, idx_array, idx_start, idx_end, split_index):
	# Find the split dimension
	n_features = data.shape[1]

	split_dim = 0
	max_spread = 0

	for j in range(n_features):
		max_val = -np.inf
		min_val = np.inf
		for i in range(idx_start, idx_end):
			val = data[idx_array[i], j]
			max_val = max(max_val, val)
			min_val = min(min_val, val)
		if max_val - min_val > max_spread:
			max_spread = max_val - min_val
			split_dim = j

	# Partition using the split dimension
	left = idx_start
	right = idx_end - 1

	while True:
		midindex = left
		for i in range(left, right):
			d1 = data[idx_array[i], split_dim]
			d2 = data[idx_array[right], split_dim]
			if d1 < d2:
				tmp = idx_array[i]
				idx_array[i] = idx_array[midindex]
				idx_array[midindex] = tmp
				midindex += 1
		tmp = idx_array[midindex]
		idx_array[midindex] = idx_array[right]
		idx_array[right] = tmp
		if midindex == split_index:
			break
		elif midindex < split_index:
			left = midindex + 1
		else:
			right = midindex - 1


@numba.jit(nopython=True, nogil=True)
def _recursive_build(i_node, idx_start, idx_end,
					 data, node_centroids, node_radius, idx_array,
					 node_idx_start, node_idx_end, node_is_leaf,
					 n_nodes, leaf_size):
	# determine Node centroid
	for j in range(data.shape[1]):
		node_centroids[i_node, j] = 0
		for i in range(idx_start, idx_end):
			node_centroids[i_node, j] += data[idx_array[i], j]
		node_centroids[i_node, j] /= (idx_end - idx_start)

	# determine Node radius
	sq_radius = 0.0
	for i in range(idx_start, idx_end):
		sq_dist = rdist(node_centroids, i_node, data, idx_array[i])
		if sq_dist > sq_radius:
			sq_radius = sq_dist

	# set node properties
	node_radius[i_node] = np.sqrt(sq_radius)
	node_idx_start[i_node] = idx_start
	node_idx_end[i_node] = idx_end

	i_child = 2 * i_node + 1

	# recursively create subnodes
	if i_child >= n_nodes:
		node_is_leaf[i_node] = True
		if idx_end - idx_start > 2 * leaf_size:
			# this shouldn't happen if our memory allocation is correct.
			# We'll proactively prevent memory errors, but raise a
			# warning saying we're doing so.
			#warnings.warn("Internal: memory layout is flawed: "
			#              "not enough nodes allocated")
			pass

	elif idx_end - idx_start < 2:
		# again, this shouldn't happen if our memory allocation is correct.
		#warnings.warn("Internal: memory layout is flawed: "
		#              "too many nodes allocated")
		node_is_leaf[i_node] = True

	else:
		# split node and recursively construct child nodes.
		node_is_leaf[i_node] = False
		n_mid = int((idx_end + idx_start) // 2)
		_partition_indices(data, idx_array, idx_start, idx_end, n_mid)
		_recursive_build(i_child, idx_start, n_mid,
						 data, node_centroids, node_radius, idx_array,
						 node_idx_start, node_idx_end, node_is_leaf,
						 n_nodes, leaf_size)
		_recursive_build(i_child + 1, n_mid, idx_end,
						 data, node_centroids, node_radius, idx_array,
						 node_idx_start, node_idx_end, node_is_leaf,
						 n_nodes, leaf_size)


#----------------------------------------------------------------------
# Tools for querying the tree
@numba.jit(nopython=True, nogil=True)
def _query_recursive(i_node, X, i_pt, heap_distances, heap_indices, sq_dist_LB,
					 data, idx_array, node_centroids, node_radius,
					 node_is_leaf, node_idx_start, node_idx_end):
	#------------------------------------------------------------
	# Case 1: query point is outside node radius:
	#         trim it from the query
	if sq_dist_LB > heap_distances[i_pt, 0]:
		pass

	#------------------------------------------------------------
	# Case 2: this is a leaf node.  Update set of nearby points
	elif node_is_leaf[i_node]:
		for i in range(node_idx_start[i_node],
					   node_idx_end[i_node]):
			dist_pt = rdist(data, idx_array[i], X, i_pt)
			if dist_pt < heap_distances[i_pt, 0]:
				heap_push(i_pt, dist_pt, idx_array[i],
						  heap_distances, heap_indices)

	#------------------------------------------------------------
	# Case 3: Node is not a leaf.  Recursively query subnodes
	#         starting with the closest
	else:
		i1 = 2 * i_node + 1
		i2 = i1 + 1
		sq_dist_LB_1 = min_rdist(node_centroids,
								 node_radius,
								 i1, X, i_pt)
		sq_dist_LB_2 = min_rdist(node_centroids,
								 node_radius,
								 i2, X, i_pt)

		# recursively query subnodes
		if sq_dist_LB_1 <= sq_dist_LB_2:
			_query_recursive(i1, X, i_pt, heap_distances,
							 heap_indices, sq_dist_LB_1,
							 data, idx_array, node_centroids, node_radius,
							 node_is_leaf, node_idx_start, node_idx_end)
			_query_recursive(i2, X, i_pt, heap_distances,
							 heap_indices, sq_dist_LB_2,
							 data, idx_array, node_centroids, node_radius,
							 node_is_leaf, node_idx_start, node_idx_end)
		else:
			_query_recursive(i2, X, i_pt, heap_distances,
							 heap_indices, sq_dist_LB_2,
							 data, idx_array, node_centroids, node_radius,
							 node_is_leaf, node_idx_start, node_idx_end)
			_query_recursive(i1, X, i_pt, heap_distances,
							 heap_indices, sq_dist_LB_1,
							 data, idx_array, node_centroids, node_radius,
							 node_is_leaf, node_idx_start, node_idx_end)


@numba.jit(nopython=True, parallel=True, nogil=True)
def _query_parallel(i_node, X, heap_distances, heap_indices,
					 data, idx_array, node_centroids, node_radius,
					 node_is_leaf, node_idx_start, node_idx_end):
	for i_pt in numba.prange(X.shape[0]):
		sq_dist_LB = min_rdist(node_centroids, node_radius, i_node, X, i_pt)
		_query_recursive(i_node, X, i_pt, heap_distances, heap_indices, sq_dist_LB,
						 data, idx_array, node_centroids, node_radius, node_is_leaf,
						 node_idx_start, node_idx_end)


@numba.jit(nopython=True, nogil=True)
def query_batch(k, i_node, X,
				data, idx_array, node_centroids, node_radius,
				node_is_leaf, node_idx_start, node_idx_end):
	# initialize heap for neighbors
	heap_distances, heap_indices = heap_create(X.shape[0], k)
	for i_pt in range(X.shape[0]):
		sq_dist_LB = min_rdist(node_centroids, node_radius, i_node, X, i_pt)
		_query_recursive(i_node, X, i_pt, heap_distances, heap_indices, sq_dist_LB,
						data, idx_array, node_centroids, node_radius, node_is_leaf,
						node_idx_start, node_idx_end)
	return (heap_distances, heap_indices)

#----------------------------------------------------------------------
# The Ball Tree object
class BallTreeJS(object):
	def __init__(self, data, leaf_size=40, n_threads=0):
		self.data = data
		self.leaf_size = leaf_size
		self.n_threads = n_threads
		if n_threads == 0:
			if os.cpu_count() is not None:
				self.n_threads = max(os.cpu_count() // 2, 1)  # type: ignore
			else:
				self.n_threads = 1

		# validate data
		if self.data.size == 0:
			raise ValueError("X is an empty array")

		if leaf_size < 1:
			raise ValueError("leaf_size must be greater than or equal to 1")

		self.n_samples = self.data.shape[0]
		self.n_features = self.data.shape[1]

		# determine number of levels in the tree, and from this
		# the number of nodes in the tree.  This results in leaf nodes
		# with numbers of points betweeen leaf_size and 2 * leaf_size
		self.n_levels = 1 + np.log2(max(1, ((self.n_samples - 1)
											// self.leaf_size)))
		self.n_nodes = int(2 ** self.n_levels) - 1

		# allocate arrays for storage
		self.idx_array = np.arange(self.n_samples, dtype=int)
		self.node_radius = np.zeros(self.n_nodes, dtype=float)
		self.node_idx_start = np.zeros(self.n_nodes, dtype=int)
		self.node_idx_end = np.zeros(self.n_nodes, dtype=int)
		self.node_is_leaf = np.zeros(self.n_nodes, dtype=int)
		self.node_centroids = np.zeros((self.n_nodes, self.n_features),
									   dtype=float)

		# Allocate tree-specific data from TreeBase
		_recursive_build(0, 0, self.n_samples,
						 self.data, self.node_centroids,
						 self.node_radius, self.idx_array,
						 self.node_idx_start, self.node_idx_end,
						 self.node_is_leaf, self.n_nodes, self.leaf_size)

	def query(self, X=None, k=1, sort_results=True):
		"""
		Remarks: if X is None, use data and do not return self as neighbor
		"""
		return_self = True
		if X is None:
			X = self.data
			return_self = False
		X = np.asarray(X, dtype=float)

		if X.shape[-1] != self.n_features:
			raise ValueError("query data dimension must "
							 "match training data dimension")

		n_samples = self.data.shape[0]
		if n_samples < k:
			raise ValueError("k must be less than or equal "
							 "to the number of training points")

		# flatten X, and save original shape information
		Xshape = X.shape
		X = X.reshape((-1, self.data.shape[1]))

		with ThreadPoolExecutor(max_workers=self.n_threads) as tx:
			batch_size = int(n_samples // self.n_threads)
			start = 0
			futures = []
			while start < n_samples:
				futures.append(tx.submit(query_batch, k, 0, X[start:start + batch_size, :], 
							self.data, self.idx_array, self.node_centroids, self.node_radius,
							self.node_is_leaf, self.node_idx_start, self.node_idx_end))
				start += batch_size
			dist_batches = []
			indices_batches = []
			for future in futures:
				d, i = future.result()
				d, i = heap_sort(d, i)
				dist_batches.append(d)
				indices_batches.append(i)
		distances = np.concatenate(dist_batches).reshape(Xshape[:-1] + (k,))
		indices = np.concatenate(indices_batches).reshape(Xshape[:-1] + (k,))
		if not return_self:
			distances = distances[:, 1:]
			indices = indices[:, 1:]
		return distances, indices


#----------------------------------------------------------------------
# Testing function

def test_tree(N=1000, D=3, K=5, LS=40):
	from time import time
	from sklearn.neighbors import BallTree as skBallTree

	print("-------------------------------------------------------")
	print("Numba version: " + numba.__version__)

	rseed = np.random.randint(10000)
	print("-------------------------------------------------------")
	print("{0} neighbors of {1} points in {2} dimensions".format(K, N, D))
	print("random seed = {0}".format(rseed))
	np.random.seed(rseed)
	X = np.random.random((N, D))

	# pre-run to jit compile the code
	BallTree(X, leaf_size=LS).query(X, K)

	t0 = time()
	bt1 = skBallTree(X, leaf_size=LS)
	t1 = time()
	dist1, ind1 = bt1.query(X, K)
	t2 = time()

	bt2 = BallTree(X, leaf_size=LS)
	t3 = time()
	dist2, ind2 = bt2.query(X, K)
	t4 = time()

	print("results match: {0} {1}".format(np.allclose(dist1, dist2),
										  np.allclose(ind1, ind2)))
	print("")
	print("sklearn build: {0:.3g} sec".format(t1 - t0))
	print("numba build  : {0:.3g} sec".format(t3 - t2))
	print("")
	print("sklearn query: {0:.3g} sec".format(t2 - t1))
	print("numba query  : {0:.3g} sec".format(t4 - t3))


if __name__ == '__main__':
	test_tree()