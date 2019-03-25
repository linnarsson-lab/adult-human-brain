from typing import *
import logging
from collections import defaultdict
import numpy as np
import numpy_groupies as npg
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, leaves_list
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker as ticker
import loompy
import os
import logging as lg
import luigi
import random
import string
from sklearn.preprocessing import LabelEncoder


pd.options.mode.chained_assignment = None  # this is because of a warning in prepare_heat_map
lg.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=lg.INFO)


def div0(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	""" ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
	with np.errstate(divide='ignore', invalid='ignore'):
		c = np.true_divide(a, b)
		c[~np.isfinite(c)] = 0  # -inf inf NaN
	return c


def logging(task: luigi.Task, log_dependencies: bool = False) -> lg.Logger:
	logger_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
	log_file = task.output().path + ".log"
	logger = lg.getLogger(logger_name)
	formatter = lg.Formatter('%(asctime)s %(levelname)s: %(message)s')
	fileHandler = lg.FileHandler(log_file, mode='w')
	fileHandler.setFormatter(formatter)
	# streamHandler = lg.StreamHandler()
	# streamHandler.setFormatter(formatter)

	logger.setLevel(lg.INFO)
	logger.addHandler(fileHandler)
	# logger.addHandler(streamHandler)

	if log_dependencies:
		logger.info("digraph G {")
		graph: Dict[str, List[str]] = {}

		def compute_task_graph(task: luigi.Task) -> None:
			name = task.__str__().split('(')[0]
			for dep in task.deps():
				if name in graph:
					graph[name].append(dep.__str__().split('(')[0])
				else:
					graph[name] = [dep.__str__().split('(')[0]]
				compute_task_graph(dep)

		compute_task_graph(task)
		for k, v in graph.items():
			for u in set(v):
				logger.info('"' + u + '" -> "' + k + '";')
		logger.info("}")
		logger.info("")

	for p in task.get_param_names():
		logger.info(f"{p} = {task.__dict__[p]}")
	logger.info("===")
	return logger


def cap_select(labels: np.ndarray, items: np.ndarray, max_n: int) -> np.ndarray:
	"""
	Return a list of items but with no more than max_n entries
	having each unique label
	"""
	n_labels = np.max(labels) + 1
	sizes = np.bincount(labels, minlength=n_labels)
	result = []  # type: List[int]
	for lbl in range(n_labels):
		n = min(max_n, sizes[lbl])
		selected = np.where(labels == lbl)[0]
		result = result + list(np.random.choice(selected, n, False))
	return items[np.array(result)]


def loompy2data(filename: str) -> pd.DataFrame:
	"""Load a loompy file as a pandas dataframe dropping column and row annotation
	
	Args
	----
	filename : str path to the .loom file

	Returns
	-------
	df: pd.DataFrame data matrix with columns:CellId index:Accession 
	"""
	ds = loompy.connect(filename)
	return pd.DataFrame(data=ds[:, :], columns=ds.col_attrs['CellID'], index=ds.row_attrs['Accession']).astype(int)


def loompy2annot(filename: str) -> pd.DataFrame:
	"""Load the column attributes from the loompy file as a pandas dataframe
	
	Args
	----
	filename : str path to the .loom file

	Returns
	-------
	df: pd.DataFrame column annotations index:CellId
	"""
	ds = loompy.connect(filename)
	return pd.DataFrame(ds.col_attrs, index=ds.col_attrs['CellID']).T


def loompy2data_annot(filename: str, dtype: type = int) -> Tuple[loompy.LoomConnection, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Load a loompy file as a pandas dataframe dropping column and row annotation
	
	Args
	----
	filename : str path to the .loom file

	Returns
	-------
	ds: loompy.LoomConnection the connection to the loom file
	df: pd.DataFrame data matrix with columns:CellId index:Gene
	cols_df: pd.DataFrame column annotations index:CellId
	rows_df: pd.DataFrame column annotations index:Accession
	"""
	ds = loompy.connect(filename)
	ret= (ds,
			pd.DataFrame(data=ds[:, :],
						columns=ds.col_attrs['CellID'],
						index=ds.row_attrs['Accession'], dtype=dtype),
			pd.DataFrame( ds.col_attrs,
						index=ds.col_attrs['CellID'] ).T,
			pd.DataFrame( ds.row_attrs,
						index=ds.row_attrs['Accession'] ).T)
	ds.close()
	return ret


def loompy2valid_data_annot(filename: str, dtype: type = int) -> Tuple[loompy.LoomConnection, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Load valid cells from a loompy file as a pandas dataframe dropping column and row annotation
	Based on the _Valid column attribute
	Args
	----
	filename : str path to the .loom file

	Returns
	-------
	ds: loompy.LoomConnection the connection to the loom file
	df: pd.DataFrame data matrix with columns:CellId index:Gene
	cols_df: pd.DataFrame column annotations index:CellId
	rows_df: pd.DataFrame column annotations index:Accession
	"""
	ds, df, cols_df, rows_df = loompy2data_annot(filename, dtype)
	# Filter Valid
	bool_selection = (cols_df.ix["_Valid", :] == 1)
	df = df.ix[:, bool_selection]
	cols_df = cols_df.ix[:, bool_selection]
	return ds, df, cols_df, rows_df


def marker_table(df: pd.DataFrame, groups: np.ndarray, avg_N: int = 30) -> Tuple[DefaultDict, np.ndarray]:
	"""Produce a marker table using fold enrichemnt * fraction expressing heuristic

	Args
	----
	df: pd.Dataframe (rows: Acessions, cols: Cells)
	groups: labels of the groups (it is assumed to be integers without gaps) shape df.shape[0]
	avg_N: Average number of genes to select per cluster (actual number depends from the overlaps of the 3 scores)

	Returns
	-------
	markers: dict(set) key = group, values sets of Accessions
	mus : np.ndarray shape=(df.shape[0], len(set(groups)))
	"""
	logging.debug("Computing Marker Table")
	N = int(np.ceil(avg_N / 3))
	mus = npg.aggregate_numba.aggregate(groups, df.values, func="mean", axis=1)
	counts_per_group = npg.aggregate_numba.aggregate(groups,1)
	mu0 = np.sum((mus * counts_per_group) / len(groups), 1)  # faster than X.mean(1)
	fold = np.zeros_like(mus)
	iz = mu0 > 0.001  # avoid Nans and useless high precision calculations
	fold[iz, :] = mus[iz, :] / mu0[iz, None]
	fs = npg.aggregate_numba.aggregate(groups, df.values > 0, func="mean", axis=1)
	
	# Filters
	fold *= (mus > 1) * (fold > 1.5)
	fs *= (fs > 0.25)
	
	# Scores and sorting
	score00 = fold
	score05 = fold * fs**0.5
	score10 = fold * fs
	ix00 = np.argsort(score00, 0)[::-1, :]
	ix05 = np.argsort(score05, 0)[::-1, :]
	ix10 = np.argsort(score10, 0)[::-1, :]
	score10_df = pd.DataFrame(score10, columns=np.unique(groups), index=df.index)
	genes00 = df.index.values[ix00][:N, :]
	genes05 = df.index.values[ix05][:N, :]
	genes10 = df.index.values[ix10][:N, :]
	markers_array = np.vstack((genes00, genes05,genes10))
	markers = defaultdict(set)  # type: defaultdict
	for ct in range(mus.shape[1]):
			markers[ct] |= set(markers_array[:,ct])
	for ct in range(mus.shape[1]):
		for mk in list(markers[ct]):
			if score10_df.ix[mk, ct] < 0.05:
				markers[ct] -= set([mk])
		for mk in markers[ct]:
			for ct2 in list(set(range(mus.shape[1])) - set([ct])):
				if score10_df.ix[mk,ct] >= score10_df.ix[mk, ct2]:
					markers[ct2] -= set([mk])
		
					
	return markers, mus

def prepare_heat_map(df: pd.DataFrame, cols_df: pd.DataFrame,
					rows_df: pd.DataFrame, marker_n: int) -> Tuple[Any, Any, Any, Any, Any, Any]:
	'''
	Prepare all the inputs necessary to plot a marker heatmap
	
	Args
	----
	df
	cols_df
	fows_df
	marker_n
	
	Returns
	-------
	df_markers
	rows_df_markers
	cols_df_sorted
	accession_list
	gene_cluster
	mus

	Note
	====
	First it adds a random number to the cols to avoid duplicates cell names
	Then it filters away the cells flagged as _Valid==False
	
	'''
	
	# Reorganize the inputs
	logging.debug("Reorganizing the input")
	np.random.seed(15071990)
	cols_df.columns = cols_df.columns + np.random.randint(1000, 9999, size=cols_df.shape[1]).astype(str)
	cols_df.ix["Accession"] = cols_df.columns.values
	df.columns = cols_df.columns
	labels = cols_df.ix["Clusters"].values.astype(int)
	valid = cols_df.ix["_Valid"].values.astype(bool)
	df = df.ix[:,valid]
	labels = labels[valid]
	cols_df = cols_df.ix[:,valid]
	
	# Preare Table with top markers
	table, mus = marker_table(df, labels, marker_n)
	accession_selected = []  # type: List
	for i in table.values():
		accession_selected += list(i)
	mus_selected = mus[np.in1d(rows_df.columns.values, accession_selected),:]
	
	# Perform single linkage on correlation of the average pattern of the markers
	logging.debug("Sort the clusters by single linkage")
	z = linkage(np.log2(mus_selected.T + 1), 'average', 'correlation')
	order = leaves_list(z)
	
	logging.debug("Preparing output")
	# Modify the values of the labels to respect the dendrogram order
	ixs = np.argsort(order, kind="mergesort")
	labels_updated = ixs[labels]  # reattribute the label on the basis of the linkage

	# Sort the cells based on the updated labels
	ix0 = np.argsort(labels_updated, kind="mergesort")
	labels_sorted = labels_updated[ix0]
	cols_df_sorted = cols_df.ix[:, ix0]
	df_sorted = df.ix[:, ix0]
	cols_df_sorted.loc["Total Molecules", :] = df_sorted.sum(0).values

	# Generate a list of genes and gene cluster labels
	accession_list = []  # type: List
	gene_cluster = []  # type: List
	for i in order:
		accession_list += list(table[i])
		gene_cluster += [i]*len(table[i])
	gene_cluster = np.array(gene_cluster)
		
	rows_df_markers = rows_df.ix[:, accession_list]
	rows_df_markers.loc["Cluster", :] = np.array(gene_cluster)
	
	return df_sorted.ix[accession_list, :], rows_df_markers, cols_df_sorted, accession_list, gene_cluster, mus


def generate_pcolor_args(attribute_values: np.ndarray, kind: str = "categorical", cmap: Any = None, custom_list: List = None) -> Tuple[np.ndarray, Any]:
	"""
	
	Args
	----
	
	attribute_values (np.ndarray) : values of the attrybute
	
	kind (str) : one of "categorical"(default), "continuous", "binary", "bool", "custom"
	
	cmap (mpl.color.Colormap) : colormap to be used. The default is 0.3*cm.prism + 0.7*cm.spectral
	
	custom_list (list, default None) : if kind=="custom" is the list of selected colors to attribute to the sorted
	attribute values
	
	Return
	------
	values (np.ndarray) : array ready to be passed as first argument to pcolorfast
	
	colormap (mpl.color.Colormap) : colormap ready to be passed as cmap argument to pcolorfast

	"""
	if kind == "categorical":
		attributes, _, attrs_ix = np.unique(attribute_values, return_index=True, return_inverse=True)
		n_attrs = len(attribute_values)
		if not cmap:
			def spectral_prism(x: np.ndarray) -> np.ndarray:
				return 0.3 * plt.cm.prism(x) + 0.7 * plt.cm.spectral(x)
			cmap = spectral_prism
		color_list = cmap(attrs_ix / np.max(attrs_ix))
		generated_cmap = matplotlib.colors.ListedColormap(color_list, name='from_list')
		values = np.arange(n_attrs)
	elif kind == "continuous":
		values = np.array( attribute_values )
		values -= np.min(values)
		values /= np.max(values)
		if cmap:
			generated_cmap = cmap
		else:
			generated_cmap = plt.cm.viridis
	elif kind == "bool":
		generated_cmap = plt.cm.gray_r
		values = attribute_values.astype(int)
	elif kind == "binary":
		generated_cmap = plt.cm.gray_r
		values = (attribute_values == attribute_values[0]).astype(int)
	elif kind == "custom":
		levels, ix = np.unique(attribute_values, return_inverse=True )
	else:
		raise NotImplementedError("kind '%s' is not supported" % kind)
		
	return values, generated_cmap

def calculate_intensities(df_markers: pd.DataFrame) -> pd.DataFrame:
	logging.debug("Calculating intensites of the pixels")
	intensities = np.log2(df_markers + 1)
	intensities = intensities.sub(intensities.mean(1), axis="rows")
	standard_deviations = intensities.std(1).replace([np.nan, -np.inf], np.inf)  # substitute weird value with np.inf to get zero after division 
	return intensities.div(standard_deviations, axis="rows")


def super_heatmap(intensities: pd.DataFrame,
				  cols_annot: pd.DataFrame,
				  rows_annot: pd.DataFrame,
				  col_attrs: List[Tuple] = [ ("SampleID",), ("DonorID", ), ("Age", "multi"), ("Clusters", ) ],
				  row_attrs: List[Tuple] = [ ("Cluster",)]) -> None:
	'''Plots an interactive and informative heat map
	
	Args
	----
	intensities: pd.DataFrame
	cols_annot: pd.DataFrame
	rows_annot: pd.DataFrame
	col_attrs: List[Tuple]
	row_attrs: List[Tuple]
	
	Returns
	-------
	Nothing, plots the heatmap
	
	'''
	e = 0.03
	h_col_bar = 0.019
	w_row_bar = 0.03
	n_col_bars = len(col_attrs)
	n_row_bars = len(row_attrs)
	# Add extra column bars if there is a multiple column
	for (i, *k) in col_attrs:
		if "multi" in k:
			n_col_bars += len(np.unique(cols_annot.ix[i].values)) - 1
	delta_x = w_row_bar * n_row_bars
	delta_y = h_col_bar * n_col_bars

	fig = plt.figure(figsize=(12,9))
	# Determine the boudary and plot the heatmap
	left, bottom, width, height = delta_x + 3*e, 0 + e, 1 - delta_x - 4*e, 1 - delta_y - 2*e
	heatmap_bbox = [left, bottom, width, height]
	heatmap_ax = fig.add_axes(heatmap_bbox)
	heatmap_ax.pcolorfast(intensities.values, cmap=plt.cm.YlOrRd,\
		vmin=np.percentile(intensities, 2.5), vmax=np.percentile(intensities, 98.5))
	# Suppress labels on the heatmap axis
	heatmap_ax.tick_params(axis='x', labeltop='off', labelbottom='off', bottom="off" )
	heatmap_ax.tick_params(axis='y', labelleft='off', left='off', right='off', labelsize=1)

	# Column bars
	c = 0
	for (col_name, *kind) in col_attrs[::-1]:
		if kind == []:
			if len(np.unique(cols_annot.ix[col_name].values)) > 2:
				kind = ("categorical",)
			else:
				kind = ("binary",)
		if kind == ["multi"]:
			if col_name == "Age":
				# Make sure that the order of timepoints is correct even when they are not zero padded
				# Example E7.5 ahould be before E11.5
				age_float = []
				age_dict ={}
				for age in cols_annot.ix[col_name].values:
					age_float.append( float(age.strip("pPeE")) )
					age_dict[age_float[-1]] = age
				uq, inverse_uq = np.unique(age_float, return_inverse=True)
				uq = np.array([age_dict[i] for i in uq])
			else:
				uq, inverse_uq = np.unique(cols_annot.ix[col_name].values, return_inverse=True)
			
			for entry_ix in np.unique(inverse_uq)[::-1]:
				columnbar_bbox = [left , bottom + height + c*h_col_bar , width, h_col_bar]
				column_bar = fig.add_axes(columnbar_bbox, sharex=heatmap_ax)
				values, generated_cmap = generate_pcolor_args(inverse_uq == entry_ix, kind="bool")
				column_bar.pcolorfast(values[None,:], cmap=generated_cmap)
				column_bar.tick_params(axis='y', left='off', right='off', labelleft='off', labelright='off' )
				column_bar.tick_params(axis='x', bottom='off', top='off', labelbottom='off', labeltop='off' )
				plt.text(left-0.1*e, bottom + height + c*h_col_bar + 0.5*h_col_bar, col_name + " %s" % uq[entry_ix],
				ha='right', va='center', fontsize=7,transform = fig.transFigure) 
				c += 1
		else:
			columnbar_bbox = [left , bottom + height + c*h_col_bar , width, h_col_bar]
			column_bar = fig.add_axes(columnbar_bbox, sharex=heatmap_ax)
			values, generated_cmap = generate_pcolor_args(cols_annot.ix[col_name].values, kind=kind[0])
			column_bar.pcolorfast(values[None,:], cmap=generated_cmap)
			column_bar.tick_params(axis='y', left='off', right='off', labelleft='off', labelright='off' )
			# Special cases
			if col_name == "Clusters":
				column_bar.tick_params(axis='x', bottom='off', top='off', labelbottom='on', labeltop='off' )
				column_bar.tick_params(direction='out', pad=-9, colors='w')
				# boundary and center cluster positions
				bpos = np.where(np.diff(cols_annot.ix["Clusters"].values))[0]
				cpos = (np.r_[0, bpos] + np.r_[bpos, cols_annot.shape[1]]) / 2.
				# vertical cluster lines
				for b in bpos:
					heatmap_ax.axvline(b + 1, linewidth=0.5, c="darkred", alpha=0.6)
				uq, ix = np.unique(cols_annot.ix["Clusters"].values, return_index=True)
				order_pos = uq[np.argsort(ix)]
				# labels with names of the clusters
				plt.xticks(cpos, order_pos+1, fontsize=7, ha="center", va="center")
				for t in column_bar.xaxis.get_major_ticks():
					t.label1.set_fontweight('bold')
			else:
				column_bar.tick_params(axis='x', bottom='off', top='off', labelbottom='off', labeltop='off' )
			plt.text(left-0.1*e, bottom + height + c*h_col_bar + 0.5*h_col_bar, col_name,
				ha='right', va='center', fontsize=7,transform = fig.transFigure)  
			c += 1

	# Row bars
	for r, (row_name, *kind) in enumerate(row_attrs):
		if kind == []:
			if len(np.unique(rows_annot.ix[row_name].values)) > 2:
				kind = ("categorical",)
			else:
				kind = ("binary",)
		rowbar_bbox = [left-w_row_bar, bottom, w_row_bar, height]
		row_bar = fig.add_axes(rowbar_bbox, sharey=heatmap_ax)
		r_values = rows_annot.ix[row_name].values
		if np.equal(np.mod(r_values[0], 1), 0):
			# Try to avoid that, because of an empty marker list, the colors of
			# cell clusters and gene clusters stop to be the same
			missing = np.setdiff1d( np.arange(np.max(r_values)+1), r_values )
			logging.debug("Missing markers for clusters: %s" % missing)
			r_values = np.r_[r_values, missing]
			values, generated_cmap = generate_pcolor_args(r_values, kind=kind[0])
			row_bar.pcolorfast(values[:len(r_values),None], cmap=generated_cmap)
		else:
			values, generated_cmap = generate_pcolor_args(r_values, kind=kind[0])
			row_bar.pcolorfast(values[:,None], cmap=generated_cmap)
		row_bar.tick_params(axis='both', bottom='off', top='off',
							right='on',left='off',labelright="off",labelleft="on",labelbottom='off',
							direction='in', labelsize=9, colors='k')

		names = rows_annot.ix["Gene", :].values.astype(str)

		# Locator and formatter to show and hide gene names depnding on zoom level
		class Y_Locator(ticker.MaxNLocator):
			def tick_values(self, vmin: float, vmax: float) -> np.ndarray:
				if vmin < 0:
					vmin = 0
				if vmax - vmin > 90:
					return []
				else:
					return np.arange(int(vmin), int(vmax + 1)) + 0.5

		def my_formatter(x: Any, pos: float = None) -> str:
			return names[int(x)]

		row_bar.yaxis.set_major_formatter(ticker.FuncFormatter(my_formatter))
		row_bar.yaxis.set_major_locator(Y_Locator())
		
	fig.canvas.draw()  # plt.show() might be needed depending the backend and mode of execution


def create_markers_file(loom_file_path: str, marker_n: int = 100, overwrite: bool = False) -> None:
	"""Create a .marker (loom format) file that contains a (marker x cell) tables and all necessary annotation to plot it
	
	Args
	----
	loom_file_path: the path to the .loom file
	marker_n: the total number of genes will be approximatelly  N_clusters * marker_n / 3.
	
	Returns
	-------
	Nothing. Saves a file at loom_file_path.marker
	"""
	if os.path.exists(loom_file_path):
		marker_file_path = os.path.splitext(loom_file_path)[0] + ".markers"
		if os.path.exists(marker_file_path):
			if overwrite:
				logging.debug("Removing old version of %s" % marker_file_path)
				os.remove(marker_file_path)
			else:
				logging.debug("Previous version of %s was found, saving a backup" % marker_file_path)
				os.rename(marker_file_path, marker_file_path + '.bak')
	else:
		raise IOError("%s does not exist" % loom_file_path)

	ds, df, cols_df, rows_df = loompy2data_annot(loom_file_path)
	df_markers, rows_annot, cols_annot, accession_list, gene_cluster, mus = prepare_heat_map(df, cols_df, rows_df, marker_n=marker_n)
	loompy.create(marker_file_path, df_markers.values,
			{k:np.array(v) for k,v in rows_annot.T.to_dict("list").items()},
			{k:np.array(v) for k,v in cols_annot.T.to_dict("list").items()})

def plot_markers_file(markers_file_path: str,
					col_attrs: List = [("DonorID", ), ("SampleID",), ("Age", "multi"),("Clusters", )],
					row_attrs: List = [("Cluster",)]) -> None:
	"""Loads and plot Marker file
	
	Args
	----
	marker_file_path
	col_attrs
	row_attrs
	
	Returns
	-------
	Nothing. Opens a matplotlib window with the heatmap.
	"""
	ds, df_markers, cols_annot, rows_annot = loompy2data_annot(markers_file_path)
	intensities = calculate_intensities(df_markers)
	logging.debug("Preparing the plot")
	super_heatmap(intensities, cols_annot, rows_annot, col_attrs, row_attrs)

	
def replace(it1, it2):
	if len(it1) != len(it2):
		raise ValueError("Lengths of iterables are different")
	return zip(it1, it2)

color_alphabet = np.array([
    [240,163,255],[0,117,220],[153,63,0],[76,0,92],[0,92,49],[43,206,72],[255,204,153],[128,128,128],[148,255,181],[143,124,0],[157,204,0],[194,0,136],[0,51,128],[255,164,5],[255,168,187],[66,102,0],[255,0,16],[94,241,242],[0,153,143],[224,255,102],[116,10,255],[153,0,0],[255,255,128],[255,255,0],[255,80,5]
])/256

colors75 = np.concatenate([color_alphabet, 1 - (1 - color_alphabet) / 2, color_alphabet / 2])

def colorize(x, *, bgval = None):
	le = LabelEncoder().fit(x)
	xt = le.transform(x)
	colors = colors75[np.mod(xt, 75), :]
	if bgval is not None:
		colors[x == bgval, :] = np.array([0.8, 0.8, 0.8])
	return colors


def species(ds: loompy.LoomConnection) -> str:
	if "Gene" not in ds.ra:
		return "Unknown"
	for gene, species in {
		"NOTCH2NL": "Homo sapiens",
		"Tspy1": "Rattus norvegicus",
		"Actb": "Mus musculus",  # Note must come after rat, because rat has the same gene name
		"actb1": "Danio rerio",
		"Act5C": "Drosophila melanogaster",
		"ACT1": "Saccharomyces cerevisiae",
		"act1": "Schizosaccharomyces pombe",
		"act-1": "Caenorhabditis elegans",
		"ACT12": "Arabidopsis thaliana",
		"AFTTAS": "Gallus gallus"
	}.items():
		if gene in ds.ra.Gene:
			return species
	return "Unknown"


cc_genes_human = np.array([
	'ABHD3', 'AC016205.1', 'AC073529.1', 'AC084033.3', 'AC087632.1',
	'AC091057.6', 'AC097534.2', 'AC099850.2', 'AC135586.2', 'ACAA2',
	'ACADM', 'ACP1', 'ACTL6A', 'ACYP1', 'ADCY3', 'ADD3', 'ADK', 'AHCY',
	'AKIRIN2', 'AKR7A2', 'AL359513.1', 'AL449266.1', 'AL513165.2',
	'ANAPC11', 'ANLN', 'ANP32A', 'ANP32B', 'ANP32E', 'AP001347.1',
	'APOLD1', 'ARHGAP11A', 'ARHGEF39', 'ARID1A', 'ARL6IP1', 'ARL6IP6',
	'ARMC1', 'ARPP19', 'ASF1B', 'ASPM', 'ASRGL1', 'ATAD2', 'ATAD5',
	'ATP1B3', 'AURKA', 'AURKB', 'BANF1', 'BARD1', 'BAZ1A', 'BAZ1B',
	'BIRC5', 'BLM', 'BORA', 'BRCA1', 'BRCA2', 'BRD8', 'BRIP1', 'BTG3',
	'BUB1', 'BUB1B', 'BUB3', 'C11orf58', 'C19orf48', 'C1orf112',
	'C1orf35', 'C21orf58', 'C5orf34', 'CACYBP', 'CALM2', 'CAMTA1',
	'CARHSP1', 'CBX1', 'CBX3', 'CBX5', 'CCAR1', 'CCDC14', 'CCDC167',
	'CCDC18', 'CCDC34', 'CCDC77', 'CCNA1', 'CCNA2', 'CCNB1', 'CCNB2',
	'CCNE2', 'CCNF', 'CCT4', 'CCT5', 'CDC20', 'CDC25B', 'CDC25C',
	'CDC27', 'CDC45', 'CDC6', 'CDC7', 'CDCA2', 'CDCA3', 'CDCA4',
	'CDCA7L', 'CDCA8', 'CDK1', 'CDK19', 'CDK2', 'CDK4', 'CDK5RAP2',
	'CDKAL1', 'CDKN1B', 'CDKN2C', 'CDKN3', 'CDT1', 'CENPA', 'CENPC',
	'CENPE', 'CENPF', 'CENPH', 'CENPI', 'CENPJ', 'CENPK', 'CENPL',
	'CENPM', 'CENPN', 'CENPO', 'CENPP', 'CENPQ', 'CENPU', 'CENPW',
	'CENPX', 'CEP112', 'CEP128', 'CEP135', 'CEP192', 'CEP295', 'CEP55',
	'CEP57', 'CEP57L1', 'CEP70', 'CETN3', 'CFAP20', 'CFL2', 'CGGBP1',
	'CHAF1A', 'CHCHD2', 'CHEK1', 'CHEK2', 'CHRAC1', 'CIP2A', 'CIT',
	'CKAP2', 'CKAP2L', 'CKAP5', 'CKLF', 'CKS1B', 'CKS2', 'CLSPN',
	'CMC2', 'CMSS1', 'CNIH4', 'CNN3', 'CNTLN', 'CNTRL', 'COA1',
	'COMMD4', 'COX8A', 'CSE1L', 'CTCF', 'CTDSPL2', 'CWF19L2', 'CYB5B',
	'CYCS', 'DACH1', 'DBF4', 'DBF4B', 'DBI', 'DCAF7', 'DCP2', 'DCXR',
	'DDAH2', 'DDX39A', 'DDX46', 'DEK', 'DEPDC1', 'DEPDC1B', 'DESI2',
	'DHFR', 'DIAPH3', 'DKC1', 'DLEU2', 'DLGAP5', 'DNA2', 'DNAJB1',
	'DNAJC9', 'DNMT1', 'DPM1', 'DR1', 'DSCC1', 'DSN1', 'DTL', 'DTYMK',
	'DUSP16', 'DUT', 'DYNLL1', 'DYRK1A', 'E2F3', 'E2F7', 'E2F8',
	'ECT2', 'EED', 'EEF1D', 'EID1', 'EIF1AX', 'EIF2S2', 'EIF4A3',
	'EIF4E', 'EIF5', 'EMC9', 'ENAH', 'ENO1', 'ENY2', 'ERH', 'ESCO2',
	'EWSR1', 'EXOSC8', 'EZH2', 'FAM111B', 'FAM122B', 'FAM72C',
	'FAM72D', 'FAM83D', 'FANCB', 'FANCD2', 'FANCI', 'FANCL', 'FBL',
	'FBXL5', 'FBXO5', 'FDPS', 'FDX1', 'FEN1', 'FGFR1OP', 'FILIP1L',
	'FOXM1', 'FUS', 'FUZ', 'FXR1', 'FZR1', 'G2E3', 'G3BP1', 'GABPB1',
	'GAPDH', 'GAS2L3', 'GEMIN2', 'GEN1', 'GGCT', 'GGH', 'GINS2',
	'GLO1', 'GMNN', 'GMPS', 'GNG5', 'GPBP1', 'GPSM2', 'GTSE1', 'H1FX',
	'H2AFV', 'H2AFX', 'H2AFY', 'H2AFZ', 'HACD3', 'HADH', 'HAT1',
	'HAUS1', 'HAUS6', 'HAUS8', 'HDAC2', 'HDGF', 'HELLS', 'HES1',
	'HINT1', 'HIRIP3', 'HIST1H1A', 'HIST1H1C', 'HIST1H1D', 'HIST1H2BH',
	'HIST1H4C', 'HIST2H2AC', 'HJURP', 'HMG20B', 'HMGA1', 'HMGA2',
	'HMGB1', 'HMGB2', 'HMGB3', 'HMGN1', 'HMGN2', 'HMGN3', 'HMGN5',
	'HMGXB4', 'HMMR', 'HNRNPA0', 'HNRNPA1', 'HNRNPA2B1', 'HNRNPA3',
	'HNRNPAB', 'HNRNPC', 'HNRNPD', 'HNRNPDL', 'HNRNPF', 'HNRNPH3',
	'HNRNPK', 'HNRNPLL', 'HNRNPM', 'HNRNPU', 'HNRNPUL1', 'HP1BP3',
	'HPF1', 'HSD17B11', 'HSP90AA1', 'HSP90B1', 'HSPA13', 'HSPA1B',
	'HSPB11', 'HSPD1', 'HSPE1', 'HYLS1', 'IDH2', 'IFT122', 'IGF2BP3',
	'IKBIP', 'ILF2', 'ILF3', 'ILVBL', 'IMMP1L', 'INCENP', 'IPO5',
	'IQGAP3', 'ISCA2', 'ISOC1', 'ITGAE', 'ITGB3BP', 'JADE1', 'JPT1',
	'KATNBL1', 'KCTD9', 'KIAA0586', 'KIF11', 'KIF14', 'KIF15',
	'KIF18A', 'KIF18B', 'KIF20A', 'KIF20B', 'KIF22', 'KIF23', 'KIF2C',
	'KIF4A', 'KIF5B', 'KIFC1', 'KMT5A', 'KNL1', 'KNSTRN', 'KPNA2',
	'KPNB1', 'LARP7', 'LBR', 'LCORL', 'LDHA', 'LDHB', 'LIG1', 'LIN52',
	'LINC01224', 'LINC01572', 'LMNB1', 'LMNB2', 'LRR1', 'LSM14A',
	'LSM2', 'LSM3', 'LSM4', 'LSM5', 'LSM6', 'LSM7', 'LSM8', 'LUC7L2',
	'MAD2L1', 'MAGI1', 'MAGOH', 'MAGOHB', 'MAPK1IP1L', 'MAPRE1',
	'MARCKS', 'MASTL', 'MBNL2', 'MCM10', 'MCM2', 'MCM3', 'MCM4',
	'MCM5', 'MCM7', 'MDH1', 'MED30', 'MELK', 'MGME1', 'MIS18A',
	'MIS18BP1', 'MKI67', 'MMS22L', 'MND1', 'MNS1', 'MORF4L2',
	'MPHOSPH9', 'MRE11', 'MRPL18', 'MRPL23', 'MRPL47', 'MRPL51',
	'MRPL57', 'MRPS34', 'MTFR2', 'MYBL2', 'MYEF2', 'MZT1', 'MZT2B',
	'NAA38', 'NAA50', 'NAE1', 'NAP1L1', 'NAP1L4', 'NASP', 'NCAPD2',
	'NCAPD3', 'NCAPG', 'NCAPG2', 'NCAPH', 'NCL', 'NDC1', 'NDC80',
	'NDE1', 'NDUFA6', 'NDUFAF3', 'NDUFS6', 'NEDD1', 'NEIL3', 'NEK2',
	'NELFE', 'NENF', 'NFATC3', 'NFYB', 'NIPBL', 'NMU', 'NONO', 'NOP56',
	'NOP58', 'NRDC', 'NSD2', 'NSMCE2', 'NSMCE4A', 'NUCKS1', 'NUDC',
	'NUDCD2', 'NUDT1', 'NUDT15', 'NUDT21', 'NUDT5', 'NUF2', 'NUP107',
	'NUP35', 'NUP37', 'NUP50', 'NUP54', 'NUSAP1', 'ODC1', 'ODF2',
	'OIP5', 'ORC6', 'PA2G4', 'PAICS', 'PAIP2', 'PAK4', 'PAPOLA',
	'PARP1', 'PARPBP', 'PAXX', 'PBK', 'PCBD2', 'PCBP2', 'PCM1', 'PCNA',
	'PCNP', 'PDS5B', 'PHF19', 'PHF5A', 'PHGDH', 'PHIP', 'PIF1',
	'PIMREG', 'PIN1', 'PKM', 'PLCB1', 'PLGRKT', 'PLIN3', 'PLK1',
	'PLK4', 'PMAIP1', 'PNISR', 'PNN', 'PNRC2', 'POC1A', 'POLD2',
	'POLD3', 'POLE2', 'POLQ', 'POLR2C', 'POLR2D', 'POLR2G', 'POLR2J',
	'POLR2K', 'POLR3K', 'PPIA', 'PPIG', 'PPIH', 'PPP1CC', 'PPP2R3C',
	'PPP2R5C', 'PPP6R3', 'PRC1', 'PRDX3', 'PRIM1', 'PRIM2', 'PRKDC',
	'PRPF38B', 'PRPSAP1', 'PRR11', 'PSIP1', 'PSMA3', 'PSMA4', 'PSMB2',
	'PSMB3', 'PSMC3', 'PSMC3IP', 'PSMD10', 'PSMD14', 'PSMG2', 'PSRC1',
	'PTBP1', 'PTGES3', 'PTMA', 'PTMS', 'PTTG1', 'PUF60', 'RAB8A',
	'RACGAP1', 'RAD21', 'RAD51AP1', 'RAD51B', 'RAD51C', 'RAN',
	'RANBP1', 'RANGAP1', 'RASSF1', 'RBBP4', 'RBBP8', 'RBL1', 'RBM17',
	'RBM39', 'RBM8A', 'RBMX', 'RCC1', 'RDX', 'REEP4', 'RFC1', 'RFC2',
	'RFC3', 'RFC4', 'RFWD3', 'RHEB', 'RMI2', 'RNASEH2B', 'RNASEH2C',
	'RNF138', 'RNF168', 'RNF26', 'RNPS1', 'RPA1', 'RPA3', 'RPL35',
	'RPL39L', 'RPLP0', 'RPLP1', 'RPLP2', 'RPN2', 'RPP30', 'RPS15',
	'RPS16', 'RPS20', 'RPS21', 'RPSA', 'RRM1', 'RSRC1', 'RSRC2',
	'RTKN2', 'RUVBL2', 'SAC3D1', 'SAE1', 'SAP18', 'SAPCD2', 'SCAF11',
	'SCLT1', 'SDHAF3', 'SELENOK', 'SEM1', 'SEPHS1', 'SEPT10', 'SEPT2',
	'SEPT7', 'SERBP1', 'SET', 'SF1', 'SF3B2', 'SFPQ', 'SGO1', 'SGO2',
	'SHCBP1', 'SINHCAF', 'SIVA1', 'SKA1', 'SKA2', 'SKA3', 'SLBP',
	'SLC20A1', 'SLC25A3', 'SLTM', 'SMC1A', 'SMC2', 'SMC3', 'SMC4',
	'SMC5', 'SMCHD1', 'SNAPC1', 'SNRNP25', 'SNRNP40', 'SNRNP70',
	'SNRPA', 'SNRPA1', 'SNRPB', 'SNRPC', 'SNRPD1', 'SNRPD2', 'SNRPD3',
	'SNRPE', 'SNRPF', 'SNRPG', 'SON', 'SPAG5', 'SPATA5', 'SPC25',
	'SPCS2', 'SPDL1', 'SREK1', 'SRI', 'SRP9', 'SRRM1', 'SRSF1',
	'SRSF10', 'SRSF11', 'SRSF2', 'SRSF3', 'SRSF4', 'SRSF7', 'SSB',
	'SSBP1', 'SSNA1', 'SSRP1', 'ST13', 'STAG1', 'STIL', 'STIP1',
	'STK17B', 'STK3', 'STOML2', 'SUGP2', 'SUMO1', 'SUMO3', 'SUPT16H',
	'SUV39H2', 'SUZ12', 'SYNE2', 'TACC3', 'TBC1D31', 'TBC1D5', 'TDP1',
	'TEAD1', 'TEX30', 'TFDP1', 'THRAP3', 'TICRR', 'TIMELESS', 'TIMM10',
	'TK1', 'TMED5', 'TMEM106C', 'TMEM237', 'TMEM60', 'TMEM97', 'TMPO',
	'TMSB15A', 'TOP1', 'TOP2A', 'TPI1', 'TPR', 'TPRKB', 'TPX2',
	'TRA2B', 'TRAIP', 'TROAP', 'TTC28', 'TTF2', 'TTK', 'TUBA1B',
	'TUBA1C', 'TUBB', 'TUBB4B', 'TUBG1', 'TUBGCP3', 'TXNDC12', 'TYMS',
	'UBA2', 'UBB', 'UBE2C', 'UBE2D2', 'UBE2D3', 'UBE2I', 'UBE2N',
	'UBE2S', 'UBE2T', 'UHRF1', 'UNG', 'UQCC2', 'UQCC3', 'UQCRC1',
	'UQCRFS1', 'USP1', 'VBP1', 'VDAC3', 'VEZF1', 'VRK1', 'WAPL',
	'WDHD1', 'WDPCP', 'WDR34', 'WDR76', 'XPO1', 'XRCC4', 'XRCC5',
	'XRCC6', 'YAP1', 'YBX1', 'YEATS4', 'Z94721.1', 'ZFP36L1', 'ZGRF1',
	'ZMYM1', 'ZNF22', 'ZNF367', 'ZNF43', 'ZNF704', 'ZNF83', 'ZRANB3',
	'ZSCAN16-AS1', 'ZWINT'], dtype=object)

cc_genes_mouse = np.array([x[0] + x[1:].lower() for x in cc_genes_human], dtype=object)