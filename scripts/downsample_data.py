import loompy
import numpy as np
import pandas as pd

# Data was collected below and then run with Cytograph as described in the manuscript


## DOWNSAMPLE BY CLUSTER ##

neuron_file = '/proj/human_adult/20220222/harmony/new_embeddings/data/harmony_A.loom'

for n_cells in [10, 100, 1000]:

	print(n_cells)

	# get selected cells
	with loompy.connect(neuron_file, 'r') as ds:
		# downsample full dataset
		cell_selection = []
		for c in np.unique(ds.ca.Clusters):
			cells = np.where(ds.ca.Clusters == c)[0]
			if len(cells) > n_cells:
				cells = np.random.choice(cells, n_cells, replace=False)
			cell_selection.append(cells)
		cell_selection = np.hstack(cell_selection)
		cell_selection = np.isin(np.arange(ds.shape[1]), cell_selection)

	print(len(cell_selection))
	print(cell_selection)

	# make new file
	print("combine_faster")
	loompy.combine_faster([neuron_file], f'data/Downsampled{n_cells}.loom.rerun', selections=[cell_selection])


## USE SPLATTER VARIABLE GENES ##

with loompy.connect('/proj/human_adult/20220222/harmony/paris_top_bug/data/harmony_A_A.loom', 'r') as ds:
	splatter_variable = ds.ra.Gene[ds.ra.Selected == 1]

for subset in ['Downsampled10', 'Downsampled100', 'Downsampled1000']:	

	print(subset)
	loom_file = f'data/{subset}.loom'

	with loompy.connect(loom_file, 'r') as ds:

		selected_genes = np.isin(ds.ra.Gene, splatter_variable)
		m = ds.sparse(rows=selected_genes)
		print(m.shape)
		row_attrs = {k:v[selected_genes] for k, v in ds.ra.items()}
		col_attrs = {k:v for k, v in ds.ca.items()}

		loompy.create(f'data/{subset}SplatterVar.loom.rerun', m, row_attrs, col_attrs)


## USE RANDOM GENES ##

for subset in ['Downsampled10', 'Downsampled100']:	

	print(subset)
	loom_file = f'data/{subset}.loom'

	with loompy.connect(loom_file, 'r') as ds:

		valid_genes = np.where(ds.ra.Valid == 1)[0]
		selected_genes = np.random.choice(valid_genes, 2000, replace=False)
		selected_genes = np.sort(selected_genes)
		m = ds.sparse(rows=selected_genes)
		print(m.shape)
		row_attrs = {k:v[selected_genes] for k, v in ds.ra.items()}
		col_attrs = {k:v for k, v in ds.ca.items()}

		loompy.create(f'data/{subset}Random.loom.rerun', m, row_attrs, col_attrs)


## REANALYZE AND NEURONS (AND SPLATTER NEURONS) FROM EACH DISSECTION ##

def clean_roi(roi_attr):
    return pd.Series(roi_attr).replace({r'[^\x00-\x7F]+':''}, regex=True).to_numpy()

with loompy.connect('/proj/human_adult/20220222/harmony/paris_top_bug/data/Pool.loom', 'r') as ds:
    all_dissections = np.unique(clean_roi(ds.ca.Roi))
    all_dissections = pd.Series(all_dissections).str.replace('Human ', '').to_numpy()
    cell_dict = dict(zip(ds.ca.CellID, ds.ca.Punchcard))
    print(len(all_dissections))

with open('roi_list.txt', 'w') as f:
		
	for dis in all_dissections:

		filename = f'/proj/human_adult/20220222/harmony/regions_clean/data/{dis}.loom'

		# get selected cells
		with loompy.connect(filename, 'r') as ds:
			
			punchcard_attr = np.array([cell_dict.get(x) for x in ds.ca.CellID])
			cell_selection = pd.Series(punchcard_attr).str.startswith('harmony_A_').to_numpy()
			print(f'cell_selection.sum() cells')

		# make new file
		print(f"Combining {dis}")
		loompy.combine_faster([filename], f'data/Dis-{dis}.loom.rerun', selections=[cell_selection])
		f.write(f'Dis-{dis}/n')

		cell_selection = punchcard_attr == 'harmony_A_A'
		if cell_selection.sum() > 5000:
			# make new file
			print(f"Combining {dis}Splatter")
			loompy.combine_faster([filename], f'data/Dis-Splatter{dis}.loom.rerun', selections=[cell_selection])
			f.write(f'Dis-Splatter{dis}/n')
