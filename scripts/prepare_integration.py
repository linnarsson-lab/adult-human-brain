import loompy
import numpy as np
import pandas as pd
from cytograph.enrichment import FeatureSelectionByVariance
from cytograph.species import Species
import os


# Make file lists
human_files = [
    '/proj/human_adult/20220222/harmony/regions_clean/data/CaB.loom',
    '/proj/human_adult/20220222/harmony/regions_clean/data/Pu.loom',
    '/proj/human_adult/20220222/harmony/regions_clean/data/NAC.loom'
    ]

mouse_files = [
    'mouse_downloads/saunders_striatum.loom'
    ]

# Load gene homologues into a dictionary
# http://www.informatics.jax.org/downloads/reports/HMD_HumanPhenotype.rpt
hmlg = pd.read_csv('HMD_HumanPhenotype.rpt', delimiter='\t', header=None)
mouse_to_human = dict(zip(hmlg[2], hmlg[0]))

# To do: get rid of gene duplicates

# Add homologues to loom file
for f in mouse_files:
    with loompy.connect(f) as ds:
        # convert names from mouse to human
        gene_names = np.array([mouse_to_human.get(x, 'N/A') for x in ds.ra.Gene])
        ds.ra.HumanGene = gene_names
        unique_genes = np.unique(ds.ra.HumanGene)
        print(f'{len(unique_genes)} human genes found in the mouse dataset')
        print(f"{', '.join(list(unique_genes[:5]))}...\n")

# Get top variable genes in each file
n_genes = 2000
mask = ("cellcycle", "sex", "ieg", "mt")
# start with genes that are in both datasets
with loompy.connect(human_files[0], 'r') as hs:	
    with loompy.connect(mouse_files[0], 'r') as mm:
        human_selected = mouse_selected = np.intersect1d(hs.ra.Gene, mm.ra.HumanGene)

# find human variable genes
for f in human_files:
    with loompy.connect(f, 'r') as ds:
        selected = FeatureSelectionByVariance(n_genes, mask=Species.mask(ds, mask)).fit(ds)
        selected_genes = ds.ra.Gene[selected]
        human_selected = np.intersect1d(human_selected, selected_genes)
# find mouse variable genes
for f in mouse_files:
    with loompy.connect(f, 'r') as ds:
        selected = FeatureSelectionByVariance(n_genes, mask=Species.mask(ds, mask)).fit(ds)
        selected_genes = ds.ra.HumanGene[selected]
        mouse_selected = np.intersect1d(mouse_selected, selected_genes)
# take union
selected_genes = np.union1d(human_selected, mouse_selected)

print(f'{len(np.unique(selected_genes))} genes selected for integration')
print(f"{', '.join(list(selected_genes))}")

# Make a new matrix
new_matrices = []
CellID = []
Tissue = []
Species = []
for f in human_files:
    with loompy.connect(f, 'r') as ds:
        m = np.vstack([ds[ds.ra.Gene == g, :][0] for g in selected_genes])
        print(m.shape)
        new_matrices.append(m)
        CellID.append(ds.ca.CellID)
        Tissue.append(ds.ca.Tissue)
        Species.append(np.array(['Homo sapiens'] * ds.shape[1]))
for f in mouse_files:
    with loompy.connect(f, 'r') as ds:
        m = np.vstack([ds[ds.ra.HumanGene == g, :][0] for g in selected_genes])
        print(m.shape)
        new_matrices.append(m)
        CellID.append(ds.ca.CellID)
        Tissue.append(ds.ca.Tissue)
        Species.append(np.array(['Mus musculus'] * ds.shape[1]))

m = np.hstack(new_matrices)
row_attrs = {'Gene': selected_genes}
col_attrs = {
    'CellID': np.hstack(CellID), 
    'Tissue': np.hstack(Tissue), 
    'Species': np.hstack(Species)
    }

loompy.create('data/Striatum.loom.rerun', m, row_attrs, col_attrs)
