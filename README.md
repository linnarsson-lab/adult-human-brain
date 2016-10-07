
# cello

## Installation

1. Install [Anaconda](https://www.continuum.io/downloads)
2. Clone this repository
3. Run `python setup.py install`

## Game plan

### Starting point

The starting point is a collection of loom files, one per Chromium sample. No preprocessing other than that provided
by the cellranger pipeline has been performed.

### Cell validation

Cell validation is performed as detailed below, and the result is stored as a new column attribute `Valid` which
takes the values `1` (the cell passed) or `0` (the cell failed). No other changes are made to the Loom file.

For each Loom file individually:

1. If the total number of cells detected exceeds 150% of the expected, mark all cells as failed
2. For each cell, mark the cell failed if 
    
    2.1 The fraction mitochondrial genes exceeds 10%
    2.2 The total number of molecules detected is less than 500

3. Mark all other cells as passed


### Preprocessing of genes

Five classes of genes were treated specially: the olfactory receptors, the vomeronasal receptors, the clustered protocadherins, 
and the sex-specific genes (male and female).

In each case, a metagene was created by summing the expression of all genes in the class, and this metagene was used in place
of the original genes for the purposes of clustering, correlation etc. The original genes were retained for purposes of
differential gene expression analysis etc.

The metagenes were denoted OrMeta, VmnrMeta, PcdhMeta, MaleMeta and FemaleMeta.

In the Loom files, this was accomplished by creating new modified files having five new rows (the metagenes).

### Initial feature selection

Calculate the top n noisiest genes and use those for the initial iteration

### Calculate diffusion map (transition matrix)

Use the kNN approximation with `annoy`, but link disjoint components using uniform sampling of long distances.

I.e. after making kNN graph, find connected components. For each pair of connected components, sample n pairs of cells
uniformly, and add them to the kNN graph. These pairs will get low transition probabilities, but they will be non-zero 
and thus make the graph fully connected, ensuring diffusion to all nodes.


### Compute diffusion pseudotime from root

Use essentially the approach in the DPT paper. Consider what might be a good root node:

* Average or single ES cells (might be bad if it's cycling)
* Average of all cells
* Synthetic cell expressing only common genes
* Earliest possible neuroepithelial cells

### Skeletonize using the mapper algorithm (TDA)

Slice the cells by pseudotime, in overlapping slices. On first iteration, use wide slices, then narrow them.

Cluster each slice using e.g. affinity propagation to obtain individual components
(or simply use connected kNN components?)

### Refine local distances and restart

For each cluster:

1. Perform gene selection
2. Compute full pairwise distance matrix
3. Recalculate kNN for each cell

Go back to calculating the diffusion map using this refined kNN graph. Do this any number of times.

### Construct the graph

Remove the long-range links that were added to link disjoint components. Lay out each component separately.

Make a skeleton graph by creating a node for each cluster, and linking nodes where clusters overlap.

Lay out the skeleton graph using t-SNE on the medoids and (skeleton-based?) edge bundling. Edge lengths are 
set equal to the pseudotime interval they contain.

Place each cell along the linear (edge) dimension according to its pseudotime position.

Place each cell randomly to the left or right of the edge, at a distance inversely related to its nearest
neighbour-distance. This ensures that dense regions.


### Estimate gene expression on the pseudotime graph

Use a moving kernel along the pseudotime to detect significant changes in expression? As a result, 
you would have intervals of gene expression. Disadvantage is that it is a purely local measure
so will suffer from noise.

Maybe an EM algorithm to place breakpoints iteratively until the model no longer improves?

