# Single-cell analysis of the adult human brain

This repository contains the code used for analysis by Siletti et al. (2022). You can also find links below to the complete dataset of 3,369,219 cells.

## Data availability

The final dataset is available for download at https://storage.cloud.google.com/linnarsson-lab-human. Two files are available:
- Genes x cells: [adult_human_20221007.loom](https://storage.cloud.google.com/linnarsson-lab-human/adult_human_20221007.loom)
- Genes x clusters: [adult_human_20221007.agg.loom](https://storage.cloud.google.com/linnarsson-lab-human/adult_human_20221007.agg.loom)

Note that [adult_human_20221007.loom](https://storage.cloud.google.com/linnarsson-lab-human/adult_human_20221007.loom) contains both "Cluster" and "Subcluster" attributes that correspond to the 461 clusters and 3313 subclusters described in the paper. The loom file additionally contains the attributes "Roi" and "ROIGroupCoarse" that correspond to "dissections" and "regions" in the paper, respectively.

## Code used for analysis

Clustering was performed using cytograph as described [here](https://github.com/linnarsson-lab/adult-human-brain/tree/main/cytograph). Other scripts named in the Methods section can be found in the "scripts" folder. Notebooks used to make figures are found in "notebooks."
