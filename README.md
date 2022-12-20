# Single-cell analysis of the adult human brain

This repository contains the code used for analysis by Siletti et al. (2022). You can also find links below to the complete dataset of 3,369,219 cells.

<img width="805" alt="image" src="https://user-images.githubusercontent.com/10656387/198325102-80260347-1bc3-4c30-91ac-f42e682cff26.png">

## Preprint (bioRxiv)

[https://www.biorxiv.org/content/10.1101/2022.10.12.511898v1](https://www.biorxiv.org/content/10.1101/2022.10.12.511898v1)

## Browser

The dataset can be browsed at [CELLxGENE](https://cellxgene.cziscience.com/collections/283d65eb-dd53-496d-adb7-7570c7caa443) ([more information](https://cellxgene.cziscience.com/)). There is one browser per dissection, and one browser per supercluster. A browser for the combined non-neuronal cells is also available (but note that some immune cells are found in the Miscellaneous supercluster).

## Data availability

The final dataset is available for download at https://storage.cloud.google.com/linnarsson-lab-human. Two files are available:
- Genes x cells: [adult_human_20221007.loom](https://storage.cloud.google.com/linnarsson-lab-human/adult_human_20221007.loom)
- Genes x clusters: [adult_human_20221007.agg.loom](https://storage.cloud.google.com/linnarsson-lab-human/adult_human_20221007.agg.loom)

These files use the [loom](http://loompy.org) file format. 

Note that [adult_human_20221007.loom](https://storage.cloud.google.com/linnarsson-lab-human/adult_human_20221007.loom) contains both "Cluster" and "Subcluster" attributes that correspond to the 461 clusters and 3313 subclusters described in the paper. The loom file additionally contains the attributes "Roi" and "ROIGroupCoarse" that correspond to "dissections" and "regions" in the paper, respectively.

## EEL Data

The files with the molecule coordinates (as .parquet) and gene x cell counts (as .loom) are available in the EEL_adult folder at: https://storage.cloud.google.com/linnarsson-lab-human

Data in the [.parquet](https://parquet.apache.org/) format and can be opened by [FISHscale](https://github.com/linnarsson-lab/FISHscale), Python [Pandas](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html) or any other Parquet reader.  
`r_px_microscope_stitched` and `c_px_microscope_stitched` contain the RNA molecule coordinates in pixels (pixel size of 0.18um).  
`r_transformed` and	`c_transformed` contain the RNA molecule coordinates in pixels (pixel size of 0.27um).  

## Code used for analysis

Clustering was performed using `cytograph`. Installation and usage are described [here](https://github.com/linnarsson-lab/adult-human-brain/tree/main/cytograph). Other materials include:
- `scripts`: other scripts named in the Methods section
- `notebooks`: some of the code used to make figures
- `tables`: the sample-metadata and cluster-annotation tables (Tables S1 and S2). Note that Tables S3 and S4 correspond to "de_oligos.csv" and "de_opcs.csv" in the "notebooks" folder.

 Auto-annotations are available in a separate [repository](https://github.com/linnarsson-lab/auto-annotation-ah).
