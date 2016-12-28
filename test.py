
#%%
import os
import numpy as np
import loompy
import differentiation_topology as dt
import matplotlib.pyplot as plt

config = {
    "sample_dir": "/Users/Sten/loom-datasets/Whole brain/",
    "build_dir": "/Users/Sten/builds/build_20161127_180745/",
    "tissue": "Dentate gyrus",
    "samples": ["10X43_1", "10X46_1"],

    "preprocessing": {
        "do_validate_genes": True,
        "make_doublets": False
    },
    "knn": {
        "k": 50,
        "n_trees": 50,
        "mutual": True,
        "min_cells": 10
    },
    "louvain_jaccard": {
        "cache_n_columns": 5000,
        "n_components": 50,
        "n_genes": 2000,
        "normalize": True,
        "standardize": False
    },
    "prommt": {
        "n_genes": 1000,
        "n_S": 100,
        "k": 5,
        "max_iter": 100
    },
    "annotation": {
        "pep": 0.05,
        "f": 0.2,
        "annotation_root": "/Users/Sten/Dropbox (Linnarsson Group)/Code/autoannotation/"
    }
}
result = dt.cytograph(config)