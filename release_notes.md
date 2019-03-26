## Major changes

* Everything organized in subfolders, but still exported at the top level of the module (except as noted)
* Plots not exported at top level, and renamed removing "plot_" prefix, so you need to do "import cytograph.plotting as cplt" and then "cplt.manifold(...)" etc.
* aggregate_loom is now a new method on the LoomConnection, named "aggregate". It returns a view, not a matrix, so you need to do "ds.aggregate(...)[:, :]" to get the matrix now. It no longer takes a "return_matrix" param
* Utilities for running the cytograph pipeline are not imported at the top level (incl the Cytograph2 and Aggregate classes), but are in the pipeline namespace, so you need to do "from cytograph.pipeline import Cytograph2"
* fit_gamma() renamed to velocity_gamma()


## TODO

* Integrate with luigi to run the pipeline automatically from config files
* Create a robust "collect" step to collect samples into a joint loom, with QC and doublet removal
* Integrate auto-annotation and the punchcard mechanism
