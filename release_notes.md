## Major changes

* Everything organized in subfolders, but still exported at the top level of the module (except as noted)
* Plots not exported at top level, and renamed removing "plot_" prefix, so you need to do "import cytograph.plotting as pp" and then "pp.manifold(...)" etc.
* aggregate_loom is now a new method on the LoomConnection, named "aggregate". It returns a view, not a matrix, so you need to do "ds.aggregate(...)[:, :]" to get the matrix now. It no longer takes a "return_matrix" param
* Utilities for running the cytograph pipeline are not imported at the top level (incl the Cytograph2 and Aggregate classes), but are in the pipeline namespace, so you need to do "from cytograph.pipeline import Cytograph2"
* fit_gamma() renamed to velocity_gamma()


## TODO

* Integrate with luigi to run the pipeline automatically from config files
* Create a robust "collect" step to collect samples into a joint loom, with QC and doublet removal
* Integrate auto-annotation and the punchcard mechanism
* Create a command-line tool "cytograph" for running the pipeline
* Lift the plotting functions from loompy
* Make a better config system, for all aspects of configuration (per build, per set of builds, per species, general settings)
* Make species config files, with sensible defaults for human and mouse
* A general Mask class, to mask by named category ("sex", "ieg", "cell_cycle") or by bool array, or a combination; make all methods that use a mask accept it

* Make it easy to run a single sample without any assumptions 
* Make it easy to progressively increase the complexity with auto-annotations and punchcards
* Make it as easy to run on clusters as on a laptop


