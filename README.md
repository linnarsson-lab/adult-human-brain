
# cytograph

## Requirements

`graphviz`, with the `dot` command available in the $PATH

**Note:** On Mac, graphviz must be installed from the [package](http://www.graphviz.org/pub/graphviz/stable/macos/lion/graphviz-2.40.1.pkg) 
(neither Conda or Homebrew works).

`pygraphviz`, install by `pip install pygraphviz``


## Installation

1. Install [Anaconda](https://www.continuum.io/downloads) 4.3.0, Python 3.6 version
2. `git clone https://github.com/linnarsson-lab/cytograph.git`
3. `cd bhtsne`
4. `g++ sptree.cpp tsne.cpp -o bhtsne -O2`
5. `cd ..`
3. `python setup.py install`

