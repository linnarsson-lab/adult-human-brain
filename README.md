
# cytograph

## Requirements

`graphviz`, with the `dot` command available in the $PATH

**Note:** On Mac, graphviz must be installed from the [package](http://www.graphviz.org/pub/graphviz/stable/macos/lion/graphviz-2.40.1.pkg) 
(neither Conda or Homebrew works).

`pygraphviz`, install by `pip install pygraphviz``


## Installation

The following instructions should work for Linux and Mac (unfortunately, we have no 
experience with Windows).

1. [Install Anaconda 4.3.0](https://www.continuum.io/downloads), Python 3.6 version

2. Install `loompy``:

```
git clone https://github.com/linnarsson-lab/Loom.git
cd Loom/python
python setup.py install
```

3. Install `cytograph`:

```
git clone https://github.com/linnarsson-lab/cytograph.git
cd cytograph/bhtsne
g++ sptree.cpp tsne.cpp -o bhtsne -O2
cd ..
python setup.py install
```
