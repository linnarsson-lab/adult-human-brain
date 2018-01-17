
# cytograph

## Installation

The following instructions should work for Linux and Mac (unfortunately, we have no 
experience with Windows).

1. [Install Anaconda](https://www.continuum.io/downloads), Python 3.6 version

2. Install `loompy`:

```
git clone https://github.com/linnarsson-lab/Loom.git
cd Loom/python
python setup.py install
```

3. Install `cytograph`:

```
git clone https://github.com/linnarsson-lab/cytograph.git
cd cytograph/bhtsne
g++ sptree.cpp tsne.cpp -o bh_tsne -O2
cd ..
python setup.py install
```

Make sure `bhtsne` is in your $PATH

## Running the pipeline

**Note:** pipelines are now separated out into their own repos:

* [Adolescent mouse brain](https://github.com/linnarsson-lab/adolescent-mouse)
* [Developing mouse brain](https://github.com/linnarsson-lab/development-mouse)

