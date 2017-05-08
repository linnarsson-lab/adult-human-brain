
# cytograph

## Installation

The following instructions should work for Linux and Mac (unfortunately, we have no 
experience with Windows).

1. [Install Anaconda 4.3.0](https://www.continuum.io/downloads), Python 3.6 version

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
g++ sptree.cpp tsne.cpp -o bhtsne -O2
cd ..
python setup.py install
```

## Preparations

1. Download the raw data `http://.... link here`
2. Unpack it to the folder `loom_samples`
3. Create the build folder: `mkdir loom_builds`

## Pipeline examples

* Level 2 analysis for specific class and tissue

`luigi --local-scheduler --module cytograph PlotGraphL2 --major-class Oligos --tissue All`

Replace `Oligos` with any major class (`Oligos`, `Astrocyte`, `Neurons`, `Ependymal`, `Immune`, 
`Vascular`, `Cycling`, `Erythrocytes`).

To split by tissue, replace `All` with the tissue name (e.g. `Hippocampus`)

* Redo auto-annotation

Delete the {major-class}.aa.tab and {major-class}.mknn.png files, then rerun the command above.
