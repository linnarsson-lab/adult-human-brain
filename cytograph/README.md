
# cytograph

## Installation

The following instructions should work for Linux and Mac. We have no 
experience with Windows.

1. [Install Anaconda](https://www.continuum.io/downloads)

2. Create a new conda environment and install `cytograph` as below:

```
git clone https://github.com/linnarsson-lab/adult-human-brain.git
cd adult-human-brain
conda create -n cgenv python==3.9.12 h5py==2.10.0
conda activate cgenv
pip install -e .
```

## Creating a build

An analysis in cytograph is called a "build" and is driven by configurations (settings, such as parameters to the algorithms) and punchcards (text files that define how samples are combined in the analysis). To create a build, you must first create a
*build folder* with the proper config and punchcards, and then execute the build. 

### Preparations

1. Prepare your input samples and place them in a folder. Below, the full path to this folder will be called `{samples}`. Each sample file should be named `{SampleID}.loom`. For example, `lung23.loom` is a sample with `SampleID` equal to `lung23`.

2. Prepare auto-annotations in a folder (we'll call its full path `{auto-annotations}`). For example, you can start with `Human_adult` from this [auto-annotation](https://github.com/linnarsson-lab/auto-annotation-ah) repository.

3. Optionally create a metadata file (we'll call its full path `{metadata}`). The file should be semicolon-delimited with one header row, and one column heading should be `SampleID`. For each sample included in the build, the SampleID is matched against this column, and the corresponding row of metadata is added to the sample. For example, if the SampleID is `lung23`, then the metadata row where the `SampleID` column has value `lung23` is used. For each column, an attribute is added to the file with name equal to the column heading, and value equal to the value in the corresponding row. 

4. Optionally create a settings file `.cytograph` in your home directory, with content like this:

```
paths:
  samples: "{samples}"
  autoannotation: "{auto-annotation}"
  metadata: "{metadata}"
```

For example:

```
paths:
  samples: "/proj/loom"
  autoannotation: "/home/sten/proj/auto-annotation/Human"
  metadata: "/home/sten/proj/samples_csv.txt"
```

### Setting up the build folder

1. Create a build folder (full path `{build}`), and in this folder create a subfolder `punchcards`. 

2. Optionally create a file in the build folder called `config.yaml`, with content similar to the global config file (`~/.cytograph`). Any setting given in the build-specific config file will override the corresponding
    setting in the global config. For example, to use a different auto-annotation for a build, `config.yaml` would be give like so:

```
paths:
  autoannotation: "/home/sten/proj/auto-annotation/Mouse"
```

3. Create a root punchcard defintion named `Root.yaml` in the `punchcards` subfolder, with the following content:

```
MySamples:
    include: [[10X121_1, 10X122_2]]
```

Here, `MySamples` is the name that you give to a collection of samples, called a *Punchcard subset*. The samples are listes in double brackets using their SampleIDs. Cytograph will collect the samples by reading from `{samples}/10X121_1.loom` and `{samples}/10X122_2.loom`
and will create an output file called `MySamples.loom`. Note that you can define multiple subsets, which you can then analyze separately.

Your build folder should look now like this:

```
/punchcards
    Root.yaml  # root punchcard, which defines the samples to include in the build
config.yaml    # optional build-specific config
```

### Running the pipeline

Run the following command from the terminal:

```
$ cytograph --build-location {build} process MySamples
```

This tells cytograph to process the Punchcard subset `MySamples`, which we defined in `Root.yaml`. Cytograph will run through a standard set of algorithms (pooling the samples, removing doublets,
Poisson pooling, matrix factorization, manifold learning, clustering, embedding and velocity inference). Several outputs will be produced, resulting in the following build folder:

```
/data
    MySamples.loom
    MySamples.agg.loom
/exported
    MySamples/
        .
        . Lots of plots
        .
/punchcards
    Root.yaml  # root punchcard, which defines the samples to include in the build
config.yaml    # optional build-specific config

```


### Splitting the data, and running a complete build

The command in the previous section runs a single punchcard subset (`MySamples`). In this section, we will sequentially split the analysis and run a complete set of punchcards (a "deck") both locally and on a compute cluster. Two punchcard fields can be used to split:

1. `include` splits a dataset according to auto-annotation. Start by adding another punchcard, named `MySamples.yaml`, with the following content:

```
Microglia:
  include: [MGL]
Astrocytes:
  include: [ASTRO]
Neurons:
  include: []
```

The name of the punchcard (`MySamples.yaml`) indicates that this punchcard takes its cells from the `MySamples` subset in `Root.yaml`. The name is case-sensitive.

Each section in the punchcard (e.g. `Microglia`) defines a new subset. However, this time we are not getting cells from raw samples, but rather selecting cells using auto-annotation, and taking them from the parent punchcard (i.e. `MySamples.loom`). The statement `include: [MGL]` means *include all clusters that were annotated with `MGL`*. 
Note that subsets are evaluated in order, and cells are assigned to the first subset they match. The empty subset `[]` is special; it means 
*include all cells that have not yet been included*. In the case above, any cell that was not assigned to `Microglia` or `Astrocytes` is assigned to `Neurons`.

2. `onlyif ` splits a dataset according to a Boolean expression. The expressions must refer to an attribute in the loom file. For example, the punchcard below would remove all cells with fewer than 1000 total unique molecular identifiers (UMIs) if total UMI counts were stored in the attribute "TotalUMI." 

```
GoodCells:
  include: []
  onlyif: TotalUMI < 1000
```

Note that auto-annotations and Boolean contributions may be used in combination. 

```
GoodMicroglia:
  include: [MGL]
  onlyif: TotalUMI < 1000
```

#### Processing the new subsets

Now we have two punchcards that three subsets (`MySamples` in `Root.yaml`, and `Microglia`, `Astrocytes`, `Neurons` in `MySamples.yaml`). 

If you have already processed `MySamples` as above and later added the `MySamples.yaml` punchcard, first run:

```
cytograph subset MySamples
```

and then process the subsets by running:

```
cytograph process MySamples_Microglia
```

(Note: if you run `cytograph process {subset}` in the build folder, you can omit the `--build-location` parameter as we did here.)

Each subset is named by the sequence of punchcards analyzed to reach it (but 
omitting `Root`). Here the new subsets will be called `MySamples_Microglia`, `MySamples_Astrocytes`, and `MySamples_Neurons`.

#### Running a complete build automatically

It gets tedious to run all these subsets one by one using `cytograph process {subset}`. Cytograph therefore has a `build`
command that automates the process of figuring out what all the punchcards are, and how the subsets defined in them depend on each other:

```
cytograph build --engine local
```

Cytograph computes an execution graph based on the punchcard dependencies, sorts it so that dependencies come before the
subsets that depends on them, and then runs `cytograph process` on them sequentially.

#### Running a complete build on a compute cluster

Finally, instead of running the build locally and sequentially, you can run it on a cluster and in parallel. Simply change
the engine:

```
cytograph build --engine condor
```

This will use DAGman to run the build according to the dependency graph. Log files and outputs from the individual steps
are saved in `condor/` in the build folder, which now looks like this:

```
/condor
    _dag.condor
    MySamples_Astrocytes.condor
    MySamples_Microglia.condor
    MySamples.condor
    MySamples_Neurons.condor
    Pool.condor
    ...log files etc...
/data
    MySamples.loom
    MySamples.agg.loom
/exported
    MySamples/
        .
        . Lots of plots
        .
    Pool/
        ...plots...
    ...more folders...
/punchcards
    Root.yaml  # root punchcard, which defines the samples to include in the build
config.yaml    # optional build-specific config
```


