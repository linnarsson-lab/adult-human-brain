
# cytograph

## Installation

The following instructions should work for Linux and Mac (unfortunately, we have no 
experience with Windows).

1. [Install Anaconda](https://www.continuum.io/downloads), Python 3.7 version

2. Install [loompy](http://loompy.org)

3. Install `cytograph`:

```
git clone https://github.com/linnarsson-lab/cytograph-dev.git
cd cytograph-dev
pip install -e .
```

## Creating a build

An analysis in cytograph is called a "build" and is driven by configurations (settings, such as parameters to the algorithms) and punchcards (text files that define how samples are combined in the analysis). To create a build, you must first create a
*build folder* with the proper config and punchcards, and then execute the build. 

### Preparations

1. Prepare your input samples using [velocyto](http://velocyto.org) and place them in a folder. Below, the full path to this folder will be called `{samples}`. Each sample file should be named `{SampleID}.loom`. For example, `lung23.loom` is a sample with `SampleID` equal to `lung23`.

2. Prepare auto-annotations in a folder (we'll call its full path `{auto-annotations}`). For example, you can start with `Human` or `Mouse` from the [auto-annotation](https://github.com/linnarsson-lab/auto-annotation) repository.

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
Poisson pooling, matrix factorization, manifold learning, clustering, embedding and velocity inference).