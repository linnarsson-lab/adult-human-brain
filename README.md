# Single-cell analysis of the adult human brain

This repository contains the code used for analysis by Siletti et al. (2022). You can also find links below to the complete dataset of 3,369,219 cells.

<img width="805" alt="image" src="https://user-images.githubusercontent.com/10656387/198325102-80260347-1bc3-4c30-91ac-f42e682cff26.png">

## Preprint (bioRxiv)

[https://www.biorxiv.org/content/10.1101/2022.10.12.511898v1](https://www.biorxiv.org/content/10.1101/2022.10.12.511898v1)

## Browser

The dataset can be browsed from [our collection](https://cellxgene.cziscience.com/collections/283d65eb-dd53-496d-adb7-7570c7caa443) at [CELLxGENE](https://cellxgene.cziscience.com/). There is one browser per dissection, and one browser per supercluster. A browser for the combined non-neuronal cells is also available (but note that some immune cells are found in the Miscellaneous supercluster).

## Data availability

### Raw sequence reads

Raw data in fastq format are available [at NeMO](http://data.nemoarchive.org/biccn/grant/u01_lein/linnarsson/transcriptome/sncell/10x_v3/human/).

BAM files were generated with [STAR aligner](https://github.com/alexdobin/STAR) version 2.7.10 using the following code:

```
    if [[ $CHEMISTRY -eq "v3" ]]; then UMIARG=" --soloUMIlen 12"; fi
    STAR --soloFeatures Gene Velocyto GeneFull --soloBarcodeReadLength 0 --limitOutSJcollapsed 50000000 \
     --outFileNamePrefix $PREFIX --genomeDir $GENOMEDIR --readFilesIn $FQGZFILE_R2 $FQGZFILE_R1 \
     --soloType CB_UMI_Simple --soloCBwhitelist $WHITELISTFILE $UMIARG \
     --soloCellFilter EmptyDrops_CR $TARGETNUMBERCELLS 0.99 10 45000 90000 500 0.01 20000 0.01 10000 \
     --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR --soloUMIdedup 1MM_CR \
     --clipAdapterType CellRanger4 --outFilterScoreMin 30 \
     --outSAMattributes NH HI nM AS CR UR CB UB GX GN sS sQ sM --outSAMtype BAM SortedByCoordinate --runThreadN 24 --readFilesCommand zcat"
```
Chemistry was "v2" for sample IDs <= 10X135_4, and v3 for these above. 
Target number of cells was 5000 for all samples with the exceptions: 10X120_6, 10X121_3, 10X203_2: 3000, 10X203_1, 10X445_3: 2000, 10X455_1, 10X455_2: 6000

Our gene and transcript annotation is based on GRCh38.p13 gencode V35 primary sequence assembly. We discarded genes or transcripts that overlapped or mapped to other genes' or non-coding RNAs' 3’ UTRs. Here we provide [the GTF file used to count reads](https://storage.googleapis.com/linnarsson-lab-human/gb_pri_annot_filtered.gtf.gz), and [the genes and transcripts that were discarded](https://storage.googleapis.com/linnarsson-lab-human/gb_pri_filtered_transcripts.txt.gz).

### Expression matrices

The final dataset is available for download at https://storage.cloud.google.com/linnarsson-lab-human. Two files are available in [loom](http://loompy.org) file format:
- Genes x cells: [adult_human_20221007.loom](https://storage.cloud.google.com/linnarsson-lab-human/adult_human_20221007.loom)
- Genes x clusters: [adult_human_20221007.agg.loom](https://storage.cloud.google.com/linnarsson-lab-human/adult_human_20221007.agg.loom)

The genes x cells dataset is alternatively available in two .h5ad files:
- Neurons: [Neurons.h5ad](https://storage.googleapis.com/linnarsson-lab-human/Neurons.h5ad)
- Non-neuronal cells: [Nonneurons.h5ad](https://storage.googleapis.com/linnarsson-lab-human/Nonneurons.h5ad)

💡**Tip:** Data for superclusters and dissections can also be downloaded from CELLxGENE in `.h5ad` (AnnData, for Scanpy) and `.rds` (for Seurat) formats by following the links to the browsers above.

Column attribute 'Tissue' corresponds to dissections in the paper, and 'ROIGroupCoarse' to the 10 regions in Fig 1C.

In addition, expression matrices generated with the "standard" cellranger + velocyto pipeline using cellranger GRCh38-3.0.0 annotations are available in [loom](https://loompy.org) and [anndata](https://anndata.readthedocs.io/en/latest/) formats:

[human_adult_GRCh38-3.0.0.loom](https://storage.googleapis.com/linnarsson-lab-human/human_adult_GRCh38-3.0.0.loom)

[human_adult-GRCh38-3.0.0.h5ad](https://storage.googleapis.com/linnarsson-lab-human/human_adult_GRCh38-3.0.0.h5ad) (Annotations basically follow [CELLxGENE](https://cellxgene.cziscience.com/) standards.)

These files contain exactly the same cells as adult_human_20221007.loom. Some ~70000 cells that were filtered out by this procedure have zero total UMI count.

### EEL Data (multiplexed RNA FISH)

The files with the molecule coordinates (as .parquet) and gene x cell counts (as .loom) are available in the EEL_adult folder at: https://storage.cloud.google.com/linnarsson-lab-human

Data in the [.parquet](https://parquet.apache.org/) format can be opened by [FISHscale](https://github.com/linnarsson-lab/FISHscale), Python [Pandas](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html) or any other Parquet reader.  
`r_px_microscope_stitched` and `c_px_microscope_stitched` contain the RNA molecule coordinates in pixels (pixel size of 0.18um).  
`r_transformed` and	`c_transformed` contain the RNA molecule coordinates in pixels (pixel size of 0.27um).  

## Code used for analysis and other output

Clustering was performed using `cytograph`. Installation and usage are described [here](https://github.com/linnarsson-lab/adult-human-brain/tree/main/cytograph). Other materials include:
- `scripts`: other scripts named in the Methods section
- `notebooks`: the code used to make figures
- `tables`: the manuscript's supplementary tables, as well as a subcluster annotation table.

 Auto-annotations are available in a separate [repository](https://github.com/linnarsson-lab/auto-annotation-ah).
