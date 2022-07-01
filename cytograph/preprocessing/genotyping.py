import loompy
import numpy as np
import os
import io
import pandas as pd
from scipy.io import mmread
import logging
import sys
import gzip


def read_vcf(path, compression: str = None):
    """
    reads a vcf file into a pandas dataframe
    """

    if compression is None:
        with open(path, 'r') as f:
            lines = [l for l in f if not l.startswith('##')]
    if compression == 'gzip':
        with gzip.open(path, 'r') as f:
            lines = [l.decode('UTF-8') for l in f if not l.decode('UTF-8').startswith('##')]
    return pd.read_csv(
        io.StringIO(''.join(lines)),
        dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
                'QUAL': str, 'FILTER': str, 'INFO': str},
        sep='\t'
    ).rename(columns={'#CHROM': 'CHROM'})


def genotype(ds: loompy.LoomConnection, cellsnp_path, threshold=0.95) -> None:
    """
    Counts positions in each cell that do not map to reference.
    Note that cellsnp must have been run on each sample in the file

    		Args:
			ds :				A loom connection
			cellsnp_path: 	    folder that contains cellsnp output for each sample for each donor, e.g. cellsnp output for test_sample against donor1 should be cellsnp_path/donor1/test_sample
			threshold:			percent of reads that must match reference in order to be considered ref
    """

    donor_list = ['H18_30_002', 'H19_30_001', 'H19_30_002']
    depth_attr = {donor:np.zeros(ds.shape[1]) for donor in donor_list}
    ref_attr = {donor:np.zeros(ds.shape[1]) for donor in donor_list}

    for sample in np.unique(ds.ca.SampleID):
            
        for donor in donor_list:
            
            logging.info(f"Genotyping {sample} cells against {donor}")

            # check for cellsnp files
            cellsnp_out = os.path.join(cellsnp_path, donor)
            for file in os.listdir(cellsnp_out):
                if file.startswith(sample):
                    cellsnp_out = os.path.join(cellsnp_out, file)
            if not os.path.exists(cellsnp_out):
                logging.info(f"cellsnp output does not exist for {sample}")
                sys.exit(1)

            # open donor_vcf
            donor_vcf = read_vcf(
                os.path.join(cellsnp_path, 'vcfs', donor + '.final.vcf.gz'),
                compression='gzip'
                )
            donor_vcf['HBAgenomics'] = donor_vcf['HBAgenomics'].str.split(':').str[0]

            # load cellsnp output
            cells = np.loadtxt(cellsnp_out + '/cellSNP.samples.tsv', dtype='str')
            ad = mmread(cellsnp_out + '/cellSNP.tag.AD.mtx')
            dp = mmread(cellsnp_out + '/cellSNP.tag.DP.mtx')

            # filter for homozygous positions
            counted = read_vcf(cellsnp_out + '/cellSNP.base.vcf')
            counted_genome = counted.merge(donor_vcf,  how='inner', left_on=["CHROM", "POS"], right_on = ["CHROM", "POS"])
            hom = counted_genome['HBAgenomics'] == '1/1'
            logging.info(f"{hom.sum()} homozygous alternate positions")

            ad = ad.tocsr()[hom].A
            dp = dp.tocsr()[hom].A
            f = ad / dp
            logging.info(f.shape)

            # calc homozygous alt depth and % ref
            # find corresponding cell ID in cellSNP output
            cell_attr = pd.Series(ds.ca.CellID[ds.ca.SampleID == sample])
            cell_attr = cell_attr.str.split(':').str[1].to_numpy()
            ix = np.array([np.where(x == cells)[0][0] for x in cell_attr])
            # count genotype positions
            depth = np.count_nonzero(dp[:, ix], axis=0)
            # count fewer than threshold of reads is alt
            f = ad[:, ix] / dp[:, ix]
            ref = np.count_nonzero(f <= threshold, axis=0)

            depth_attr[donor][ds.ca.SampleID == sample] = depth
            ref_attr[donor][ds.ca.SampleID == sample] = ref

    for donor in donor_list:
        ds.ca[f'GenotypingDepth_{donor}'] = depth_attr[donor]
        ds.ca[f'GenotypingRef_{donor}'] = ref_attr[donor]

    return


