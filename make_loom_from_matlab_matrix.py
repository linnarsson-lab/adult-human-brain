#!python3
import sys, os.path
import pandas, numpy
import loompy

dataset_path = "/srv/datasets-private" # On loom-server
title = tsne_path = ""
project = "Other"

if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
	print ("Usage:\npython3 make_loom_from_matlab_matrix.py [OPTIONS] LOOMFILE MOLDATAPATH CELLANNOTPATH")
	print ("  --project=PROJECT           The loom browser project folder [default=" + project + "]")
	print ("  --tsne=TSNEPATH             File with tSNE coordinates: Lines of [cellid TAB x TAB y] in same cell order as matlab files")
	print ("  --title=TITLE               Title of the dataset to use in loom browser")
	print ("  --dataset-path=PATH         Path to dataset folder in loom browser [default=" + dataset_path + "]")
	sys.exit(0)

i = 1
while i < len(sys.argv) and sys.argv[i].startswith("-"):
	if sys.argv[i].startswith("--tsne="):
		tsne_path = sys.argv[i][7:]
	elif sys.argv[i].startswith("--title="):
		title = sys.argv[i][8:]
	elif sys.argv[i].startswith("--project="):
		project = sys.argv[i][10:]
	elif sys.argv[i].startswith("--dataset-path="):
		dataset_path = sys.argv[i][15:]
	i += 1

outfile, moldata_path, cellannot_path = sys.argv[i:]
# Example input from sanger://data/seq/HannahDentateGyrus/:
#outfile = "dentate_gyrus_c1.loom"
#moldata_path = "moldata_C1_data_DG_no_annot_01-Mar-2017.txt"
#cellannot_path = "cell_annotation_C1_data_DG_01-Mar-2017.txt"
#tsne_path = "tsne_coordinates_C1_finalClsuter_perplexity_60_Ngenes=526_NPCA=40_23-Feb-2017.txt"
#title = "dentate_gyrus_c1"

mol_df = pandas.read_csv(moldata_path, header=0, index_col=0)
matrix = mol_df.as_matrix()

if not title:
	title = os.path.basename(outfile).replace(".loom", "")
file_attrs = { "matrix_path": moldata_path, "annotation_path": cellannot_path, "title": title, "project": project }

row_attrs = {}
row_attrs['Gene'] = numpy.array(mol_df.index, dtype='str')

cellannot_df = pandas.read_csv(cellannot_path, header=None, delimiter='\t', index_col=0)
col_attrs = {}
for col_attr, row in cellannot_df.iterrows():
	valuetype = 'float64'
	try:
		for value in row:
			tmp = float(value)
	except ValueError:
		valuetype = 'str'
	row_array = numpy.array(row, dtype=valuetype)
	if col_attr.lower() in ("cellid", "cell_id"):
		col_attrs["CellID"] = row_array
	elif col_attr.strip().lower().replace(" ", "").replace("_", "") in ("clustername", "cluster", "class", "classname", "celltype", "clusters"):
		col_attrs["Cluster"] = row_array
	else:
		col_attrs[col_attr] = row_array

if not "Cluster" in col_attrs:
	print ("Warning: Could not find a cluster attribute in %s" % cellannot_path)
if not "CellID" in col_attrs:
	print ("Error: There is no attribute 'cellid' in %s" % cellannot_path)
	sys.exit(1)
if not "_Total" in col_attrs and not "Total" in col_attrs:
	total = matrix.sum(0)
	col_attrs["_Total"] = total

if tsne_path:
	tsne_df = pandas.read_csv(tsne_path, header=None, delimiter='\t').T
	if len(tsne_df.iloc[0].values) != len(col_attrs['CellID']):
		print ("ERROR: # of CellIDs differ between %s and %s" % (tsne_path, cellannot_path))
		sys.exit(1)
	if not numpy.all(tsne_df.iloc[0].values == col_attrs['CellID']):
		print ("ERROR: The CellIDs in %s have to be in same order as in %s" % (tsne_path, cellannot_path))
		sys.exit(1)
	if tsne_df.shape[0] != 3:
		print ("ERROR: %s should have exactly 3 columns: (CellID, x_tSNE, and y_tSNE)" % tsne_path)
		sys.exit(1)
	col_attrs['_tSNE1'] = numpy.array(tsne_df.iloc[1].values, dtype='float64')
	col_attrs['_tSNE2'] = numpy.array(tsne_df.iloc[2].values, dtype='float64')

print ("Creating .loom file " + outfile + "...")
l = loompy.create(outfile, matrix, row_attrs, col_attrs, file_attrs)

outname = os.path.basename(outfile)
print ("Now you may upload and prepare for loom-viewer by pasting this on the command line (requires access to GCloud):")
print ("gcloud compute copy-files %s loom:%s/%s/ --zone us-central1-a" % (outfile, dataset_path, project))
print ("gcloud compute ssh loom --zone us-central1-a --command 'loom --dataset-path %s tile %s'" % (dataset_path, outname))
print ("gcloud compute ssh loom --zone us-central1-a --command 'loom --dataset-path %s expand -r %s'" % (dataset_path, outname))

