#!/usr/bin/env python
import os, glob, re, sys
import sqlite3
from subprocess import run, PIPE, DEVNULL
import loompy

local_sample_dir = "/data/proj/chromium/"
gs_sample_dir = "gs://linnarsson-lab-chromium/"
sqlite3_db_file = "/mnt/sanger-data/10X/DB/sqlite3_chromium.db"
velocyto_ivl_dir = "/data/proj/chromium/intervals"
ivl_path_pat = "/data/proj/chromium/intervals/*_gene_ivls.txt"
loom_server_upload_dir = "/home/peterl/loom-datasets-private/Other/"
loom_server_upload_cmd = "gcloud compute copy-files %s loom:" + loom_server_upload_dir + " --zone us-central1-a"
loom_server_cmd_pat = "gcloud compute ssh loom --zone us-central1-a --command '/home/ubuntu/anaconda3/bin/python /home/ubuntu/anaconda3/bin/loom %s >> loom_tile_expand.log 2>&1'"

class OutsProcessorProps:
	def __init__(self):
		self.force_loom = False
		self.never_overwrite_loom = False
		self.overwrite_loom = True
		self.use_velocyto = False
		self.upload_to_loom_server = False

class OutsProcessor:
	def __init__(self, props = None):
		if props is None:
			props = OutsProcessorProps()
		self.props = props

	def process_dir(self, d, sampleid, globalattrs = None):
		outsprefix = os.path.join(d, "outs") + os.sep
		gsprefix = gs_sample_dir + sampleid + "/"
		loom_file = os.path.join(d, sampleid + ".loom")
		if self.props.force_loom or not os.path.exists(loom_file):
			loom_made = False
			if self.props.use_velocyto:
				tr = globalattrs['transcriptome']
				ivlfile = ivl_path_pat.replace('*', tr)
				print("Making loom file with velocyto using " + ivlfile)
				cmd = ["velocyto", "run10x", "--outputfolder", d, d, ivlfile]
				errcode = run(cmd).returncode # Should return outputfile, or allow passing  it
				if errcode == 0:
					loom_made = True
			if not loom_made:
				print("Making loom file with loompy")
				loompy.create_from_cellranger(d)
			if globalattrs == None:
				globalattrs = {}
			xa = []
			for attr in ("species", "sex", "tissue", "age"):
				if attr in globalattrs: xa.append( globalattrs[attr] )
			xa = " (" + ",".join(xa) + ")" if len(xa) > 0 else ""
			if "title" not in globalattrs:
				globalattrs["title"] = sampleid + xa
			if "description" not in globalattrs:
				globalattrs["description"] = sampleid + xa
			globalattrs["url"] = "https://storage.googleapis.com/linnarsson-lab-chromium/%s/%s.html" % (sampleid, sampleid)
			loom = loompy.connect(loom_file)
			for key, value in globalattrs.items():
				key8 = key.encode('ascii', 'replace')
				value8 = str(value).encode('ascii', 'replace')
				loom.attrs[key8] = value8
			loom.close()
		if not os.path.exists(os.path.join(d, sampleid + ".zip")):
    			zipfile = os.path.join(d, sampleid + ".zip")
    			run(["zip", "-x", "*.bam", "-r", zipfile, outsprefix], stdout=DEVNULL)

		for localpath, gspath in [(loom_file, gsprefix + sampleid + ".loom"), \
					  (outsprefix + "possorted_genome_bam.bam", gsprefix + sampleid + ".bam") , \
					  (outsprefix + "possorted_genome_bam.bam.bai", gsprefix + sampleid + ".bai") , \
					  (os.path.join(d, sampleid + ".zip"), gsprefix + sampleid + ".zip") , \
					  (outsprefix + "web_summary.html", gsprefix + sampleid + ".html")]:
			if self.props.never_overwrite_loom and gspath.endswith(".loom"):
				continue
			if not os.path.exists(localpath):
				print("WARNING: " + localpath + " is missing.")
			else:
				localsize = os.stat(localpath).st_size
				gsexists = "(new)"
				cproc = run(["gsutil", "ls", "-l", gspath], stdout=PIPE, stderr=DEVNULL)
				if cproc.returncode == 0:
					m = re.search("^ *([0-9]+) ", cproc.stdout.decode().split('\n')[0])
					gssize = int(m.group(1))
					if gssize >= localsize:
						op = "==" if gssize == localsize else ">"
						print("Skipping " + os.path.basename(localpath) + ". Already on gstorage with " + op + " size.")
						continue
					gsexists = "(overwrite, %dk)" % (gssize/1024)
				print("%s (%dk) -> %s %s" % (localpath, localsize/1024, gspath, gsexists))
				errcode = run(["gsutil", "cp", localpath, gspath], stdout=DEVNULL, stderr=DEVNULL).returncode
				if errcode != 0:
					print("  ERROR: Transfer error code: %s" % errcode)
		if self.props.upload_to_loom_server and os.path.exists(loom_file):
			print("Uploading, tiling and expanding om loom-server...")
			cmd = loom_server_upload_cmd % loom_file
			errcode = os.system(cmd)
			if errcode == 0:
				uploaded_path = loom_server_upload_dir + "/" + os.path.basename(loom_file)
				cmd = loom_server_cmd_pat % ("tile " + uploaded_path)
				errcode = os.system(cmd)
				if errcode == 0:
					cmd = loom_server_cmd_pat % ("expand -m -a -r -c " + uploaded_path)
					errcode = os.system(cmd)
			else:
				print("  ERROR: '%s',  error code: %s" % (cmd, errcode))

class MetadataDB:
	def __init__(self, sqlite3_db_file):
		self.db = sqlite3.connect(sqlite3_db_file)
		self.db.row_factory = sqlite3.Row

	def get_sample(self, sampleid):
		cur = self.db.execute("select * from sample where name = ?", [sampleid])
		row = cur.fetchone()
		sample = { "title": sampleid }
		if not row:
			return sample
		for key in row.keys():
			if key not in ("id", "name"):
				sample[key] = row[key]
		return sample


if __name__ == "__main__":
	if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
		print ("Construct .loom file (with loompy or velocyto) and upload together with 10X output to " + gs_sample_dir + "/<sampleid>")
		print ("Usage:\npython3 make_loom_and_upload.py [OPTIONS] [CELLRANGER_OUTDIR...]")
		print ("--use-velocyto             Make .loom file using velocyto instead of loompy.")
		print ("--force-loom               Make new .loom file even if it already exists locally.")
		print ("--never-overwrite-loom     Never overwrite a .loom file on gstorage, even if it is smaller than local/new file.")
		print ("--overwrite-loom           Always overwrite .loom file on gstorage.")
		print ("                           Default is to overwrite if local/new .loom file is larger.")
		print ("--upload-to-loom-server    Upload .loom file also to loom-server (requires gcloud compute access to loom-server) .")
		print ("Without CELLRANGER_OUTDIR(s), all sample folders matching '10X*' under " + local_sample_dir + " will be processed.")
		sys.exit(0)
	skipped = []
	props = OutsProcessorProps()
	argidx = 1
	while len(sys.argv) > argidx and sys.argv[argidx].startswith('-'):
		if sys.argv[argidx] == "--force-loom":
			props.force_loom = True
		elif sys.argv[argidx] == "--never-overwrite-loom":
			props.never_overwrite_loom = True
		elif sys.argv[argidx] == "--overwrite-loom":
			props.overwrite_loom = True
		elif sys.argv[argidx] == "--upload-to-loom-server":
			props.upload_to_loom_server = True
		elif sys.argv[argidx] == "--use-velocyto":
			props.use_velocyto = True
		else:
			print ("Unknown option: " + sys.argv[argidx])
			sys.exit(1)
		argidx += 1
	if props.never_overwrite_loom:
		props.force_loom = False
		props.overwrite_loom = False
	outsprocessor = OutsProcessor(props)
	metadatadb = MetadataDB(sqlite3_db_file)
	dirs = sys.argv[argidx:] if len(sys.argv) > argidx else glob.glob(local_sample_dir + "10X*")
	for d in dirs:
		sampleid = os.path.basename(d)
		print(sampleid)
		print("=" * len(sampleid))
		if not os.path.exists(os.path.join(d, "outs")):
			print("Skipping " + sampleid + " (no 'outs' folder)")
			skipped.append(sampleid)
		else:
			sample = metadatadb.get_sample(sampleid)
			outsprocessor.process_dir(d, sampleid, sample)
		print("")

if len(skipped) > 0:
	print("Dirs missing /outs subdir were skipped: " + ", ".join(skipped))
