#!/usr/bin/env python
import os, glob, re, sys
import sqlite3
from subprocess import run, PIPE, DEVNULL
import loompy

local_sample_dir = "/data/proj/chromium/"
gs_sample_dir = "gs://linnarsson-lab-chromium/"
sqlite3_db_file = "/mnt/sanger-data/10X/DB/sqlite3_chromium.db"
velocyto_ivl_dir = "/data/proj/chromium/intervals"

class OutsProcessor:
	def __init__(self, force_loom = False, never_overwrite_loom = False, use_velocyto = False):
		self.force_loom = force_loom
		self.never_overwrite_loom = never_overwrite_loom
		self.use_velocyto = use_velocyto

	def process_dir(self, d, sampleid, globalattrs_dict = None):
		outsprefix = os.path.join(d, "outs") + os.sep
		gsprefix = gs_sample_dir + sampleid + "/"
		loom_file = os.path.join(d, sampleid + ".loom")
		if self.force_loom or not os.path.exists(loom_file):
			loom_made = False
			if use_velocyto:
				print("Making loom file with velocyto")
				cmd = ["velocyto", "run10x", d, ivlfile]
				errcode = subprocess.run(cmd).returncode # Should return outputfile, or allow passing  it
				if errcode == 0:
					loom_made = True
			if not loom_made:
				print("Making loom file with loompy")
				loompy.create_from_cellranger(d)
			if globalattrs_dict == None:
				globalattrs_dict = {}
			if "title" not in globalattrs_dict:
				globalattrs_dict["title"] = sampleid
			loom = loompy.connect(loom_file)
			for key, value in globalattrs_dict.items():
				key8 = key.encode('ascii', 'replace')
				value8 = str(value).encode('ascii', 'replace')
				loom.attrs[key8] = value8
			loom.close()
		if not os.path.exists(os.path.join(d, sampleid + ".zip")):
    			zipfile = os.path.join(d, sampleid + ".zip")
    			run(["zip", "-x", "*.bam", "-r", zipfile, outsprefix], stdout=DEVNULL)

		for localpath, gspath in [(os.path.join(d, sampleid + ".loom"), gsprefix + sampleid + ".loom"), \
					  (outsprefix + "possorted_genome_bam.bam", gsprefix + sampleid + ".bam") , \
					  (outsprefix + "possorted_genome_bam.bam.bai", gsprefix + sampleid + ".bai") , \
					  (os.path.join(d, sampleid + ".zip"), gsprefix + sampleid + ".zip") , \
					  (outsprefix + "web_summary.html", gsprefix + sampleid + ".html")]:
			if self.never_overwrite_loom and gspath.endswith(".loom"):
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
						print ("Skipping " + os.path.basename(localpath) + ". Already on gstorage with " + op + " size.")
						continue
					gsexists = "(overwrite, %dk)" % (gssize/1024)
				print("%s (%dk) -> %s %s" % (localpath, localsize/1024, gspath, gsexists))
				errcode = run(["gsutil", "cp", localpath, gspath], stdout=DEVNULL, stderr=DEVNULL).returncode
				if errcode != 0:
					print ("  ERROR: Transfer error code: %s" % errcode)

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
		print ("Usage:\n./make_loom.py [OPTIONS] [CELLRANGER_OUTDIR...]")
		print ("--force-loom               Make new .loom file even if it already exists locally.")
		print ("--never-overwrite-loom     Never overwrite a .loom file on gstorage, even if it is smaller than local/new file.")
		print ("                           Default is to overwrite if local/new .loom file is larger.")
		print ("--use-velocyto             Make .loom file using velocyto instead of loompy.")
		print ("Without CELLRANGER_OUTDIR(s), all sample folders matching '10X*' under " + local_sample_dir + " will be processed.")
		sys.exit(0)
	skipped = []
	force_loom = False
	never_overwrite_loom = False
	use_velocyto = False
	argidx = 1
	while len(sys.argv) > argidx and sys.argv[argidx].startswith('-'):
		if sys.argv[argidx] == "--force-loom":
			force_loom = True
		elif sys.argv[argidx] == "--never-overwrite-loom":
			never_overwrite_loom = True
		elif sys.argv[argidx] == "--use-velocyto":
			use_velocyto = True
		else:
			print ("Unknown option: " + sys.argv[argidx])
			sys.exit(1)
		argidx += 1
	if never_overwrite_loom:
		force_loom = False
	outsprocessor = OutsProcessor(force_loom, never_overwrite_loom)
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

print("Dirs missing /outs were skipped: " + ", ".join(skipped))
