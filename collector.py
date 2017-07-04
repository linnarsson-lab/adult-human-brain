#!/usr/bin/env python
import os
import sys
from subprocess import run
import loompy


if __name__ == "__main__":
	skipped = []
	for d in sys.argv[1:]:
		sampleid = os.path.split(os.path.abspath(d))[-1]
		print(sampleid)
		print("=" * len(sampleid))
		print("")
		if not os.path.exists(os.path.join(d, "outs")):
			print("Skipping " + sampleid + " (no 'outs' folder)")
			skipped.append(sampleid)
		else:
			print("Making loom file")
			loompy.create_from_cellranger(d)

			print("Making zip file")
			zipfile = os.path.join(d, sampleid + ".zip")
			run(["zip", "-x", "*.bam", "-r", zipfile, os.path.join(d, "outs")])

			localpath = sampleid + ".loom"
			gspath = "gs://linnarsson-lab-chromium/" + sampleid + "/" + sampleid + ".loom"
			print(localpath + " -> " + gspath)
			run(["gsutil", "cp", os.path.join(d, localpath), gspath])

			localpath = sampleid + ".zip"
			gspath = "gs://linnarsson-lab-chromium/" + sampleid + "/" + sampleid + ".zip"
			print(localpath + " -> " + gspath)
			run(["gsutil", "cp", os.path.join(d, localpath), gspath])

			localpath = "outs/possorted_genome_bam.bam"
			gspath = "gs://linnarsson-lab-chromium/" + sampleid + "/" + sampleid + ".bam"
			print(localpath + " -> " + gspath)
			run(["gsutil", "cp", os.path.join(d, localpath), gspath])

			localpath = "outs/possorted_genome_bam.bam.bai"
			gspath = "gs://linnarsson-lab-chromium/" + sampleid + "/" + sampleid + ".bai"
			print(localpath + " -> " + gspath)
			run(["gsutil", "cp", os.path.join(d, localpath), gspath])

			localpath = "outs/web_summary.html"
			gspath = "gs://linnarsson-lab-chromium/" + sampleid + "/" + sampleid + ".html"
			print(localpath + " -> " + gspath)
			run(["gsutil", "cp", os.path.join(d, localpath), gspath])

			print("")
			print("")
	print("Skipped: " + str(skipped))
