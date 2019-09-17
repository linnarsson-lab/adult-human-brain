import loompy
import numpy as np
import cytograph as cg

d = "/Users/stelin/kallisto_GRCh38/"
kt = cg.Karyotyper()
kt.fit(d + "10X174_2.loom")
