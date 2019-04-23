import loompy
import os
import numpy as np
from cytograph import Species, FeatureSelectionByEnrichment
from cytograph.pipeline import Punchcard, PunchcardDeck


def compute_subsets(card: Punchcard) -> None:
	with loompy.connect(os.path.join("/Users/stelin/cytograph/build_20190412", "data", card.name + ".loom"), mode="r+") as ds:
		subset_per_cell = np.zeros(ds.shape[1], dtype=object)
		taken = np.zeros(ds.shape[1], dtype=bool)
		with loompy.connect(os.path.join("/Users/stelin/cytograph/build_20190412", "data", card.name + ".agg.loom"), mode="r") as dsagg:
			for subset in card.subsets.values():
				selected = np.zeros(ds.shape[1], dtype=bool)
				if len(subset.include) > 0:
					# Include clusters that have any of the given auto-annotations
					for aa in subset.include:
						for ix in range(dsagg.shape[1]):
							if aa in dsagg.ca.AutoAnnotation[ix].split(" "):
								selected = selected | (ds.ca.Clusters == ix)
					# Exclude cells that don't match the onlyif expression
					subset.onlyif = "Clusters == 0"
					if subset.onlyif != "" and subset.onlyif is not None:
						selected = selected & eval(subset.onlyif, globals(), ds.ca)
				else:
					selected = ~taken
				# Don't include cells that were already taken
				selected = selected & ~taken
				subset_per_cell[selected] = subset.name
		ds.ca.Subset = subset_per_cell


with loompy.connect("/Users/stelin/cytograph/build_20190412/data/Root_Samples.loom") as ds:
	res = eval("(ds[ds.ra.Gene == 'Actb', :] > 2)[0]", globals(), ds.ca)
	print(res.sum())
	deck = PunchcardDeck("/Users/stelin/cytograph/build_20190412")
	card_for_subset = deck.get_card("Root_Samples")
	if card_for_subset is not None:
		compute_subsets(card_for_subset)
