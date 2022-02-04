from setuptools import setup, find_packages

__version__ = "0.0.0"
exec(open('cytograph/_version.py').read())

setup(
	name="cytograph",
	version=__version__,
	packages=find_packages(),
	install_requires=[
		'loompy',
		'numpy',
		'scikit-learn',
		'scipy',
		'networkx',
		'python-louvain',  # is imported as "community"
		'hdbscan',
		'pyyaml',
		'statsmodels',  # for multiple testing
		'numpy-groupies',
		# 'python-igraph',  # imported as "igraph"; a challenge to install so better do it manually
		'tqdm',
		'umap-learn',  # imported as "umap"
		'torch',
		# 'torchvision',
		'harmony-pytorch',
		'pynndescent',
		'click',
		'leidenalg',
		'unidip',
		'opentsne<=0.5.1'
	],
	include_package_data=True,
	entry_points='''
		[console_scripts]
		cytograph=cytograph.pipeline.commands:cli
	''',
	# entry_points={
	# 	"console_scripts": "cytograph=cytograph.pipeline.commands:cli"
	# },
	# metadata
	author="Linnarsson Lab",
	author_email="sten.linnarsson@ki.se",
	description="Pipeline for single-cell RNA-seq analysis",
	license="MIT",
	url="https://github.com/linnarsson-lab/cytograph-dev",
)
