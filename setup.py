from setuptools import setup, find_packages

__version__ = "0.0.0"
exec(open('cytograph/_version.py').read())

setup(
	name="cytograph",
	version=__version__,
	packages=find_packages(),
 	python_requires='==3.9.12',
	install_requires=[
		'loompy==3.0.7',
		'numpy==1.22.0',
		'scipy==1.8.0',
		'networkx==2.6.3',
		'python-louvain==0.15',  # imported as community
  		'h5py==2.10.0',
		'pyyaml==6.0',  # imported as yaml
		'statsmodels==0.13.0',
		'numpy-groupies==0.9.13',  # imported as numpy_groupies
		'python-igraph==0.9.1',  # imported as igraph
		'tqdm==4.62.3',
		'umap-learn==0.5.2',  # imported as umap 
		'harmony-pytorch==0.1.7',  # imported as harmony
		'pynndescent==0.5.5',
		'click==8.0.3',
		'leidenalg==0.8.8',
		'unidip==0.1.1',
		'opentsne==0.6.1',
  		'scikit-network==0.26.0',  # imported as sknetwork
		'numba==0.53.1'
	],
	include_package_data=True,
	entry_points='''
		[console_scripts]
		cytograph=cytograph.pipeline.commands:cli
	''',
	author="Linnarsson Lab",
	author_email="sten.linnarsson@ki.se",
	description="Pipeline for single-cell RNA-seq analysis",
	license="MIT",
	url="https://github.com/linnarsson-lab/adult-human-brain",
)
