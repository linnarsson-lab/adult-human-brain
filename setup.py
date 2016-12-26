from setuptools import setup, find_packages

__version__ = "0.0.0"
exec(open('differentiation_topology/_version.py').read())

setup(
    name="differentiation-topology",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        'loompy',
        'numpy',
        'scikit-learn',
        'scipy',
        'annoy',
        'pygraphviz-1.4rc1'
    ],

    # metadata
    author="Linnarsson Lab",
    author_email="sten.linnarsson@ki.se",
    description="Differentiation topology algorithm",
    license="MIT",
    url="https://github.com/linnarsson-lab/differentiation-topology",
)
