from setuptools import setup, find_packages

exec(open('differentiation_topology/_version.py').read())

setup(
    name = "differentiation-topology",
    version = __version__,
    packages = find_packages(),
    install_requires = [
        'loompy',
        'numpy',
        'scikit-learn',
        'scipy<=0.17.1',
        'annoy'
    ],

    # metadata
    author = "Linnarsson Lab",
    author_email = "sten.linnarsson@ki.se",
    description = "Differentiation topology algorithm",
    license = "MIT",
    url = "https://github.com/linnarsson-lab/differentiation-topology",
)
