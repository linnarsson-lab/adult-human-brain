from setuptools import setup, find_packages

exec(open('diffusion_topology/_version.py').read())

setup(
    name = "diffusion-topology",
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
    description = "Diffusion topology algorithm",
    license = "MIT",
    url = "https://github.com/linnarsson-lab/diffusion-topology",
)

