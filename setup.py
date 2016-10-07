from setuptools import setup, find_packages

exec(open('dtop/_version.py').read())

setup(
    name = "dtop",
    version = __version__,
    packages = find_packages(),
    install_requires = [
        'loompy',
        'numpy',
        'scikit-learn',
        'scipy<=0.17.1',
        'annoy',
        'hdbscan'
    ],
    
    # loom command
#    scripts=['krom/krom'],
    
    # metadata
    author = "Linnarsson Lab",
    author_email = "sten.linnarsson@ki.se",
    description = "Diffusion topology algorithm",
    license = "MIT",
    url = "https://github.com/linnarsson-lab/dtop",
)

