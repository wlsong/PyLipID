# setup.py
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.rst').read_text(encoding='utf-8')

# read version info
import re
VERSIONFILE="pylipid/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# setup
setup(
    name='pylipid',
    version=verstr,
    description='PyLipID - A Python Library For Lipid Interaction Analysis',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/wlsong/PyLipID',
    author='Wanling Song',
    author_email='wanling.song@hotmail.com',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='simulation tools, network community, binding site',
    python_requires='>=3.6, <4',
    packages=find_packages(),
    install_requires=[
        "mdtraj<=1.9.4",
        "numpy",
        "pandas",
        "matplotlib>=3.3.3",
        "networkx",
        "scipy",
        "python-louvain",
        "logomaker",
        "statsmodels",
        "scikit-learn",
        "tqdm",
        "kneebow"
    ]
)
