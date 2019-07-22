[![Build Status](https://travis-ci.com/prody/rhapsody.svg?branch=master)](https://travis-ci.com/prody/rhapsody)
[![PyPI](https://img.shields.io/pypi/v/prody-rhapsody.svg)](https://pypi.org/project/prody-rhapsody/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/prody-rhapsody.svg)](http://rhapsody.csb.pitt.edu/download.php)
[![Documentation Status](https://readthedocs.org/projects/rhapsody/badge/?version=latest)](https://rhapsody.readthedocs.io/en/latest/?badge=latest)

# Rhapsody
Python program, based on ProDy, for pathogenicity prediction of human missense variants.

## Install latest published version using pip
Rhapsody is published on [PyPI](https://pypi.org/). To install Rhapsody, please use pip in the terminal:
```console
$ pip install -U prody-rhapsody
```
It might be necessary to manually install the DSSP program, for instance by typing on Linux:
```console
$ sudo apt install dssp
```

## Install from source
Rhapsody is written in pure Python so no local compilation is needed.

To install all needed dependencies, we strongly suggest to use Conda and create a new environment with:
```console
conda create -n rhapsody python=3 numpy scikit-learn requests pyparsing
conda activate rhapsody
pip install biopython prody
```

After cloning/forking the Rhapsody repository, you can permanently add the repository path to the conda environment with:
```console
conda develop path/to/local/repository
```

If not using Conda, you can manually install all dependencies and then add the repository location to the `PYTHONPATH` environmental variable. For example, on Linux simply add the following line to your `~/.bashrc`:
```console
export PYTHONPATH="path/to/local/repository/:$PYTHONPATH"
```

If you are running on Windows, please follow this [tutorial](https://stackoverflow.com/a/4855685).

