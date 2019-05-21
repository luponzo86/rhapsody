[![Build Status](https://travis-ci.com/luponzo86/rhapsody.svg?branch=master)](https://travis-ci.com/luponzo86/rhapsody)

# rhapsody
Python program, based on Prody, for pathogenicity prediction of human missense variants.

## Installation
Rhapsody is published on [PyPI](https://pypi.org/). To install rhapsody, please use pip in the terminal:
```
pip install -U rhapsody
```

## To use a local ProDy package, use:
```
import os, sys
sys.path.insert(0, os.path.realpath( 'path/to/local/prody/package' ))
```
