[![Build Status](https://travis-ci.com/luponzo86/rhapsody.svg?branch=master)](https://travis-ci.com/luponzo86/rhapsody)

# rhapsody
Python program, based on Prody, for pathogenicity prediction of human missense variants.

## Install using pip
Rhapsody is published on [PyPI](https://pypi.org/). To install rhapsody, please use pip in the terminal:
```
pip install -U rhapsody
```

## Install from source
Rhapsody is written in pure Python so no local compilation is needed. To install from the source, you can download or clone rhapsody to your prefered location (e.g. `path/to/rhapsody/`), and then add that location to `PYTHONPATH` environmental variable. For example, on Linux you can add the following line to your `~/.bashrc`:
```
EXPORT PYTHONPATH="path/to/rhapsody/:$PYTHONPATH"
```

If you are running on Windows, please follow this tutorial:
[How to add to the PYTHONPATH in Windows, so it finds my modules/packages? - stackoverflow](https://stackoverflow.com/a/4855685)

