"""Python program, based on Prody, for pathogenicity prediction of human missense variants.
See:
https://github.com/prody/rhapsody
"""

from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='rhapsody',
    version='1.0.0',
    description='Python program, based on Prody, for pathogenicity prediction of human missense variants.',  
    long_description=long_description, 
    long_description_content_type='text/markdown', 
    url='https://github.com/prody/rhapsody',
    author='Luca Ponzoni', 
    author_email='ponzoniluca@gmail.com',  
    classifiers=[ 
        'Development Status :: 4 - Beta',

        'Intended Audience :: Biologists',
        'Topic :: Genetics :: SAV Identification',
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='SAV missense variant protein dynamics',  
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
    install_requires=['prody', 'numpy', 'biopython'], 

    project_urls={  
        'Bug Reports': 'https://github.com/luponzo86/rhapsody/issues',
        'Source': 'https://github.com/prody/rhapsody/',
    },
)