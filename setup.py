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
    name='prody-rhapsody',
    version='0.9.0',
    description='Python program, based on ProDy, for pathogenicity prediction of human missense variants.',  
    long_description=long_description, 
    long_description_content_type='text/markdown', 
    url='https://github.com/prody/rhapsody',
    author='Luca Ponzoni', 
    author_email='ponzoniluca@gmail.com',  
    platforms=['Windows', 'macOS', 'Linux'],
    license='GPL',
    classifiers=[ 
        'Development Status :: 4 - Beta',

        'Intended Audience :: Biologists',
        'Topic :: Genetics :: SAV Classification',
        'License :: OSI Approved :: GPL License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='SAV missense variant protein dynamics',  
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  
    python_requires='>=3.5, <4',
    install_requires=['prody', 'numpy', 'biopython', 'sklearn'], 

    project_urls={  
        'Bug Reports': 'https://github.com/luponzo86/rhapsody/issues',
        'Source': 'https://github.com/prody/rhapsody/',
    },
)