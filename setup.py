#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys, os

if sys.version_info[:2] < (2, 6):
    raise Exception('This version needs Python 2.6 or later. ')


from distutils.core import setup
from distutils.util import convert_path
from fnmatch import fnmatchcase
from setuptools import setup, find_packages, Extension

resource_dir = os.path.join(os.path.dirname(__file__), 'deepy', 'resources')

requirements = open(os.path.join(os.path.dirname(__file__), 'requirements.txt')).read().strip().split("\n")

setup(
    name='deepy',
    version='0.0.3',
    description='Highly extensible deep learning framework based on Theano',

    author='Raphael Shu',
    author_email='raphael@uaca.com',

    url='http://nlpy.org',
    download_url='http://pypi.python.org/pypi/deepy',

    keywords=' Deep learning '
        ' Neural network '
        ' Natural language processing ',

    license='MIT',
    platforms='any',

    packages=find_packages(),

    classifiers=[ # from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],

    setup_requires = requirements,
    install_requires=requirements,

    extras_require={
    },

    include_package_data=True,
)