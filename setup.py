#!/usr/bin/env python
# -*- coding: utf-8 -*-

import deepy
import sys, os

if sys.version_info[:2] < (2, 6):
    raise Exception('This version needs Python 2.6 or later. ')

from setuptools import setup, find_packages

requirements = ["numpy", "theano", "scipy"]

setup(
    name='deepy',
    version=deepy.__version__,
    description='Highly extensible deep learning framework based on Theano',

    author='Raphael Shu',
    author_email='raphael@uaca.com',

    url='https://github.com/uaca/deepy',
    download_url='http://pypi.python.org/pypi/deepy',

    keywords=' Deep learning '
        ' Neural network '
        ' Natural language processing ',

    license='MIT',
    platforms='any',

    packages=filter(lambda s: "secret" not in s, find_packages()),

    classifiers=[ # from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
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