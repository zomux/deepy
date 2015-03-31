#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys, os

if sys.version_info[:2] < (2, 6):
    raise Exception('This version of gensim needs Python 2.6 or later. ')


from distutils.core import setup
from distutils.util import convert_path
from fnmatch import fnmatchcase
from setuptools import setup, find_packages, Extension


def find_packages(where='.', exclude=()):
    out = []
    stack = [(convert_path(where), '')]
    while stack:
        where, prefix = stack.pop(0)
        for name in os.listdir(where):
            fn = os.path.join(where, name)
            if ('.' not in name and os.path.isdir(fn) and
                os.path.isfile(os.path.join(fn, '__init__.py'))
            ):
                out.append(prefix+name)
                stack.append((fn, prefix+name+'.'))
    for pat in list(exclude) + ['ez_setup', 'distribute_setup']:
        out = [item for item in out if not fnmatchcase(item, pat)]
    return out

resource_dir = os.path.join(os.path.dirname(__file__), 'deepy', 'resources')

setup(
    name='deepy',
    version='0.0.2',
    description='Highly extensible deep learning framework based on Theano',

    author='Raphael Shu',
    author_email='raphael@uaca.com',

    url='http://nlpy.org',
    download_url='http://pypi.python.org/pypi/deepy',

    keywords=' Deep learning '
        ' Neural network ',

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

    setup_requires = [
        'numpy >= 1.3',
        'theano >= 0.6.0',
    ],
    install_requires=[
        'numpy >= 1.3',
        'theano >= 0.6.0',
    ],

    extras_require={
    },

    include_package_data=True,
)