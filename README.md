deepy: Highly extensible deep learning framework based on Theano
===

   
[![Build](https://travis-ci.org/uaca/deepy.svg)](https://travis-ci.org/uaca/deepy)
[![Quality](https://img.shields.io/scrutinizer/g/uaca/deepy.svg)](https://scrutinizer-ci.com/g/uaca/deepy/?branch=master)
[![Requirements Status](https://requires.io/github/uaca/deepy/requirements.svg?branch=master)](https://requires.io/github/uaca/deepy/requirements/?branch=master)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/uaca/deepy/blob/master/LICENSE)

Many codes are still not well documented.

Raphael Shu

Run with following setting
===

```
PYTHONPATH="." THEANO_FLAGS='floatX=float32,nvcc.fastmath=True,openmp=True,openmp_elemwise_minsize=1000' \
OMP_NUM_THREADS=8 python ...
```
