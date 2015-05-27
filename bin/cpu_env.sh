#!/bin/bash

# It turns out 'openmp=True' will cause some problems in Theano 0.7.0, so it was removed

export THEANO_FLAGS='mode=FAST_RUN,floatX=float32,openmp_elemwise_minsize=1000,allow_gc=False'
export OMP_NUM_THREADS=`nproc`
export PYTHONPATH="$PYTHONPATH:."

