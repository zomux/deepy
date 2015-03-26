#!/bin/bash

export THEANO_FLAGS='floatX=float32,nvcc.fastmath=True,openmp=True,openmp_elemwise_minsize=1000'
export OMP_NUM_THREADS=`nproc`
export PYTHONPATH="$PYTHONPATH:."

