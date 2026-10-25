#!/bin/bash

# Limit threading in numerical libraries to prevent CPU oversubscription on CPU-limited containers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
