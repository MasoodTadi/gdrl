#!/bin/bash
#PBS -N rolling_intrinsic_AR3_vectorized
#PBS -l select=1:ncpus=16:mem=64gb
#PBS -l walltime=96:00:00
#PBS -o rolling_intrinsic_AR3_vectorized.o
#PBS -e rolling_intrinsic_AR3_vectorized.e

# Go to working directory
cd /storage/praha1/home/tadim/gdrl/notebooks/chapter_12

# Load Mamba
module add mambaforge

# Activate the correct environment
mamba activate myenv

# Debug information
echo "Python being used:"
which python

echo "Python version:"
python --version

echo "PyTorch test:"
python -c "import torch; print('Torch:', torch.__version__)"

# Run program
time python -u rolling_intrinsic_AR3_vectorized.py 2>&1 | tee output_rolling_intrinsic_AR3_vectorized.log