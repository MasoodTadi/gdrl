#!/bin/bash
#PBS -N rolling_intrinsic_penalized_autoregressive_modified_AR
#PBS -l select=1:ncpus=16:mem=32gb
#PBS -l walltime=96:00:00
#PBS -o rolling_intrinsic_penalized_autoregressive_modified_AR.o
#PBS -e rolling_intrinsic_penalized_autoregressive_modified_AR.e

# Print the current working directory (optional)
pwd

# Load the necessary module
#module load python/3.8

# Activate the virtual environment
source /storage/praha1/home/tadim/myenv/bin/activate

# Set PYTHONPATH to include local packages
#export PYTHONPATH=$PYTHONPATH:/storage/praha1/home/tadim/.local/lib/python3.8/site-packages

# Change to the working directory
cd /storage/praha1/home/tadim/gdrl/notebooks/chapter_12

# Run the Python script
# time python -u autoregressive.py
# time python -u autoregressive.py > output_autoregressive.log 2>&1
time python -u rolling_intrinsic_penalized_autoregressive_modified_AR.py | tee output_rolling_intrinsic_penalized_autoregressive_modified_AR.log