#!/bin/bash
#PBS -N ddpg1
#PBS -l select=1:ncpus=16:mem=32gb
#PBS -l walltime=72:00:00
#PBS -o ddpg1.o
#PBS -e ddpg1.e

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
time python -u ddpg1.py

