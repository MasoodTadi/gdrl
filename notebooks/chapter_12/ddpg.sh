#!/bin/bash
#PBS -N ddpg
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=01:00:00
#PBS -o ddpg.o
#PBS -e ddpg.e

# Print the current working directory (optional)
pwd

# Load the necessary module
module load python/3.8

# Set PYTHONPATH to include local packages
export PYTHONPATH=$PYTHONPATH:/storage/praha1/home/tadim/.local/lib/python3.8/site-packages

# Change to the working directory
cd /storage/praha1/home/tadim/gdrl/notebooks/chapter_11

# Run the Python script
time python REINFORCE.py
