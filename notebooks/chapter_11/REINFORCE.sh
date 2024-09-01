#!/bin/bash
#SBATCH --job-name=REINFORCE        # Job name
#SBATCH --ntasks=1                  # Number of tasks (usually 1 for single-node jobs)
#SBATCH --cpus-per-task=128         # Number of CPU cores per task
#SBATCH --mem=128G                  # Amount of memory (RAM) needed
#SBATCH --time=01:00:00             # Maximum run time (hh:mm:ss)
#SBATCH --output=REINFORCE.out      # File to write standard output
#SBATCH --error=REINFORCE.err       # File to write standard error

# Change to the directory where your script is located
cd ~//gdrl//notebooks//chapter_11//

# Print the current working directory (optional, for debugging)
pwd

# Load the necessary Python module
module load python/3.8   # Adjust the Python version as needed

# Run the Python script
time python REINFORCE.py
