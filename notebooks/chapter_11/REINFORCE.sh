#!/bin/bash
#SBATCH --job-name=REINFORCE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64         # Reduce CPU cores if needed
#SBATCH --mem=64G                  # Reduce memory if needed
#SBATCH --time=01:00:00            # Adjust time as required
#SBATCH --output=REINFORCE.out
#SBATCH --error=REINFORCE.err

# Change to the directory where your script is located
cd ~//gdrl//notebooks//chapter_11//

# Print the current working directory (optional)
pwd

# Load the necessary module
module load python/3.8

# Run the Python script
time python REINFORCE.py
