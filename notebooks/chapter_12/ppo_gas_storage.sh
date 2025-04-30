#!/bin/bash
#PBS -N ppo_gas_storage
#PBS -l select=1:ncpus=64:mem=32gb 
#PBS -l walltime=96:00:00
#PBS -o ppo_gas_storage.o
#PBS -e ppo_gas_storage.e

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

# Now loop over values
for POLICY_CLIP in 0.2 0.1 0.05; do
    for POLICY_STOPPING in 0.01 0.02 0.05; do
        for VALUE_CLIP in 0.1 inf; do
            echo "Running with POLICY_CLIP=$POLICY_CLIP, POLICY_STOPPING=$POLICY_STOPPING, VALUE_CLIP=$VALUE_CLIP"
            export POLICY_CLIP
            export POLICY_STOPPING
            export VALUE_CLIP
            RUN_ID="run_$(date +%Y%m%d_%H%M%S)_plc${POLICY_CLIP}_pls${POLICY_STOPPING}_vlc${VALUE_CLIP}"
            LOG="log_${RUN_ID}.txt"
            # time python -u ppo_gas_storage.py | tee output_ppo_gas_storage_entropy_${ENTROPY}.log
            time python -u ppo_gas_storage.py
            echo "Finished $RUN_ID"
            sleep 5
        done
    done
done