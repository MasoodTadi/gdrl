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
for POLICY_LR in 0.0003 0.0001 0.00005; do
     for VALUE_LR in 0.0005 0.0003 0.0001; do
         echo "Running with POLICY_LR=$POLICY_LR, VALUE_LR=$VALUE_LR"
         export POLICY_LR
         export VALUE_LR
         RUN_ID="run_$(date +%Y%m%d_%H%M%S)_plr${POLICY_LR}_vlr${VALUE_LR}"
         LOG="log_${RUN_ID}.txt"
         # time python -u ppo_gas_storage.py | tee output_ppo_gas_storage_entropy_${ENTROPY}.log
         time python -u ppo_gas_storage.py
         echo "Finished $RUN_ID"
         sleep 5
    done
done