#!/bin/bash
#PBS -N rolling_intrinsic_param_sweep
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=48:00:00
#PBS -J 1-54
#PBS -o logs/job_${PBS_ARRAY_INDEX}.o
#PBS -e logs/job_${PBS_ARRAY_INDEX}.e

# Print job info
echo "Running job array index ${PBS_ARRAY_INDEX}"
cd /storage/praha1/home/tadim/gdrl/notebooks/chapter_12

# Activate environment
source /storage/praha1/home/tadim/myenv/bin/activate

# Read parameters for this array index
read theta_delta initial_delta kappa_delta sigma_delta <<< $(sed -n "${PBS_ARRAY_INDEX}p" params_list.txt)

# Run the Python script with these parameters
time python -u rolling_intrinsic_param_sweep.py \
    --theta_delta $theta_delta \
    --initial_delta $initial_delta \
    --kappa_delta $kappa_delta \
    --sigma_delta $sigma_delta | tee output_${PBS_ARRAY_INDEX}.log
