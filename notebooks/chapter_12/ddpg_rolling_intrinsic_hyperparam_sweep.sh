#!/bin/bash
#PBS -N rolling_intrinsic_hyperparam_sweep
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=48:00:00
# total combinations: 1296
#PBS -J 1-1296
#PBS -o logs/hyperjob_%I.o
#PBS -e logs/hyperjob_%I.e

echo "Running hyperparameter job index ${PBS_ARRAY_INDEX}"

cd /storage/praha1/home/tadim/gdrl/notebooks/chapter_12
mkdir -p logs

source /storage/praha1/home/tadim/myenv/bin/activate

# Read the hyperparameters for this array index
read policy_lr value_lr hidden_arch batch_size max_samples n_warmup tau beta0 \
    <<< $(sed -n "${PBS_ARRAY_INDEX}p" hyperparams_list.txt)

echo "HPs:"
echo "  policy_lr   = ${policy_lr}"
echo "  value_lr    = ${value_lr}"
echo "  hidden_arch = ${hidden_arch}"
echo "  batch_size  = ${batch_size}"
echo "  max_samples = ${max_samples}"
echo "  n_warmup    = ${n_warmup}"
echo "  tau         = ${tau}"
echo "  beta0       = ${beta0}"

time python -u rolling_intrinsic_hyperparam_sweep.py \
    --scenario "${PBS_ARRAY_INDEX}" \
    --policy_lr "${policy_lr}" \
    --value_lr "${value_lr}" \
    --hidden_arch "${hidden_arch}" \
    --batch_size "${batch_size}" \
    --max_samples "${max_samples}" \
    --n_warmup_batches "${n_warmup}" \
    --tau "${tau}" \
    --beta0 "${beta0}" | tee "output_hyper_${PBS_ARRAY_INDEX}.log"
