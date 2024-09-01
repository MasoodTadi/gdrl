#!/bin/bash
#PBS -N REINFORCE
#PBS -l select=1:mpiprocs=128
#PBS -l walltime=00:60:00
#PBS -q qexp
#PBS -e REINFORCE.e
#PBS -o REINFORCE.o

cd ~//gdrl//notebooks//chapter_11//
pwd

module load python/3.8

time python REINFORCE.py