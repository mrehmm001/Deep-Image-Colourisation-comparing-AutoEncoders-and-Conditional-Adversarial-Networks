#!/bin/bash
#SBATCH --time 60          # time in minutes to reserve
#SBATCH --cpus-per-task 4  # number of cpu cores
#SBATCH --mem 20G          # memory pool for all cores
#SBATCH --gres gpu:1       # number of gpu cores
#SBATCH  -o test.log       # log output

# Initialise conda environment.
eval "$(conda shell.bash hook)"
conda activate test

# Run jobs.
srun -l python --version
