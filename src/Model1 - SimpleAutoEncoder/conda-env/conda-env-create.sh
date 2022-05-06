#!/bin/bash
#SBATCH --time 20          # time in minutes to reserve
#SBATCH --cpus-per-task 4  # number of cpu cores
#SBATCH --mem 8G           # memory pool for all cores
#SBATCH --gres gpu:0       # number of gpu cores
#SBATCH  -o conda-env.log  # log output

# Initialise conda environment.
eval "$(conda shell.bash hook)"

# Create a new environment from the config file.
conda env create -f test-conda-env.yml

# Print all environments avaiable.
conda env list
