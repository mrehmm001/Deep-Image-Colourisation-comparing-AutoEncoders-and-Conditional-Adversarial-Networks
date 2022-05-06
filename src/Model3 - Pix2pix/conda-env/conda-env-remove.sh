#!/bin/bash
#SBATCH --time 5           # time in minutes to reserve
#SBATCH --cpus-per-task 1  # number of cpu cores
#SBATCH --mem 1G           # memory pool for all cores
#SBATCH --gres gpu:0       # number of gpu cores
#SBATCH  -o conda-env.log  # log output

# Initialise conda environment.
eval "$(conda shell.bash hook)"

conda env remove --name test
conda env list
