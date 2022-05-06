#!/bin/bash
#SBATCH --time 1440          # time in minutes to reserve
#SBATCH --cpus-per-task 2  # number of cpu cores
#SBATCH --mem 16G           # memory pool for all cores
#SBATCH --gres gpu:1       # number of gpu cores
#SBATCH  -o model.log      # write output to log file

# Run jobs using the srun command.
srun -l python model.py
