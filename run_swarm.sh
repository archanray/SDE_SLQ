#!/bin/bash
#SBATCH --nodes 6
#SBATCH --cpus-per-task 12
#SBATCH -p longq
#SBATCH --mem 32000
#SBATCH -t 07-00:00:00  # Job time limit
#SBATCH -o unittests.out
#SBATCH -e unittests.err

python unittests.py