#!/bin/bash
#SBATCH --nodes 6
#SBATCH --cpus-per-task 6
#SBATCH -p longq
#SBATCH --mem 3200
#SBATCH -t 07-00:00:00  # Job time limit
#SBATCH -o ./main.out
#SBATCH -e ./main.err

python main.py
