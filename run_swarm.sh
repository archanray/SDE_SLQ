#!/bin/bash
#SBATCH --nodes 6
#SBATCH --cpus-per-task 6
#SBATCH -p longq
#SBATCH --mem 3200
#SBATCH -t 07-12:00:00  # Job time limit
#SBATCH -o ./main2.out
#SBATCH -e ./main2.err

python main.py
