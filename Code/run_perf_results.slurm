#!/bin/bash -l

#SBATCH --time=24:00:00
#SBATCH --partition=small
#SBATCH --ntasks=24
#SBATCH --mem=60g

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kirsh012@umn.edu

#SBATCH -e perf_results.err
#SBATCH -o perf_results.out

cd ~/masters-thesis/Code

module load python3
source activate tfgpu

python3 collect_perf_results.py

conda deactivate
