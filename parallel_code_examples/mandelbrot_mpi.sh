#! /bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-core=1
#SBATCH --constraint=x2680
#SBATCH -J mandelbrot


export OMP_NUM_THREADS=1
module load mpi4py
module load python/3.6
srun --mpi=pmix mandelbrot_mpi.py
