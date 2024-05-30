#!/bin/bash

#SBATCH --job-name=GPU
#SBATCH --time=00:20:00
#SBATCH --account=mpcs52018
#SBATCH --nodes=1
#SBATCH --partition=gpu  # accessible partitions listed by the sinfo command
#SBATCH --ntasks-per-node=1  # number of tasks per node
#SBATCH --cpus-per-task=16    # number of CPU cores per task
#SBATCH --gres=gpu:1
#SBATCH --output=gpu.out
#SBATCH --constraint=a100
#SBATCH --exclusive

# Load the require module(s)
module load openblas
module load openmpi
module load cuda
module load gcc
module load papi
module load intel
module load emacs
# cat papi_hl_output/rank_0000 
# sinteractive --partition=gpu --gres=gpu:1 --account=mpcs52018 --time=00:60:00 --constraint=a100

nvcc -O2 -DBLOCK_SIZE=256 nbody_gpu1.cu timer.c -o nbody_gpu1 -lm -Xcompiler -fopenmp
nvcc -O2 -DBLOCK_SIZE=256 nbody_gpu2.cu timer.c -o nbody_gpu2 -lm
nvcc -O2 -DBLOCK_SIZE=256 nbody_gpu3.cu timer.c -o nbody_gpu3 -lm
nvcc -O2 -DBLOCK_SIZE=256 nbody_gpu4.cu timer.c -o nbody_gpu4 -lm

./nbody_gpu1 100000 20 16
./nbody_gpu2 100000
./nbody_gpu3 100000
./nbody_gpu4 100000

ncu --target-processes all --metrics sm__throughput.avg.pct_of_peak_sustained_active ./nbody_gpu1 100000 20 16
ncu --target-processes all --metrics sm__throughput.avg.pct_of_peak_sustained_active ./nbody_gpu2 1000

nvcc -O0 mxm_1.cu -o mxm_2
./mxm_1
nvcc -o cuBlas cuBlas.cu -lcublas
