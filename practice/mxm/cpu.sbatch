#!/bin/bash

#SBATCH --job-name=mxm
#SBATCH --time=05:00:00
#SBATCH --account=mpcs52018
#SBATCH --nodes=1
#SBATCH --partition=caslake  # accessible partitions listed by the sinfo command
#SBATCH --ntasks-per-node=1  # number of tasks per node
#SBATCH --cpus-per-task=48    # number of CPU cores per task
#SBATCH --output=mxm.out

# Load the require module(s)
# sinteractive --account=mpcs52018 --partition=caslake --time=01:00:00
module load papi
module load intel
module load emacs
export PAPI_EVENTS="PAPI_LD_INS,PAPI_L1_LDM,PAPI_FP_OPS"

gcc -o mxm -O2 -lpapi mxm.c
./mxm