#!/usr/bin/bash

#SBATCH --job-name=mpitest
#SBATCH --output=slurm-%j.out-%N
#SBATCH --time=08:00:00

#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100gb
#SBATCH --cpus-per-task=32

#SBATCH --extra-node-info=2:8:2
#SBATCH --partition=gpu

source ../modules.sh

echo "$SLURM_JOB_NUM_NODES nodes, $SLURM_NTASKS tasks, $SLURM_CPUS_PER_TASK cores"

ILLUM_CLI=../src/cpp/build/Release/illum_cli
OBJ_FILE=../../data/SHAPE0.OBJ

time srun --mpi=openmpi $ILLUM_CLI ratios -p $OBJ_FILE
# time mpirun -np 2 $ILLUM_CLI ratios -p $OBJ_FILE
