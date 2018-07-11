#!/usr/bin/bash

#SBATCH --job-name=mpitest
#SBATCH --nodes=2
#SBATCH -o slurm-%j.out-%N
#SBATCH --ntasks=2
#SBATCH --extra-node-info=2:8:2

#SBATCH --partition=gpu
#SBATCH --time=08:00:00

source ../modules.sh

ILLUM_CLI=../src/cpp/build/Release/illum_cli
OBJ_FILE=../../data/SHAPE0.OBJ

srun --mpi=openmpi $ILLUM_CLI ratios -p $OBJ_FILE --sun_pos_file ~/sunpos.txt
