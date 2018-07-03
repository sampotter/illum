#!/usr/bin/bash
#SBATCH --job-name=mpitest
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=16
#SBATCH --ntasks-per-socket=1
#SBATCH --time=08:00:00

source ../modules.sh

ILLUM_CLI=../src/cpp/build/Release/illum_cli
OBJ_FILE=../../data/SHAPE0.OBJ

srun --mpi=openmpi $ILLUM_CLI horizons -p $OBJ_FILE
