#!/usr/bin/bash
#SBATCH --job-name=mpitest
#SBATCH --mpi=openmpi
#SBATCH --nodes=4
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks-per-socket=1
#SBATCH --time=08:00:00

source ../modules.sh

ILLUM_CLI=../src/cpp/build/Release/illum_cli
OBJ_FILE=../../data/SHAPE0.OBJ

mpirun -np 4 $ILLUM_CLI horizons -p $OBJ_FILE
