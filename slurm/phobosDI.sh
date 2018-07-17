#!/usr/bin/bash

#SBATCH --job-name=vestaDI
#SBATCH --output=slurm-%j.out-%N
#SBATCH --time=08:00:00

#SBATCH --mem=100gb

#SBATCH --extra-node-info=2:8:2
#SBATCH --partition=gpu

source ../modules.sh

echo "$SLURM_JOB_NUM_NODES nodes, $SLURM_NTASKS tasks, $SLURM_CPUS_PER_TASK cores"

ILLUM_CLI=../src/cpp/build/Release/illum_cli
OBJ_FILE=../../data/SHAPE2.OBJ
SUN_POS_FILE=../../data/sunpos.txt

$ILLUM_CLI ratios -p $OBJ_FILE --sun_pos_file=$SUN_POS_FILE
