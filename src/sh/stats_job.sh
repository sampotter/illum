#!/usr/bin/bash

#SBATCH --job-name=stats-job
#SBATCH --output=slurm-%j.out-%N
#SBATCH --time=8:00:00
#SBATCH --mem=100gb
#SBATCH --extra-node-info=2:8:2
#SBATCH --partition=gpu

source ../../modules.sh

cd ~/spack
. share/spack/setup-env.sh
cd -
echo "- loaded spack"

ILLUM=../cpp/build/Release/illum_cli
GS_STEPS=8
SCRATCH=/nfshomes/sfp

declare -a SIZES=("xtiny" "tiny" "xsmall" "small" "medium" "large" "xlarge")

for SIZE in "${SIZES[@]}"
do
	${ILLUM} \
		radiosity -q -r \
		-p  ${SCRATCH}/vesta_${SIZE}.obj \
		--gs_steps ${GS_STEPS} \
		--print_residual
done


