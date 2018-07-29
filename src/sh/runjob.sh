#!/usr/bin/env bash

MODEL_SIZE=small
VESTA_MODEL=~/Projects/illum/data/vesta_${MODEL_SIZE}.obj
ILLUM=~/Projects/illum/src/cpp/build/Release/illum_cli
OUTPUT_DIR=~/vesta_${MODEL_SIZE}
GS_STEPS=3
RAD_SUN_POS_FILE=~/Projects/illum/data/vesta_sunpos_12x72.txt
THERM_SUN_POS_FILE=~/Projects/illum/data/vesta_sunpos_Jan2011_dt4min.txt
DT=240
ALBEDO=0.3
THERMAL_INERTIA=30
INITIAL_TEMPERATURE=178

mkdir -p ${OUTPUT_DIR}
${ILLUM} \
	horizons \
	-p ${VESTA_MODEL} \
	--output_dir ${OUTPUT_DIR}

mkdir -p ${OUTPUT_DIR}/rad_dir
${ILLUM} \
	radiosity \
	-p ${VESTA_MODEL} \
	--sun_pos_file ${RAD_SUN_POS_FILE} \
	--horizon_file ${OUTPUT_DIR}/horizons.bin \
	--albedo ${ALBEDO} \
	--output_dir ${OUTPUT_DIR}/rad_dir

mkdir -p ${OUTPUT_DIR}/rad_scat
${ILLUM} \
	radiosity -r \
	-p ${VESTA_MODEL} \
	--gs_steps ${GS_STEPS} \
	--print_residual \
	--sun_pos_file ${RAD_SUN_POS_FILE} \
	--horizon_file ${OUTPUT_DIR}/horizons.bin \
	--albedo ${ALBEDO} \
	--output_dir ${OUTPUT_DIR}/rad_scat

mkdir -p ${OUTPUT_DIR}/therm_dir
${ILLUM} \
	radiosity -t \
	-p ${VESTA_MODEL} \
	--sun_pos_file ${THERM_SUN_POS_FILE} \
	--horizon_file ${OUTPUT_DIR}/horizons.bin \
	--albedo ${ALBEDO} \
	--dt ${DT} \
	--ti ${THERMAL_INERTIA} \
	--T0 ${INITIAL_TEMPERATURE} \
	--output_dir ${OUTPUT_DIR}/therm_dir

mkdir -p ${OUTPUT_DIR}/therm_scat
${ILLUM} \
	radiosity -r -t \
	-p ${VESTA_MODEL} \
	--gs_steps ${GS_STEPS} \
	--print_residual \
	--sun_pos_file ${THERM_SUN_POS_FILE} \
	--horizon_file ${OUTPUT_DIR}/horizons.bin \
	--albedo ${ALBEDO} \
	--dt ${DT} \
	--ti ${THERMAL_INERTIA} \
	--T0 ${INITIAL_TEMPERATURE} \
	--output_dir ${OUTPUT_DIR}/therm_scat
