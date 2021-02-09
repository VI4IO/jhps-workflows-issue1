#!/bin/bash

filenames=( \
	"job_codings.csv" \
	"job_io_duration.csv" \
	"job_metrics.csv" \
	"job_phases.csv" \
	"job_metadata.csv" \
)

for filename in ${filenames[@]}; do
	echo "Decompressing ${filename}.tar.xz"
	tar -xJf "${filename}.tar.xz" 
done
