#!/bin/bash

filenames=( \
	"job_codings_clusters.csv" \
	"progress.csv" \
)

for filename in ${filenames[@]}; do
	echo "Decompressing ${filename}.tar.xz"
	tar -xJf "${filename}.tar.xz" 
done
