#!/bin/bash

echo "DECOMPRESSION STARTED"
pushd ./datasets
./decompress.sh
popd

pushd ./evaluation
./decompress.sh
popd


echo "CLUSTERING STARTED (time series clustering)"
pushd ./src/coding_similarity
dataset_fn="../../datasets/job_codings.csv"
output_fn="../../evaluation/job_codings_clusters.csv"
progress_fn="../../evaluation/progress.csv"

if [ ! -f ${output_fn} ] || [ ! -f ${progress_fn} ]; then
 echo "Re-computing clusters. This may take a while."
 cargo run --release -- cluster --dataset=$dataset_fn --output $output_fn --progress $progress_fn
else
 echo "Skip clustering. Clustering results exist in $output_fn and $progress_fn."
fi
popd


echo "CLUSTERING STARTED (traditional clustering algorithms)"
pushd src/profile_similarity
./run.py
popd


echo "ANALYSIS STARTED"
pushd ./scripts
./cluster_analysis.r
./progress_visualization.r
popd
