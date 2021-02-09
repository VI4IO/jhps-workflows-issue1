#!/bin/bash

label="small"
dataset_fn="../../datasets/job_codings_v3_small.csv"
output_fn="../../evaluation/job_codings_clusters_$label.csv"
progress_fn="../../evaluation/progress_$label.csv"

set -x
rm $output_fn
rm $progress_fn
cargo run --release -- cluster --dataset=$dataset_fn --output $output_fn --progress $progress_fn
