#!/bin/bash

dataset_fn="../../datasets/job_codings.csv"
output_fn="../../evaluation/job_codings_clusters.csv"
progress_fn="../../evaluation/progress.csv"

set -x
rm $output_fn
rm $progress_fn
cargo run --release -- cluster --dataset=$dataset_fn --output $output_fn --progress $progress_fn
