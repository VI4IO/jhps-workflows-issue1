#!/bin/bash

#dataset_fn="../../datasets/job_codings_v4_confidential.csv"
#jobids=( )
#jobids=( ${jobids[@]} 19865984 )
#jobids=( ${jobids[@]} 18672376 )
#jobids=( ${jobids[@]} 17944118 )

output_dir="../../datasets"
dataset_fn="../../datasets/job_codings_v4.csv"
jobids=( )
jobids=( ${jobids[@]} 7488914 )
jobids=( ${jobids[@]} 4296426 )
jobids=( ${jobids[@]} 5024292 )

set -x
for jobid in ${jobids[@]}; do
    sim_fn="$output_dir/ks2_similarities_$jobid.csv"
    progress_fn="$output_dir/ks2_progress_$jobid.csv"
    log_fn="$output_dir/ks2_fail_$jobid.log"
    time cargo run --release -- $dataset_fn $jobid $sim_fn $progress_fn $log_fn
done
