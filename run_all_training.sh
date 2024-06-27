#!/bin/bash

input_files=(
    "../slimmed_ntuples/combined_QCD_new_HT500to700_100k.h5" 
    "../slimmed_ntuples/combined_QCD_new_HT700to1000_100k.h5" 
    "../slimmed_ntuples/combined_QCD_new_HT1000to1500_100k.h5" 
    "../slimmed_ntuples/combined_QCD_new_HT1500to2000_100k.h5" 
    "../slimmed_ntuples/combined_QCD_new_HT2000toInf_100k.h5"
)

output_dir="experiments/minimal/"


for input_file in "${input_files[@]}"; do
    echo "Running job with input: $input_file and output: $output_dir"
    
    python3 train.py -i "$input_file" -o "$output_dir" \
    -c config_files/old_config.json \
    --device gpu --max_epochs 50
    
    if [ $? -ne 0 ]; then
        echo "Job failed for input: $input_file"
        exit 1
    fi
done

echo "All jobs completed successfully."
