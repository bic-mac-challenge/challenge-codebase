#!/bin/bash

# Loop from 0 to 98
for i in {0..98}; do
  # Format the number to be zero-padded to 3 digits (e.g., 000, 001)
  sub=$(printf "sub-%03d" $i)
  
  echo "Processing ${sub}..."
  
  # Define the output directory and ensure it exists 
  # (prevents errors if predict.py doesn't create missing directories)
  out_dir="/data/t/BIC-MAC-MICCAI2026/baseline_model/outputs/results/${sub}"
  mkdir -p "$out_dir"
  
  # Run the prediction command
  python predict.py \
    --features_dir "/data/t/BIC-MAC-MICCAI2026/bic-mac-data/train/${sub}/features/" \
    --output_ct "${out_dir}/ct.nii.gz"
    
done

echo "All predictions finished!"