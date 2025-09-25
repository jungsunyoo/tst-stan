#!/bin/bash
# collect_existing_results.sh
# Collect parameter recovery results from existing files, even from failed runs

echo "=== Collecting Existing Parameter Recovery Results ==="

# Create recovery_out directory if it doesn't exist
mkdir -p recovery_out

# Look for any existing summary files that might contain results
echo "Searching for summary files..."

summary_files=(stan_out/*-summary.csv)
if [ ! -f "${summary_files[0]}" ]; then
    echo "❌ No summary files found in stan_out/"
    exit 1
fi

echo "Found ${#summary_files[@]} summary files in stan_out/"

# Try to determine which ones are from parameter recovery
echo ""
echo "Checking file timestamps to identify parameter recovery runs..."

recent_files=()
cutoff_time="2025-09-24 23:00:00"  # Files after 11 PM on Sept 24

for file in "${summary_files[@]}"; do
    file_time=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$file")
    if [[ "$file_time" > "$cutoff_time" ]]; then
        recent_files+=("$file")
        echo "Recent file: $file ($file_time)"
    fi
done

if [ ${#recent_files[@]} -eq 0 ]; then
    echo "❌ No recent summary files found after $cutoff_time"
    echo "All summary files appear to be from earlier experiments"
    exit 1
fi

echo ""
echo "Found ${#recent_files[@]} potentially relevant files"

# Try to match summary files to subject IDs based on timing
echo ""
echo "Attempting to extract parameter recovery results..."

# Create combined results file
results_file="recovery_out/collected_results.csv"
echo "participant_id,alpha_true,alpha_hat,a_true,a_hat,t0_true,t0_hat,scaler_true,scaler_hat,source_file,timestamp" > "$results_file"

successful_extractions=0

for summary_file in "${recent_files[@]}"; do
    echo "Processing: $summary_file"
    
    # Extract timestamp from filename
    filename=$(basename "$summary_file")
    if [[ $filename =~ rlddm_single_subject-([0-9_]+)-summary\.csv ]]; then
        timestamp="${BASH_REMATCH[1]}"
        
        # Try to extract parameters using Python
        python -c "
import pandas as pd
import numpy as np
import sys
from pathlib import Path

summary_file = '$summary_file'
timestamp = '$timestamp'
results_file = '$results_file'

try:
    # Read summary file
    summary_df = pd.read_csv(summary_file, index_col=0)
    
    # Extract parameters
    est_alpha = float(summary_df.loc['alpha', 'mean'])
    est_a = float(summary_df.loc['a', 'mean'])
    est_t0 = float(summary_df.loc['t0', 'mean'])
    est_scaler = float(summary_df.loc['scaler', 'mean'])
    
    # For now, we don't know the true parameters or subject ID
    # We'll mark these as unknown and let the user match them later
    result_row = f'unknown,unknown,{est_alpha},unknown,{est_a},unknown,{est_t0},unknown,{est_scaler},{summary_file},{timestamp}'
    
    # Append to results file
    with open(results_file, 'a') as f:
        f.write(result_row + '\n')
    
    print(f'✅ Extracted parameters: α={est_alpha:.3f}, a={est_a:.3f}, t0={est_t0:.3f}, scaler={est_scaler:.3f}')
    
except Exception as e:
    print(f'❌ Failed to extract parameters: {e}')
    sys.exit(1)
"
        if [ $? -eq 0 ]; then
            successful_extractions=$((successful_extractions + 1))
        fi
    else
        echo "❌ Could not parse timestamp from filename"
    fi
done

echo ""
echo "=== Collection Summary ==="
echo "Successfully extracted parameters from $successful_extractions files"
echo "Results saved to: $results_file"

if [ $successful_extractions -gt 0 ]; then
    echo ""
    echo "Note: Subject IDs and true parameters are marked as 'unknown'"
    echo "You'll need to match these with the original simulation to get recovery metrics"
    echo ""
    echo "Preview of collected results:"
    head -5 "$results_file"
else
    echo "❌ No parameter recovery results could be extracted"
fi
