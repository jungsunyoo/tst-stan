#!/bin/bash
# run_parallel_recovery.sh
# Parallel parameter recovery for RL-DDM using external parallelization
# This approach avoids carryover effects between subjects that cause "Operation not permitted" errors

# Configuration with command line arguments
TOTAL_SUBJECTS=${1:-20}
TRIALS=${2:-300}
PLANETS=${3:-2}
SEED=${4:-2027}

echo "=== Parallel Parameter Recovery ==="
echo "Total subjects: $TOTAL_SUBJECTS"
echo "Trials per subject: $TRIALS"
echo "Planets: $PLANETS"
echo ""

# Step 1: Generate simulation data (run once)
echo "[Step 1] Generating simulation data..."
python recover_rlddm_params.py \
    --subject 1 \
    --total_subjects $TOTAL_SUBJECTS \
    --trials $TRIALS \
    --planets $PLANETS \
    --seed $SEED \
    --sim_only

if [ $? -ne 0 ]; then
    echo "❌ Simulation failed"
    exit 1
fi

echo "✅ Simulation complete"
echo ""

# Step 2: Fit each subject in parallel using background processes
echo "[Step 2] Fitting subjects in parallel..."

# Start all subjects as background processes
pids=()
for subj in $(seq 1 $TOTAL_SUBJECTS); do
    echo "Starting subject $subj..."
    
    # Ensure subject directory exists
    mkdir -p "recovery_out/subject_${subj}"
    
    python recover_rlddm_params.py \
        --subject $subj \
        --total_subjects $TOTAL_SUBJECTS \
        --trials $TRIALS \
        --planets $PLANETS \
        --seed $SEED \
        --warmup 2000 \
        --draws 1000 \
        > "recovery_out/subject_${subj}/fit_log.txt" 2>&1 &
    
    pids+=($!)
    
    # Longer delay between starts to reduce file system race conditions
    sleep 5
done

echo "All subjects started. Waiting for completion..."

# Wait for all background processes to complete
failed_subjects=()
for i in "${!pids[@]}"; do
    subj=$((i + 1))
    pid=${pids[$i]}
    
    echo "Waiting for subject $subj (PID: $pid)..."
    wait $pid
    
    if [ $? -eq 0 ]; then
        echo "✅ Subject $subj completed successfully"
    else
        echo "❌ Subject $subj failed"
        failed_subjects+=($subj)
    fi
done

echo ""
echo "=== Fitting Summary ==="
successful=$((TOTAL_SUBJECTS - ${#failed_subjects[@]}))
echo "Successful: $successful/$TOTAL_SUBJECTS"

if [ ${#failed_subjects[@]} -gt 0 ]; then
    echo "Failed subjects: ${failed_subjects[*]}"
fi

# Step 3: Collect results from all subjects (robust aggregation)
echo ""
echo "[Step 3] Collecting results (even from partial runs)..."

# Ensure recovery_out directory exists
mkdir -p recovery_out

if [ -f "recovery_out/combined_results.csv" ]; then
    rm "recovery_out/combined_results.csv"
fi

# Combine individual subject results from wherever they are
echo "participant_id,alpha_true,alpha_hat,a_true,a_hat,t0_true,t0_hat,scaler_true,scaler_hat" > "recovery_out/combined_results.csv"

successful_count=0
for subj in $(seq 1 $TOTAL_SUBJECTS); do
    # Look for recovery results in multiple possible locations
    result_file=""
    
    # First try the expected location
    if [ -f "recovery_out/subject_${subj}/recovery_result.csv" ]; then
        result_file="recovery_out/subject_${subj}/recovery_result.csv"
    # Then try to find summary files and create recovery results from them
    elif [ -d "recovery_out/subject_${subj}" ]; then
        summary_files=(recovery_out/subject_${subj}/*-summary.csv)
        if [ -f "${summary_files[0]}" ]; then
            echo "Found summary file for subject $subj, creating recovery result..."
            python -c "
import pandas as pd
import numpy as np
import sys
import glob
from pathlib import Path

subj = $subj
subj_dir = Path('recovery_out/subject_${subj}')

# Find summary file
summary_files = list(subj_dir.glob('*-summary.csv'))
if not summary_files:
    sys.exit(1)

summary_file = max(summary_files, key=lambda f: f.stat().st_mtime)
summary_df = pd.read_csv(summary_file, index_col=0)

# Load truth parameters
truth_file = Path('recovery_out/truth_params.csv')
if not truth_file.exists():
    sys.exit(1)

truth = pd.read_csv(truth_file)
subj_truth = truth[truth['participant_id'] == subj]
if len(subj_truth) == 0:
    sys.exit(1)

true_params = subj_truth.iloc[0]

# Extract estimated parameters
try:
    est_alpha = float(summary_df.loc['alpha', 'mean'])
    est_a = float(summary_df.loc['a', 'mean'])
    est_t0 = float(summary_df.loc['t0', 'mean'])
    est_scaler = float(summary_df.loc['scaler', 'mean'])

    # Create recovery result
    result_df = pd.DataFrame([{
        'participant_id': subj,
        'alpha_true': true_params['alpha_true'],
        'alpha_hat': est_alpha,
        'a_true': true_params['a_true'],  
        'a_hat': est_a,
        't0_true': true_params['t0_true'],
        't0_hat': est_t0,
        'scaler_true': true_params['scaler_true'],
        'scaler_hat': est_scaler,
    }])
    result_df.to_csv(subj_dir / 'recovery_result.csv', index=False)
    print(f'Created recovery result for subject {subj}')
except:
    sys.exit(1)
"
            if [ $? -eq 0 ]; then
                result_file="recovery_out/subject_${subj}/recovery_result.csv"
            fi
        fi
    fi
    
    # Add to combined results if we found/created a result file
    if [ -n "$result_file" ] && [ -f "$result_file" ]; then
        tail -n +2 "$result_file" >> "recovery_out/combined_results.csv"
        successful_count=$((successful_count + 1))
        echo "✅ Subject $subj: Added to combined results"
    else
        echo "❌ Subject $subj: No results found"
    fi
done

echo ""
echo "Successfully aggregated results from $successful_count/$TOTAL_SUBJECTS subjects"

# Display final results
echo ""
echo "=== Parameter Recovery Results ==="
if [ -f "recovery_out/combined_results.csv" ]; then
    python -c "
import pandas as pd
import numpy as np

df = pd.read_csv('recovery_out/combined_results.csv')
print(f'Successfully recovered {len(df)} subjects')
print('')

for param in ['alpha', 'a', 't0', 'scaler']:
    true_col = f'{param}_true'
    est_col = f'{param}_hat'
    
    if true_col in df.columns and est_col in df.columns:
        mask = ~(df[true_col].isna() | df[est_col].isna())
        if mask.sum() > 0:
            bias = df.loc[mask, est_col].mean() - df.loc[mask, true_col].mean()
            rmse = np.sqrt(((df.loc[mask, est_col] - df.loc[mask, true_col]) ** 2).mean())
            if mask.sum() > 1:
                corr = df.loc[mask, true_col].corr(df.loc[mask, est_col])
            else:
                corr = np.nan
            print(f'{param:7s}: bias={bias:6.3f}, rmse={rmse:6.3f}, r={corr:6.3f}')
        else:
            print(f'{param:7s}: no valid data')
"
else
    echo "No results file found"
fi

echo ""
echo "Results saved to recovery_out/combined_results.csv"
echo "Individual subject results in recovery_out/subject_*/recovery_result.csv"
echo "Fitting logs in recovery_out/subject_*/fit_log.txt"
