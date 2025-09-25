#!/bin/bash
# retry_failed_subjects.sh
# Re-run the 9 subjects that failed due to race conditions

FAILED_SUBJECTS=(10 11 12 14 15 16 17 18 19)

echo "=== Retrying Failed Subjects ==="
echo "Subjects to retry: ${FAILED_SUBJECTS[@]}"
echo ""

# Check if recovery_out directory and required files exist
if [ ! -d "recovery_out" ]; then
    echo "❌ ERROR: recovery_out directory not found!"
    echo "The original parameter recovery data has been deleted."
    echo "Please re-run the full recovery with: ./run_parallel_recovery.sh"
    exit 1
fi

if [ ! -f "recovery_out/simulated_data_2planets.csv" ]; then
    echo "❌ ERROR: Simulation data not found!"
    echo "Required file: recovery_out/simulated_data_2planets.csv"
    echo "Please re-run the full recovery with: ./run_parallel_recovery.sh"
    exit 1
fi

if [ ! -f "recovery_out/truth_params.csv" ]; then
    echo "❌ ERROR: Truth parameters not found!"
    echo "Required file: recovery_out/truth_params.csv"
    echo "Please re-run the full recovery with: ./run_parallel_recovery.sh"
    exit 1
fi

echo "✅ Required recovery files found. Proceeding with retry..."
echo ""

# Launch failed subjects with more spacing
pids=()
for subj in "${FAILED_SUBJECTS[@]}"; do
    echo "Starting subject $subj..."
    
    conda run -n stan python recover_rlddm_params.py \
        --subject $subj \
        --planets 2 \
        --trials 300 \
        --warmup 2000 \
        --draws 1000 \
        > "recovery_out/subject_${subj}/retry_log.txt" 2>&1 &
    
    pids+=($!)
    
    # Longer delay to prevent race conditions
    sleep 8
done

echo "All failed subjects restarted. Waiting for completion..."

# Wait for completion
failed_again=()
for i in "${!pids[@]}"; do
    subj=${FAILED_SUBJECTS[$i]}
    pid=${pids[$i]}
    
    echo "Waiting for subject $subj (PID: $pid)..."
    wait $pid
    
    if [ $? -eq 0 ]; then
        echo "✅ Subject $subj completed successfully on retry"
    else
        echo "❌ Subject $subj failed again"
        failed_again+=($subj)
    fi
done

echo ""
echo "=== Retry Summary ==="
successful=$((${#FAILED_SUBJECTS[@]} - ${#failed_again[@]}))
echo "Successful on retry: $successful/${#FAILED_SUBJECTS[@]}"

if [ ${#failed_again[@]} -gt 0 ]; then
    echo "Still failed: ${failed_again[@]}"
else
    echo "🎉 All subjects completed successfully!"
fi
