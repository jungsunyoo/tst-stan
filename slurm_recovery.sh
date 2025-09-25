#!/bin/bash
#SBATCH --job-name=rlddm_recovery
#SBATCH --output=logs/recovery_%A_%a.out
#SBATCH --error=logs/recovery_%A_%a.err
#SBATCH --array=1-20%5          # Run subjects 1-20, max 5 concurrent jobs
#SBATCH --time=02:00:00         # 2 hours per subject
#SBATCH --mem=4G                # 4GB RAM per subject
#SBATCH --cpus-per-task=1       # Single CPU (we use chains=1)
#SBATCH --partition=standard

# SLURM Parameter Recovery for RL-DDM
# Usage: sbatch slurm_recovery.sh
# 
# This script processes each subject as a separate SLURM array job,
# completely avoiding carryover effects and resource conflicts.

# Configuration
TOTAL_SUBJECTS=20
TRIALS=300
PLANETS=2
SEED=2027

# Create logs directory if it doesn't exist
mkdir -p logs

# Load your conda environment
source ~/.bashrc
conda activate stan  # or whatever your environment is called

# Set up subject index from SLURM array task ID
SUBJECT_ID=$SLURM_ARRAY_TASK_ID

echo "=== SLURM Parameter Recovery ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Subject: $SUBJECT_ID"
echo "Node: $HOSTNAME"
echo "Started: $(date)"
echo ""

# Ensure simulation data exists (only run on first job)
if [ $SUBJECT_ID -eq 1 ]; then
    echo "Generating simulation data (subject 1 only)..."
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
fi

# Wait for simulation to complete (for non-first jobs)
if [ $SUBJECT_ID -gt 1 ]; then
    echo "Waiting for simulation data..."
    while [ ! -f "recovery_out/simulated_data_${PLANETS}planets.csv" ]; do
        sleep 10
    done
    echo "✅ Simulation data found"
fi

echo "Fitting subject $SUBJECT_ID..."

# Fit this specific subject
python recover_rlddm_params.py \
    --subject $SUBJECT_ID \
    --total_subjects $TOTAL_SUBJECTS \
    --trials $TRIALS \
    --planets $PLANETS \
    --seed $SEED \
    --chains 1 \
    --warmup 2000 \
    --draws 1000 \
    --adapt_delta 0.999 \
    --max_treedepth 15

exit_code=$?

echo ""
echo "Completed: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ Subject $SUBJECT_ID fitted successfully"
else
    echo "❌ Subject $SUBJECT_ID failed (exit code: $exit_code)"
fi

exit $exit_code
