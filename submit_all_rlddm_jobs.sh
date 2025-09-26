#!/bin/bash
# submit_all_rlddm_jobs.sh
# Submit SLURM jobs for all subjects across all conditions

echo "=== Submitting RLDDM SLURM Jobs ==="

# Check if slurm script exists
if [ ! -f "rlddm-fit-slurm.sub" ]; then
    echo "❌ rlddm-fit-slurm.sub not found"
    exit 1
fi

# Submit jobs for each condition
total_jobs=0

for states in 2 3 4 5; do
    subject_file="state${states}_subjects.txt"
    
    if [ ! -f "$subject_file" ]; then
        echo "⚠️  $subject_file not found, skipping ${states}-state condition"
        continue
    fi
    
    echo "Processing ${states}-state condition..."
    
    condition_jobs=0
    while read -r subject; do
        # Skip empty lines
        [ -z "$subject" ] && continue
        
        echo "  Submitting Subject $subject (${states} states)"
        sbatch rlddm-fit-slurm.sub "$subject" "$states"
        
        ((condition_jobs++))
        ((total_jobs++))
        
        # Small delay to avoid overwhelming scheduler
        sleep 0.1
        
    done < "$subject_file"
    
    echo "  → Submitted $condition_jobs jobs for ${states}-state condition"
done

echo ""
echo "✅ Total jobs submitted: $total_jobs"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "After completion, gather results with:"
echo "  python gather_slurm_results.py"
