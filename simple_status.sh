#!/bin/bash
# simple_status.sh
# Simple status check without fancy display

echo "=== Current Parameter Recovery Status ==="
echo "$(date)"
echo ""

completed=0
running=0
failed=0

for subj in {1..20}; do
    result_file="recovery_out/subject_${subj}/recovery_result.csv"
    
    if [ -f "$result_file" ] && [ $(wc -l < "$result_file" 2>/dev/null || echo 0) -gt 1 ]; then
        echo "Subject $subj: ✅ COMPLETED"
        completed=$((completed + 1))
    else
        # Check if process is running
        if ps aux | grep -q "subject $subj.*recover_rlddm_params" | grep -v grep; then
            echo "Subject $subj: 🔄 RUNNING"
            running=$((running + 1))
        else
            echo "Subject $subj: ❓ UNKNOWN/FAILED"
            failed=$((failed + 1))
        fi
    fi
done

echo ""
echo "Summary: $completed completed, $running running, $failed unknown/failed"
echo "Progress: $((completed * 100 / 20))% complete"
