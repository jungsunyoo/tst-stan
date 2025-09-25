#!/bin/bash
# simple_progress.sh
# Simple display of all subjects' progress

echo "=== All Subjects Progress ==="
echo "$(date)"
echo ""

for subj in $(seq 1 20); do
    result_file="recovery_out/subject_${subj}/recovery_result.csv"
    log_file="recovery_out/subject_${subj}/fit_log.txt"
    
    printf "Subject %2d: " "$subj"
    
    if [ -f "$result_file" ] && [ $(wc -l < "$result_file" 2>/dev/null || echo 0) -gt 1 ]; then
        echo "✅ COMPLETED"
    elif [ -f "$log_file" ]; then
        # Get the last few lines and look for progress
        tail_output=$(tail -5 "$log_file" 2>/dev/null)
        
        if echo "$tail_output" | grep -q "❌"; then
            echo "❌ FAILED"
        elif echo "$tail_output" | grep -q "Sampling completed"; then
            echo "🔄 Processing results..."
        else
            # Look for the most recent progress line
            progress_line=$(tail -10 "$log_file" 2>/dev/null | grep -E "chain [0-9].*\|.*\||Iteration:" | tail -1)
            
            if [ -n "$progress_line" ]; then
                # Clean and shorten the progress line
                clean_line=$(echo "$progress_line" | sed 's/\x1b\[[0-9;]*m//g' | sed 's/^[[:space:]]*//')
                
                # Extract percentage if available
                percent=$(echo "$clean_line" | grep -o '\[ *[0-9]*%\]' | grep -o '[0-9]*')
                
                if [ -n "$percent" ]; then
                    echo "🔄 Running [$percent%]"
                else
                    echo "🔄 Running..."
                fi
            else
                echo "⏳ Starting..."
            fi
        fi
    else
        echo "⚪ Not started"
    fi
done

# Summary
completed=0
running=0
failed=0

for subj in $(seq 1 20); do
    result_file="recovery_out/subject_${subj}/recovery_result.csv"
    log_file="recovery_out/subject_${subj}/fit_log.txt"
    
    if [ -f "$result_file" ] && [ $(wc -l < "$result_file" 2>/dev/null || echo 0) -gt 1 ]; then
        completed=$((completed + 1))
    elif [ -f "$log_file" ]; then
        if tail -5 "$log_file" 2>/dev/null | grep -q "❌"; then
            failed=$((failed + 1))
        else
            running=$((running + 1))
        fi
    fi
done

echo ""
echo "Summary: $completed completed, $running running, $failed failed"
echo "Progress: $(( completed * 100 / 20 ))% complete"
