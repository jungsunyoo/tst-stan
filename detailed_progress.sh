#!/bin/bash
# detailed_progress.sh
# Show detailed progress bars for all subjects

echo "=== Detailed Progress Bars for All Subjects ==="
echo "$(date)"
echo ""

for subj in $(seq 1 20); do
    result_file="recovery_out/subject_${subj}/recovery_result.csv"
    log_file="recovery_out/subject_${subj}/fit_log.txt"
    
    printf "Subject %2d: " "$subj"
    
    if [ -f "$result_file" ] && [ $(wc -l < "$result_file" 2>/dev/null || echo 0) -gt 1 ]; then
        echo "✅ COMPLETED"
    elif [ -f "$log_file" ]; then
        # Get recent progress info
        recent_lines=$(tail -15 "$log_file" 2>/dev/null)
        
        if echo "$recent_lines" | grep -q "❌"; then
            echo "❌ FAILED"
        else
            # Look for the most recent progress bar
            progress_bar=$(echo "$recent_lines" | grep "chain [0-9].*|.*|" | tail -1)
            
            if [ -n "$progress_bar" ]; then
                # Clean the progress bar and display it
                clean_bar=$(echo "$progress_bar" | sed 's/\x1b\[[0-9;]*m//g' | sed 's/^[[:space:]]*//')
                echo "$clean_bar"
            else
                # Fallback to iteration info
                iter_line=$(echo "$recent_lines" | grep "Iteration:" | tail -1)
                if [ -n "$iter_line" ]; then
                    echo "$(echo "$iter_line" | sed 's/^[[:space:]]*//')"
                else
                    if echo "$recent_lines" | grep -q "Sampling completed"; then
                        echo "🔄 Processing results..."
                    elif echo "$recent_lines" | grep -q "CmdStan start processing"; then
                        echo "🚀 Starting MCMC..."
                    else
                        echo "⏳ Initializing..."
                    fi
                fi
            fi
        fi
    else
        echo "⚪ Not started"
    fi
done

echo ""
echo "Legend: ✅=Completed, ❌=Failed, 🚀=Starting, 🔄=Running, ⏳=Initializing, ⚪=Not started"
