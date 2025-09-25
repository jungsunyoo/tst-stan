#!/bin/bash
# show_all_progress.sh
# Display progress bars for all subjects sorted by subject ID

TOTAL_SUBJECTS=20

# Function to clean ANSI color codes from progress bars
clean_progress_bar() {
    echo "$1" | sed 's/\x1b\[[0-9;]*m//g'
}

# Function to get the most recent progress info for a subject
get_subject_progress() {
    local subj=$1
    local log_file="recovery_out/subject_${subj}/fit_log.txt"
    local result_file="recovery_out/subject_${subj}/recovery_result.csv"
    
    if [ -f "$result_file" ] && [ $(wc -l < "$result_file" 2>/dev/null || echo 0) -gt 1 ]; then
        echo "✅ COMPLETED"
    elif [ -f "$log_file" ]; then
        # Check if failed
        if tail -5 "$log_file" 2>/dev/null | grep -q "❌"; then
            echo "❌ FAILED"
        else
            # Get the most recent progress info
            progress_info=$(tail -15 "$log_file" 2>/dev/null)
            
            # Look for the most recent progress bar
            progress_bar=$(echo "$progress_info" | grep "chain [0-9].*|.*|" | tail -1)
            
            if [ -n "$progress_bar" ]; then
                # Clean and format progress bar
                clean_bar=$(clean_progress_bar "$progress_bar")
                # Extract just the essential progress info
                essential=$(echo "$clean_bar" | sed 's/^[[:space:]]*//' | sed 's/chain /ch/')
                echo "$essential"
            else
                # Look for iteration info as fallback
                iter_info=$(echo "$progress_info" | grep "Iteration:" | tail -1)
                if [ -n "$iter_info" ]; then
                    # Extract key parts: iteration count and percentage
                    iteration=$(echo "$iter_info" | grep -o 'Iteration: *[0-9]* */ *[0-9]*' | sed 's/Iteration: *//')
                    percent=$(echo "$iter_info" | grep -o '\[ *[0-9]*%\]' | grep -o '[0-9]*')
                    phase=$(echo "$iter_info" | grep -o '(Warmup\|Sampling)')
                    
                    if [ -n "$percent" ]; then
                        echo "iter $iteration [$percent%] $phase"
                    else
                        echo "iter $iteration $phase"
                    fi
                else
                    # Check other status indicators
                    if echo "$progress_info" | grep -q "Sampling completed"; then
                        echo "✅ Processing results..."
                    elif echo "$progress_info" | grep -q "CmdStan start processing"; then
                        echo "🚀 Starting MCMC..."
                    elif echo "$progress_info" | grep -q "Data summary\|NOTE:"; then
                        echo "📊 Processing data..."
                    else
                        echo "⏳ Initializing..."
                    fi
                fi
            fi
        fi
    else
        echo "⚪ Not started"
    fi
}

echo "=== All Subjects Progress (sorted by ID) ==="
echo "$(date)"
echo ""
echo "Subject ID | Status"
echo "-----------|----------------------------------------------------"

# Show progress for all subjects sorted by ID
for subj in $(seq 1 $TOTAL_SUBJECTS); do
    progress=$(get_subject_progress $subj)
    printf "Subject %2d | %s\n" "$subj" "$progress"
done

echo ""
echo "Legend: ✅=Completed, ❌=Failed, 🚀=Starting, 📊=Processing, ⏳=Initializing, ⚪=Not started"

# Summary stats
completed=$(for subj in $(seq 1 $TOTAL_SUBJECTS); do
    result_file="recovery_out/subject_${subj}/recovery_result.csv"
    if [ -f "$result_file" ] && [ $(wc -l < "$result_file" 2>/dev/null || echo 0) -gt 1 ]; then
        echo "1"
    fi
done | wc -l)

running=0
failed=0
for subj in $(seq 1 $TOTAL_SUBJECTS); do
    log_file="recovery_out/subject_${subj}/fit_log.txt"
    result_file="recovery_out/subject_${subj}/recovery_result.csv"
    
    if [ ! -f "$result_file" ] || [ $(wc -l < "$result_file" 2>/dev/null || echo 0) -le 1 ]; then
        if [ -f "$log_file" ]; then
            if tail -5 "$log_file" 2>/dev/null | grep -q "❌"; then
                failed=$((failed + 1))
            else
                running=$((running + 1))
            fi
        fi
    fi
done

echo ""
echo "Summary: $completed completed, $running running, $failed failed"
echo "Overall Progress: $(( completed * 100 / TOTAL_SUBJECTS ))% complete"
