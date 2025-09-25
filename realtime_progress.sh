#!/bin/bash
# realtime_progress.sh
# Show real-time progress bars for all subjects

TOTAL_SUBJECTS=20

# Function to clean ANSI color codes
clean_ansi() {
    echo "$1" | sed 's/\x1b\[[0-9;]*m//g'
}

# Function to get the latest progress bar for a subject
get_latest_progress_bar() {
    local subj=$1
    local log_file="recovery_out/subject_${subj}/fit_log.txt"
    local result_file="recovery_out/subject_${subj}/recovery_result.csv"
    
    if [ -f "$result_file" ] && [ $(wc -l < "$result_file" 2>/dev/null || echo 0) -gt 1 ]; then
        echo "✅ COMPLETED"
    elif [ -f "$log_file" ]; then
        # Get the most recent progress bar
        progress_bar=$(tail -20 "$log_file" 2>/dev/null | grep "chain [0-9].*|.*|" | tail -1)
        
        if [ -n "$progress_bar" ]; then
            # Clean and return the progress bar
            clean_bar=$(clean_ansi "$progress_bar")
            echo "$clean_bar"
        else
            # Fallback to other status indicators
            if tail -5 "$log_file" 2>/dev/null | grep -q "Sampling completed"; then
                echo "🔄 Processing results..."
            elif tail -5 "$log_file" 2>/dev/null | grep -q "CmdStan start processing"; then
                echo "🚀 Starting MCMC..."
            elif tail -5 "$log_file" 2>/dev/null | grep -q "Data summary"; then
                echo "📊 Processing data..."
            else
                echo "⏳ Initializing..."
            fi
        fi
    else
        echo "⚪ Not started"
    fi
}

echo "=== Real-Time Progress Bars for All Subjects ==="
echo "Press Ctrl+C to exit"
echo ""

while true; do
    # Clear screen and show header
    clear
    echo "=== Real-Time Progress Bars for All Subjects ==="
    echo "$(date)"
    echo ""
    
    # Count completion status
    completed=0
    running=0
    failed=0
    
    # Display all subjects with their progress bars
    for subj in $(seq 1 $TOTAL_SUBJECTS); do
        result_file="recovery_out/subject_${subj}/recovery_result.csv"
        log_file="recovery_out/subject_${subj}/fit_log.txt"
        
        printf "Subject %2d: " "$subj"
        
        progress=$(get_latest_progress_bar $subj)
        echo "$progress"
        
        # Count status for summary
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
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Summary: $completed completed | $running running | $failed failed"
    echo "Overall Progress: $(( completed * 100 / TOTAL_SUBJECTS ))% complete"
    echo ""
    echo "Refreshing every 3 seconds... (Ctrl+C to exit)"
    
    sleep 3
done
