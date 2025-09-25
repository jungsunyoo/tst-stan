#!/bin/bash
# live_monitor.sh
# Compact monitoring for terminal with limited lines

TOTAL_SUBJECTS=20

# Function to clean ANSI color codes
clean_ansi() {
    echo "$1" | sed 's/\x1b\[[0-9;]*m//g'
}

# Function to get subject status and progress
get_subject_status() {
    local subj=$1
    local log_file="recovery_out/subject_${subj}/fit_log.txt"
    local retry_log_file="recovery_out/subject_${subj}/retry_log.txt"
    local result_file="recovery_out/subject_${subj}/recovery_result.csv"
    
    # First check if subject is completed (has valid result file)
    if [ -f "$result_file" ] && [ $(wc -l < "$result_file" 2>/dev/null || echo 0) -gt 1 ]; then
        echo "COMPLETED"
        return
    fi
    
    # Check both original and retry log files for progress
    local active_log=""
    if [ -f "$retry_log_file" ] && [ $(wc -l < "$retry_log_file" 2>/dev/null || echo 0) -gt 5 ]; then
        active_log="$retry_log_file"
    elif [ -f "$log_file" ]; then
        active_log="$log_file"
    fi
    
    if [ -n "$active_log" ]; then
        # Get ALL progress bars and find the most advanced one
        progress_bars=$(tail -50 "$active_log" 2>/dev/null | grep "chain [0-9].*|.*|")
        
        if [ -n "$progress_bars" ]; then
            # Find the highest percentage among all chains
            max_percent=0
            best_line=""
            
            while IFS= read -r line; do
                if [ -n "$line" ]; then
                    clean_bar=$(clean_ansi "$line")
                    percent=$(echo "$clean_bar" | sed -n 's/.*\[ *\([0-9]*\)%\].*/\1/p')
                    if [ -n "$percent" ] && [ "$percent" -gt "$max_percent" ]; then
                        max_percent=$percent
                        best_line="$clean_bar"
                    fi
                fi
            done <<< "$progress_bars"
            
            if [ -n "$best_line" ]; then
                # Extract progress info from the most advanced chain
                echo "$best_line" | sed 's/.*Iteration: \([0-9]*\) \/ \([0-9]*\) \[ *\([0-9]*\)%\].*/\1\/\2 (\3%)/'
                return
            fi
        fi
        
        # Check for status messages in the active log
        if tail -10 "$active_log" 2>/dev/null | grep -q "Sampling completed\|✅.*completed successfully"; then
            echo "PROCESSING"
        elif tail -10 "$active_log" 2>/dev/null | grep -q "CmdStan start processing\|Starting.*fit"; then
            echo "STARTING"
        elif tail -10 "$active_log" 2>/dev/null | grep -q "❌.*failed"; then
            echo "FAILED"
        else
            echo "INIT"
        fi
    else
        echo "WAITING"
    fi
}

# Initialize display with all subjects
clear
echo "=== Parameter Recovery Progress ==="
echo ""
echo "📊 Overall: Initializing..."
echo ""
echo "Subjects:"
for subj in $(seq 1 $TOTAL_SUBJECTS); do
    printf "S%2d: ⚪ WAITING\n" "$subj"
done
echo ""
echo "Press Ctrl+C to exit"

while true; do
    # Move cursor to update positions without clearing
    tput cup 2 0  # Move to overall status line
    
    completed=0
    running=0
    waiting=0
    
    # Count status
    for subj in $(seq 1 $TOTAL_SUBJECTS); do
        status=$(get_subject_status $subj)
        
        if [[ "$status" == "COMPLETED" ]]; then
            completed=$((completed + 1))
        elif [[ "$status" =~ ^[0-9]+/[0-9]+ ]] || [[ "$status" =~ (PROCESSING|STARTING|INIT) ]]; then
            running=$((running + 1))
        else
            waiting=$((waiting + 1))
        fi
    done
    
    # Update overall summary (fixed position)
    printf "📊 Overall: %d✅ | %d🔄 | %d⚪ | %d%% complete                    \n" \
           "$completed" "$running" "$waiting" "$((completed * 100 / TOTAL_SUBJECTS))"
    
    # Move to subjects section and update each line
    tput cup 5 0  # Move to first subject line
    
    for subj in $(seq 1 $TOTAL_SUBJECTS); do
        status=$(get_subject_status $subj)
        
        if [[ "$status" == "COMPLETED" ]]; then
            printf "S%2d: ✅ COMPLETED                                     \n" "$subj"
        elif [[ "$status" =~ ^[0-9]+/[0-9]+ ]]; then
            printf "S%2d: 🔄 %s                                     \n" "$subj" "$status"
        elif [[ "$status" == "PROCESSING" ]]; then
            printf "S%2d: 🔄 PROCESSING RESULTS                           \n" "$subj"
        elif [[ "$status" == "STARTING" ]]; then
            printf "S%2d: 🚀 STARTING MCMC                               \n" "$subj"
        elif [[ "$status" == "FAILED" ]]; then
            printf "S%2d: ❌ FAILED                                       \n" "$subj"
        elif [[ "$status" == "INIT" ]]; then
            printf "S%2d: ⏳ INITIALIZING                                \n" "$subj"
        else
            printf "S%2d: ⚪ WAITING                                      \n" "$subj"
        fi
    done
    
    sleep 2
done