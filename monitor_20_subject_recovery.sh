#!/bin/bash
# monitor_20_subject_recovery.sh
# Real-time monitoring for the 20-subject parameter recovery

echo "=== 20-Subject Parameter Recovery Monitor ==="
echo "Started: $(date)"
echo ""

while true; do
    # Count running processes
    running_count=$(ps aux | grep "recover_rlddm_params" | grep -v grep | wc -l)
    
    # Count completed subjects (those with recovery_result.csv)
    completed_count=0
    if [ -d "recovery_out" ]; then
        for i in {1..20}; do
            if [ -f "recovery_out/subject_${i}/recovery_result.csv" ]; then
                completed_count=$((completed_count + 1))
            fi
        done
    fi
    
    # Calculate progress
    progress_pct=$((completed_count * 100 / 20))
    
    # Display status
    clear
    echo "=== 20-Subject Parameter Recovery Monitor ==="
    echo "Time: $(date)"
    echo ""
    echo "📊 Progress: $completed_count/20 subjects completed (${progress_pct}%)"
    echo "🔄 Currently running: $running_count processes"
    echo ""
    
    # Show progress bar
    printf "Progress: ["
    for ((i=1; i<=20; i++)); do
        if [ $i -le $completed_count ]; then
            printf "✅"
        elif [ $((i - completed_count)) -le $running_count ]; then
            printf "🔄"
        else
            printf "⏳"
        fi
    done
    printf "]\n\n"
    
    # Show individual subject status
    echo "Subject Status:"
    for i in {1..20}; do
        if [ -f "recovery_out/subject_${i}/recovery_result.csv" ]; then
            printf "Subject %2d: ✅ Completed\n" $i
        elif ps aux | grep "recover_rlddm_params.*subject $i " | grep -v grep > /dev/null; then
            printf "Subject %2d: 🔄 Running\n" $i
        else
            printf "Subject %2d: ⏳ Waiting\n" $i
        fi
    done
    
    # Check if all completed
    if [ $completed_count -eq 20 ]; then
        echo ""
        echo "🎉 All 20 subjects completed!"
        echo "Run time: $(date)"
        break
    fi
    
    sleep 10
done
