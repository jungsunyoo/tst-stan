#!/bin/bash
# monitor_recovery.sh
# Enhanced resource and progress monitor for parameter recovery

echo "=== Parameter Recovery Resource Monitor ==="
echo "Monitoring system resources and progress every 30 seconds..."
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "=== Parameter Recovery Monitor ==="
    echo "Time: $(date)"
    echo ""
    
    # Count completed subjects
    if [ -d "recovery_out" ]; then
        completed=$(find recovery_out -name "recovery_result.csv" 2>/dev/null | wc -l | xargs)
        echo "Completed subjects: $completed/20"
    else
        echo "❌ recovery_out directory not found!"
    fi
    
    # Resource monitoring
    echo ""
    echo "=== System Resources ==="
    
    # Memory usage
    echo "Memory: $(top -l 1 -s 0 | grep PhysMem | awk '{print $2, $4, $6}')"
    
    # CPU usage
    cpu_usage=$(top -l 1 -s 0 | grep "CPU usage" | awk '{print $3, $5}')
    echo "CPU: $cpu_usage"
    
    # Check for recovery processes
    recovery_procs=$(ps aux | grep "recover_rlddm_params" | grep -v grep | wc -l | xargs)
    echo "Active recovery processes: $recovery_procs"
    
    # Check disk space
    disk_usage=$(df -h . | tail -1 | awk '{print $4, $5}')
    echo "Disk available: $disk_usage"
    
    # Show running processes
    echo ""
    echo "=== Running Processes ==="
    ps aux | grep recover_rlddm_params | grep -v grep | head -5
    
    # Check for recent terminations in system log (last 2 minutes)
    recent_kills=$(log show --predicate 'process == "kernel"' --last 2m 2>/dev/null | grep -i "killed\|terminated" | tail -3)
    if [ -n "$recent_kills" ]; then
        echo ""
        echo "⚠️  Recent process terminations:"
        echo "$recent_kills" | sed 's/^/  /'
    fi
    
    echo ""
    echo "Press Ctrl+C to exit"
    sleep 30
done
