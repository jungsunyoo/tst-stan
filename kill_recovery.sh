#!/bin/bash
# kill_recovery.sh
# Safely terminate all parallel recovery processes

echo "=== Terminating Parallel Recovery Processes ==="
echo ""

# Method 1: Kill Python processes running recover_rlddm_params.py
echo "1. Terminating recover_rlddm_params.py processes..."
pids=$(ps aux | grep "python.*recover_rlddm_params.py" | grep -v grep | awk '{print $2}')
if [ -n "$pids" ]; then
    echo "Found PIDs: $pids"
    kill $pids 2>/dev/null
    sleep 2
    # Force kill if still running
    kill -9 $pids 2>/dev/null
    echo "✓ Terminated recover_rlddm_params.py processes"
else
    echo "✓ No recover_rlddm_params.py processes found"
fi

# Method 2: Kill Python processes running fit_rlddm_single.py
echo ""
echo "2. Terminating fit_rlddm_single.py processes..."
pids=$(ps aux | grep "python.*fit_rlddm_single.py" | grep -v grep | awk '{print $2}')
if [ -n "$pids" ]; then
    echo "Found PIDs: $pids"
    kill $pids 2>/dev/null
    sleep 2
    # Force kill if still running
    kill -9 $pids 2>/dev/null
    echo "✓ Terminated fit_rlddm_single.py processes"
else
    echo "✓ No fit_rlddm_single.py processes found"
fi

# Method 3: Kill Stan/cmdstan processes
echo ""
echo "3. Terminating Stan/cmdstan processes..."
pids=$(ps aux | grep "rlddm_single_subject" | grep -v grep | awk '{print $2}')
if [ -n "$pids" ]; then
    echo "Found Stan PIDs: $pids"
    kill $pids 2>/dev/null
    sleep 2
    # Force kill if still running
    kill -9 $pids 2>/dev/null
    echo "✓ Terminated Stan processes"
else
    echo "✓ No Stan processes found"
fi

# Method 4: Clean up any remaining background jobs
echo ""
echo "4. Cleaning up background jobs..."
jobs -p | xargs -r kill 2>/dev/null
echo "✓ Background jobs cleaned up"

# Method 5: Clean up temporary files
echo ""
echo "5. Cleaning up temporary files..."
rm -rf recovery_out/tmp_subject_* 2>/dev/null
rm -f cmdstan_tmp/rlddm_single_subject-*.csv 2>/dev/null
rm -f cmdstan_tmp/rlddm_single_subject-*-stdout.txt 2>/dev/null
echo "✓ Temporary files cleaned up"

echo ""
echo "=== All recovery processes terminated ==="
echo ""
echo "To check if any processes are still running:"
echo "  ps aux | grep -E 'recover_rlddm|fit_rlddm|rlddm_single_subject' | grep -v grep"
echo ""
echo "To restart recovery:"
echo "  ./run_parallel_recovery.sh"
