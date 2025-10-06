#!/bin/bash
# auto_extend_monitor.sh
# Monitors SLURM jobs and automatically extends time when close to expiration

# Configuration
EXTENSION_MINUTES=10          # Minutes to add each extension
WARNING_MINUTES=5            # Extend when this many minutes remain
CHECK_INTERVAL=60            # Check every N seconds
MAX_EXTENSIONS=10           # Maximum extensions per job (safety limit)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track extensions per job
declare -A extension_count

log_message() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

parse_time_to_minutes() {
    local time_str="$1"
    local total_minutes=0
    
    # Handle different time formats
    if [[ $time_str =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        # HH:MM:SS format
        local hours=${BASH_REMATCH[1]}
        local minutes=${BASH_REMATCH[2]}
        local seconds=${BASH_REMATCH[3]}
        total_minutes=$((hours * 60 + minutes))
    elif [[ $time_str =~ ^([0-9]+):([0-9]+)$ ]]; then
        # MM:SS format
        local minutes=${BASH_REMATCH[1]}
        total_minutes=$minutes
    elif [[ $time_str =~ ^([0-9]+)-([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        # D-HH:MM:SS format
        local days=${BASH_REMATCH[1]}
        local hours=${BASH_REMATCH[2]}
        local minutes=${BASH_REMATCH[3]}
        total_minutes=$((days * 24 * 60 + hours * 60 + minutes))
    else
        # Try to parse as plain minutes
        total_minutes=$(echo "$time_str" | grep -o '[0-9]*' | head -1)
    fi
    
    echo $total_minutes
}

check_and_extend_jobs() {
    log_message "Checking job status..."
    
    # Get running jobs with time info
    local job_info=$(squeue -u $USER -h -o "%.10i %.10M %.10l %.8T" --states=RUNNING 2>/dev/null)
    
    if [[ -z "$job_info" ]]; then
        log_message "${YELLOW}No running jobs found${NC}"
        return 0
    fi
    
    local jobs_checked=0
    local jobs_extended=0
    
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        
        # Parse job info: JobID, TimeUsed, TimeLimit, State
        read -r job_id time_used time_limit state <<< "$line"
        
        ((jobs_checked++))
        
        # Skip if not running
        [[ "$state" != "RUNNING" ]] && continue
        
        # Parse times to minutes
        local used_minutes=$(parse_time_to_minutes "$time_used")
        local limit_minutes=$(parse_time_to_minutes "$time_limit")
        local remaining_minutes=$((limit_minutes - used_minutes))
        
        echo -e "  Job $job_id: ${used_minutes}/${limit_minutes} min (${remaining_minutes} remaining)"
        
        # Check if extension needed
        if [[ $remaining_minutes -le $WARNING_MINUTES ]]; then
            # Check extension count
            local current_extensions=${extension_count[$job_id]:-0}
            
            if [[ $current_extensions -ge $MAX_EXTENSIONS ]]; then
                log_message "${RED}⚠️  Job $job_id reached max extensions ($MAX_EXTENSIONS), skipping${NC}"
                continue
            fi
            
            # Extend the job
            local new_limit_minutes=$((limit_minutes + EXTENSION_MINUTES))
            
            log_message "${YELLOW}🔄 Extending job $job_id by $EXTENSION_MINUTES minutes...${NC}"
            
            if scontrol update JobId=$job_id TimeLimit=$new_limit_minutes 2>/dev/null; then
                extension_count[$job_id]=$((current_extensions + 1))
                ((jobs_extended++))
                log_message "${GREEN}✅ Job $job_id extended to $new_limit_minutes minutes (extension #${extension_count[$job_id]})${NC}"
            else
                log_message "${RED}❌ Failed to extend job $job_id${NC}"
            fi
        fi
        
    done <<< "$job_info"
    
    if [[ $jobs_extended -gt 0 ]]; then
        log_message "${GREEN}Extended $jobs_extended out of $jobs_checked jobs${NC}"
    else
        log_message "No extensions needed for $jobs_checked jobs"
    fi
}

show_summary() {
    log_message "Extension summary:"
    if [[ ${#extension_count[@]} -eq 0 ]]; then
        echo "  No jobs extended"
    else
        for job_id in "${!extension_count[@]}"; do
            echo "  Job $job_id: ${extension_count[$job_id]} extensions"
        done
    fi
}

cleanup() {
    log_message "${YELLOW}Monitoring stopped${NC}"
    show_summary
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main monitoring loop
log_message "${GREEN}🚀 Starting automatic job extension monitor${NC}"
log_message "Configuration:"
log_message "  Extension: $EXTENSION_MINUTES minutes"
log_message "  Warning threshold: $WARNING_MINUTES minutes remaining"
log_message "  Check interval: $CHECK_INTERVAL seconds"
log_message "  Max extensions per job: $MAX_EXTENSIONS"
log_message ""
log_message "Press Ctrl+C to stop monitoring"
log_message "=" * 50

while true; do
    check_and_extend_jobs
    
    # Check if any jobs are still running
    local running_jobs=$(squeue -u $USER -h -t RUNNING | wc -l)
    if [[ $running_jobs -eq 0 ]]; then
        log_message "${GREEN}🎉 All jobs completed!${NC}"
        show_summary
        break
    fi
    
    log_message "Next check in $CHECK_INTERVAL seconds...\n"
    sleep $CHECK_INTERVAL
done
