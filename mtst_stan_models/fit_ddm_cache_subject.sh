#!/usr/bin/env bash
#SBATCH -J ddm_cache
#SBATCH -p all
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --time=60
#SBATCH -o slurm-%j.out

set -eo pipefail

# Arguments from wrapper:
# $1 = subject number
# $2 = states number (2, 3, 4, 5)
SUBJ="$1"
STATES="$2"

# ---- activate environment (match your server style) ----
source ~/.bashrc
conda activate stan

# ---- files ----
CSV_FILE="hddm2_fixed_final_${STATES}states.csv"
SCRIPT="fit_ddm_cache_subject.py"
STAN_MODEL="ddm_cache_regression_subject.stan"

# ---- output ----
OUTDIR="fit_out/ddm_cache_regression/${STATES}state"
mkdir -p "$OUTDIR"

echo "Starting DDM cache regression"
echo "Subject: $SUBJ | States: $STATES"
echo "CSV: $CSV_FILE"
echo "Outdir: $OUTDIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo

python3 -u "$SCRIPT" \
  --csv "$CSV_FILE" \
  --states "$STATES" \
  --subj "$SUBJ" \
  --stan "$STAN_MODEL" \
  --outdir "$OUTDIR" \
  --chains 4 \
  --warmup 3000 \
  --draws 1000 \
  --adapt_delta 0.99 \
  --max_treedepth 15 \
  --seed 2027

echo
echo "End time: $(date)"
echo "DDM cache regression completed for Subject $SUBJ, States $STATES"