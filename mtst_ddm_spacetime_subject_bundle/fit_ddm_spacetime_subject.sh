#!/usr/bin/env bash
#SBATCH -J ddm_stsubj
#SBATCH -p all
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --time=180
#SBATCH -o slurm-%j.out

# Use -e and pipefail immediately; delay -u until after conda init
set -eo pipefail

# Arguments from wrapper:
# $1 = subject number
# $2 = states number (2, 3, 4, 5)
# $3 = variant (baseline, t0_cache, v_cache, t0v_cache, t0va_cache)
# $4 = K (optional; default 1)

SUBJ="${1:?Need subject number}"
STATES="${2:?Need states number}"
VARIANT="${3:-t0_cache}"
K="${4:-1}"

# ---- safe conda activation (no ~/.bashrc) ----
if command -v conda >/dev/null 2>&1; then
  __conda_setup="$(conda shell.bash hook 2>/dev/null)" || true
  if [ -n "${__conda_setup:-}" ]; then
    eval "$__conda_setup"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "Could not initialize conda." >&2
    exit 1
  fi
  unset __conda_setup
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "conda not found in PATH and no conda.sh found." >&2
  exit 1
fi

conda activate stan

# Safe to enable nounset now
set -u

CSV_FILE="hddm2_fixed_final_${STATES}states.csv"
SCRIPT="fit_ddm_spacetime_subject.py"
STAN_MODEL="ddm_spacetime_subject.stan"

OUTDIR="fit_out/ddm_spacetime_subject/${VARIANT}/K${K}/${STATES}state"
mkdir -p "$OUTDIR"

echo "Starting subject-level space-time DDM"
echo "Subject: $SUBJ | States: $STATES | Variant: $VARIANT | K: $K"
echo "CSV: $CSV_FILE"
echo "Stan model: $STAN_MODEL"
echo "Outdir: $OUTDIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: ${SLURM_NODELIST:-NA}"
echo "Start time: $(date)"
echo

python3 -u "$SCRIPT" \
  --csv "$CSV_FILE" \
  --states "$STATES" \
  --subj "$SUBJ" \
  --variant "$VARIANT" \
  --K "$K" \
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
echo "Subject-level space-time DDM completed"
