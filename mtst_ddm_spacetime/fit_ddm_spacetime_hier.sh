#!/usr/bin/env bash
#SBATCH -J ddm_st
#SBATCH -p all
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --time=360
#SBATCH -o slurm-%j.out

set -eo pipefail

# Usage:
#   sbatch fit_ddm_spacetime_hier.sh <VARIANT> <K> [STATE1 STATE2 ...]
#
# Example:
#   sbatch fit_ddm_spacetime_hier.sh t0_space_time 1 3 4 5

VARIANT="${1:?Need model variant}"
K="${2:?Need cache capacity K}"
shift 2

if [ "$#" -gt 0 ]; then
  STATES=("$@")
else
  STATES=(3 4 5)
fi

# ---- conda activation without sourcing ~/.bashrc ----
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "Could not find conda initialization script." >&2
  exit 1
fi

conda activate stan

SCRIPT="fit_ddm_spacetime_hier.py"
STAN_MODEL="ddm_spacetime_hier.stan"
DATA_DIR="."

STATE_TAG="$(printf "%s_" "${STATES[@]}")"
STATE_TAG="${STATE_TAG%_}"

OUTDIR="fit_out/ddm_spacetime/${VARIANT}/K${K}/states_${STATE_TAG}"
mkdir -p "$OUTDIR"

echo "Starting hierarchical space-time DDM"
echo "Variant: $VARIANT"
echo "K: $K"
echo "States: ${STATES[*]}"
echo "Stan model: $STAN_MODEL"
echo "Outdir: $OUTDIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo

python3 -u "$SCRIPT" \
  --stan "$STAN_MODEL" \
  --data_dir "$DATA_DIR" \
  --states "${STATES[@]}" \
  --variant "$VARIANT" \
  --K "$K" \
  --outdir "$OUTDIR" \
  --chains 4 \
  --warmup 3000 \
  --draws 1000 \
  --adapt_delta 0.99 \
  --max_treedepth 15 \
  --seed 2027

echo
echo "End time: $(date)"
echo "Hierarchical space-time DDM completed"