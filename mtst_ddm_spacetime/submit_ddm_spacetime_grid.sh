#!/usr/bin/env bash
set -euo pipefail

# Example grid submission for the hierarchical space-time DDM.
# Edit VARIANTS / KS / STATES as needed.

VARIANTS=(
  baseline
  t0_time
  t0_space_time
  v_space_time
  t0v_space_time
  t0va_space_time
)

KS=(1 2)

# Default pooled states for the paper:
STATES=(3 4 5)

for variant in "${VARIANTS[@]}"; do
  for k in "${KS[@]}"; do
    sbatch fit_ddm_spacetime_hier.sh "$variant" "$k" "${STATES[@]}"
  done
done
