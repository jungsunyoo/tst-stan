#!/usr/bin/env bash
set -euo pipefail

# Example wrapper in the same style as your older subject-level jobs.
# Edit VARIANTS / KS / STATES as needed.

VARIANTS=(baseline t0_cache v_cache t0v_cache t0va_cache)
KS=(1)
STATES=(3 4 5)

for states in "${STATES[@]}"; do
  for subject in $(cat "state${states}_subjects.txt"); do
    for variant in "${VARIANTS[@]}"; do
      for k in "${KS[@]}"; do
        sbatch fit_ddm_spacetime_subject.sh "$subject" "$states" "$variant" "$k"
      done
    done
  done
done
