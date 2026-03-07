# Subject-level space–time DDM

This is the **non-hierarchical** version of the MTST space–time DDM.

## Why this version exists
Within a single subject × SSC fit, `logM` (menu-space size) is constant, so
`logM` and `x * logM` interactions are **not separately identifiable**.
Therefore this subject-level model includes only the *time/cache* variables directly:

- `full_hit`
- `neighbor_hit`
- `menu_dist_lag1`
- `log1p_menu_lag`
- `diversity_prop`

To test the **space–time** hypothesis without a hierarchical model:
1. Fit the **same variant** separately for S=3, 4, 5.
2. Compare the recovered coefficients across SSC.

## Recommended variants
- `baseline`
- `t0_cache`
- `v_cache`
- `t0v_cache`

The key hypothesis is supported if:
- `t0_cache` fits better than `baseline` and `v_cache`
- and the `t0` coefficients have expected signs:
  - `full_hit < 0`
  - `neighbor_hit < 0`
  - `log1p_menu_lag > 0`
  - `diversity_prop > 0`
  - `menu_dist_lag1 > 0`

## Files
- `ddm_spacetime_subject.stan`
- `mtst_ddm_spacetime_subject_covariates.py`
- `fit_ddm_spacetime_subject.py`
- `fit_ddm_spacetime_subject.sh`
- `submit_ddm_spacetime_subject_grid.sh`

## Example local run
```bash
python fit_ddm_spacetime_subject.py \
  --csv hddm2_fixed_final_5states.csv \
  --states 5 \
  --subj 12 \
  --variant t0_cache \
  --K 1 \
  --stan ddm_spacetime_subject.stan \
  --outdir ddm_spacetime_subject_out
```

## Example SLURM run
```bash
sbatch fit_ddm_spacetime_subject.sh 12 5 t0_cache 1
```
