
# MTST Stan DDM models: menu-graph caching vs recomputation

This folder contains Stan + Python (cmdstanpy) templates to *strengthen* the behavioral menu-graph results
by decomposing effects into DDM components:

- **drift** `v` (evidence/value)
- **nondecision time** `t0` (encoding/retrieval; where "caching" most naturally lives)
- **boundary** `a` (caution/conflict)

It also includes an optional **2-state IOHMM-DDM** that is the DDM analogue of your
choice+Gaussian-RT IOHMM.

## Files

### Core utilities
- `mtst_covariates.py` : loads MTST CSVs and builds per-subject predictors:
  - separated/component predictors: `diff_lastR_state`, `diff_loglag_state`
  - menu/conjunctive predictors: `menu_pref`, `menu_pref_win`, `menu_pref_lag`
  - retrieval predictors for `t0`: `menu_dist_lag1`, `log1p_menu_lag`, `trial_scaled`
  - transition predictors for IOHMM: `menu_dist_lag1`, `log1p_menu_lag`, `log1p_lag_pred_option`, `prev_reward`

### DDM regression (recommended first)
- `ddm_cache_regression_subject.stan`
- `fit_ddm_cache_subject.py`

### Hierarchical DDM regression (strongest, but slowest)
- `ddm_cache_regression_hier.stan`
- `fit_ddm_cache_hier.py`

### Optional: 2-state IOHMM-DDM (arbitration + DDM)
- `iohmm_ddm_cache_subject.stan`
- `fit_iohmm_ddm_cache_subject.py`

## Install

```bash
pip install cmdstanpy arviz numpy pandas
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

## Run: per-subject DDM regression

```bash
python fit_ddm_cache_subject.py \
  --csv hddm2_fixed_final_5states.csv \
  --states 5 --subj 12 \
  --stan ddm_cache_regression_subject.stan \
  --outdir stan_ddm_cache_out
```

## Run: hierarchical DDM regression

Start small:

```bash
python fit_ddm_cache_hier.py \
  --csv hddm2_fixed_final_5states.csv \
  --states 5 \
  --max_subjects 30 \
  --warmup 500 --draws 500 \
  --outdir stan_ddm_hier_out
```

Then scale up if it samples well.

## Run: IOHMM-DDM

```bash
python fit_iohmm_ddm_cache_subject.py \
  --csv hddm2_fixed_final_5states.csv \
  --states 5 --subj 12 \
  --stan iohmm_ddm_cache_subject.stan \
  --outdir stan_iohmm_ddm_out
```

## What to look at

### For the "cached vs recomputation" claim:
1. **t0 slopes** (`b_t0`) for:
   - `menu_dist_lag1` (positive = farther menu jump -> longer nondecision time)
   - `log1p_menu_lag` (positive = stale menu -> longer nondecision time)
2. **drift slopes** (`b_v`) for:
   - separated signals (`diff_lastR_state`, `diff_loglag_state`)
   - menu signals (`menu_pref`, `menu_pref_win`, `menu_pref_lag`)
   - especially `menu_pref_lag` should be negative if cache influence decays with staleness.
3. **boundary slopes** (`b_a`) to check speed–accuracy tradeoff explanations.

### For the "arbitration rule" claim (IOHMM-DDM):
- `delta_from_menu`: covariates that increase **MENU->OPTION** switching
- `delta_from_opt` : covariates that increase **OPTION staying** (so negative increases OPTION->MENU switching)

Use the same interpretation you already used in your choice-only IOHMM.

---
