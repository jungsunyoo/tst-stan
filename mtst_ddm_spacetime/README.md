# MTST DDM model set for the **space (memory) – time (computation)** tradeoff

This package adds a targeted DDM analysis for the hypothesis that participants face a memory–time tradeoff when deciding whether to use **conjunctive menu caching** versus **recomputation from separated components**.

## Intuition

If first-stage menus are cached as conjunctive objects, then:
- exact cache hits should **reduce nondecision time** (`t0`),
- partial/neighbor hits should also reduce `t0`, but less strongly,
- menu staleness and intervening-menu diversity should **increase `t0`**,
- and those costs/benefits should depend on the size of the menu space, `M = C(S,2)`.

So the cleanest DDM test is:

> Do cache / interference / menu-space predictors live mainly in **`t0`** (retrieval / recompute time), rather than in **`v`** (value/evidence) or **`a`** (caution)?

## Files

- `ddm_spacetime_hier.stan`
  - generic safe hierarchical Wiener regression
- `mtst_ddm_spacetime_covariates.py`
  - pooled stage-1 covariates across S=3/4/5
- `fit_ddm_spacetime_hier.py`
  - main runner for nested model variants
- `mtst_covariates.py`
  - copied helper used for loading/decoding MTST CSVs
- `mtst_ddm_spacetime_workflow.ipynb`
  - short notebook showing recommended fits and interpretation

## Predictors

### Drift (`v`) policy/value content
These predictors capture what choice *should* be favored, independent of caching cost:
- `diff_lastR_state_z`
- `diff_loglag_state_z`
- `menu_pref`
- `menu_pref_win`
- `menu_pref_lag_z`

### `t0` cache / interference / memory-space predictors
These are the key variables for the space–time tradeoff:
- `full_hit`                 : exact menu in LRU cache
- `neighbor_hit`             : a cached menu shares one option
- `menu_dist_lag1_z`         : distance from previous menu
- `log1p_menu_lag_z`         : staleness of exact menu
- `diversity_prop_z`         : distinct intervening menus / lag (interference density)
- `logM_z`                   : log menu-space size, where `M = C(S,2)`
- `full_hit_x_logM_z`        : whether per-hit benefit scales with menu-space size
- `neighbor_hit_x_logM_z`    : same for partial hits
- `diversity_x_logM_z`       : whether interference penalty scales with menu-space size
- `menulag_x_logM_z`         : whether menu-lag penalty scales with menu-space size

### Boundary (`a`) control
- `abs_diff_lastR_state_z`
- optionally `menu_dist_lag1_z`

## Recommended nested models

### 1) baseline
No explicit cache/space variables beyond trial progression in `t0`.

```bash
python fit_ddm_spacetime_hier.py \
  --data_dir . --states 3 4 5 \
  --variant baseline --K 1 \
  --warmup 500 --draws 500 \
  --outdir ddm_spacetime_baseline
```

### 2) t0_time
Add cache/interference predictors to `t0` only.

```bash
python fit_ddm_spacetime_hier.py \
  --data_dir . --states 3 4 5 \
  --variant t0_time --K 1 \
  --warmup 500 --draws 500 \
  --outdir ddm_spacetime_t0time
```

### 3) t0_space_time  **(main space–time tradeoff model)**
Add cache/interference predictors **and** menu-space interactions to `t0`.

```bash
python fit_ddm_spacetime_hier.py \
  --data_dir . --states 3 4 5 \
  --variant t0_space_time --K 1 \
  --warmup 500 --draws 500 \
  --outdir ddm_spacetime_t0spacetime
```

### 4) v_space_time
Put the same cache/space predictors in `v` instead of `t0`.

```bash
python fit_ddm_spacetime_hier.py \
  --data_dir . --states 3 4 5 \
  --variant v_space_time --K 1 \
  --warmup 500 --draws 500 \
  --outdir ddm_spacetime_vspacetime
```

### 5) t0v_space_time
Allow both `v` and `t0` to carry cache/space predictors.

```bash
python fit_ddm_spacetime_hier.py \
  --data_dir . --states 3 4 5 \
  --variant t0v_space_time --K 1 \
  --warmup 500 --draws 500 \
  --outdir ddm_spacetime_t0vspacetime
```

## Main predictions

The space–time tradeoff hypothesis is strongest if:

1. `t0_space_time` beats `baseline` and `v_space_time` in out-of-sample fit (e.g. LOO), and
2. the key `t0` coefficients have the expected signs:
   - `full_hit < 0`                (exact cache hit speeds retrieval)
   - `neighbor_hit < 0`            (partial reuse speeds retrieval)
   - `log1p_menu_lag > 0`          (stale menu retrieval is slower)
   - `diversity_prop > 0`          (interference slows retrieval)
   - `diversity_x_logM > 0`        (interference penalty grows with menu-space size)
   - `full_hit_x_logM < 0`         (per-hit benefit grows with menu-space size)

Interpretation:
- **cache hits and overlap** save time,
- **staleness and interference** cost time,
- and **menu-space size modulates** how valuable or fragile the cache is.

## Why this helps the paper

This DDM analysis does not replace the behavioral story. It sharpens it by asking:

> Is the menu-space / cache effect mainly a **retrieval-time cost** (`t0`) rather than a value/evidence effect (`v`)?

If the answer is yes, that is direct support for framing your result as a **space–time tradeoff** rather than just a generic value or caution effect.
