#!/usr/bin/env python3
"""
fit_ddm_spacetime_hier.py

Hierarchical DDM model-comparison runner for the MTST *space (memory) - time* tradeoff.

Main question
-------------
Do menu-space / cache / interference variables primarily affect:
  (a) nondecision time t0  -> retrieval / recompute cost (space-time tradeoff)
  (b) drift v             -> value/evidence itself
  (c) both?

The script pools subjects across S=3/4/5 (or any subset), constructs stage-1 DDM regressors,
and fits one of several nested design variants with the same generic hierarchical Wiener model.

Recommended variants
--------------------
- baseline          : value/content only in drift; trial effect only in t0
- t0_time           : cache/interference predictors in t0 only
- t0_space_time     : cache/interference + menu-space interactions in t0 only
- v_space_time      : same predictors in drift only
- t0v_space_time    : predictors in both drift and t0
- t0va_space_time   : as above + a small boundary control term

Interpretation guide
--------------------
Support for the space-time tradeoff is strongest if:
1) t0_space_time beats baseline and v_space_time in out-of-sample fit (e.g., LOO), and
2) its key coefficients have the expected signs:
   - full_hit < 0 on t0          (exact cache hits speed retrieval)
   - neighbor_hit < 0 on t0      (partial reuse speeds retrieval)
   - log1p_menu_lag > 0 on t0    (stale menu retrieval is slower)
   - diversity_prop > 0 on t0    (interference slows retrieval)
   - diversity_x_logM > 0 on t0  (interference penalty grows with menu-space size)
   - full_hit_x_logM < 0 on t0   (per-hit benefit grows as menu-space size grows)

Usage
-----
python fit_ddm_spacetime_hier.py \
  --data_dir . \
  --states 3 4 5 \
  --variant t0_space_time \
  --K 1 \
  --warmup 500 --draws 500 \
  --outdir ddm_spacetime_out
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import time
from datetime import datetime
from pathlib import Path

import arviz as az
from cmdstanpy import CmdStanModel

from mtst_ddm_spacetime_covariates import build_pooled_stage1_ddm_data, make_stan_data, get_variant_designs


def _prepare_local_build(stan_file: str, tag: str) -> tuple[Path, Path]:
    src = Path(stan_file).resolve()
    scratch_root = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
    run_root = scratch_root / f"stanrun_{os.getpid()}_{int(time.time())}_{tag}"
    run_root.mkdir(parents=True, exist_ok=True)
    dst = run_root / src.name
    shutil.copyfile(src, dst)
    with open(dst, "rb") as fh:
        os.fsync(fh.fileno())
    os.environ.setdefault("MAKEFLAGS", "-j1")
    return dst, run_root


def _compile_on_scratch(stan_file: str, tag: str) -> tuple[CmdStanModel, Path]:
    stan_scratch, run_dir = _prepare_local_build(stan_file, tag)
    model = CmdStanModel(stan_file=str(stan_scratch))
    return model, run_dir


def _cleanup_run_dir(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return
    for p in sorted(run_dir.rglob("*"), key=lambda x: (x.is_dir(), len(x.as_posix())), reverse=True):
        try:
            if p.is_file() or p.is_symlink():
                for _ in (1, 2, 3):
                    try:
                        p.unlink(missing_ok=True)
                        break
                    except PermissionError:
                        try:
                            os.chmod(p, p.stat().st_mode | stat.S_IWUSR)
                        except Exception:
                            pass
                        time.sleep(0.2)
            elif p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
        except Exception:
            pass
    shutil.rmtree(run_dir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stan", type=str, default="ddm_spacetime_hier.stan")
    ap.add_argument("--data_dir", type=str, default=".")
    ap.add_argument("--states", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--variant", type=str, default="t0_space_time",
                    choices=["baseline", "t0_time", "t0_space_time", "v_space_time", "t0v_space_time", "t0va_space_time"])
    ap.add_argument("--K", type=int, default=1, help="LRU menu cache capacity K.")
    ap.add_argument("--outdir", type=str, default="ddm_spacetime_out")

    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--adapt_delta", type=float, default=0.98)
    ap.add_argument("--max_treedepth", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    csv_by_S = {S: data_dir / f"hddm2_fixed_final_{S}states.csv" for S in args.states}

    pooled = build_pooled_stage1_ddm_data(csv_by_S=csv_by_S, states=args.states, K=args.K)
    stan_data = make_stan_data(pooled, variant=args.variant)

    v_cols, t0_cols, a_cols = get_variant_designs(pooled.df, args.variant)
    meta = {
        "variant": args.variant,
        "states": args.states,
        "K": args.K,
        "N": int(len(pooled.df)),
        "J": int(pooled.df["subj_global"].nunique()),
        "drift_names": v_cols,
        "t0_names": t0_cols,
        "a_names": a_cols,
    }
    (outdir / "predictor_names.json").write_text(json.dumps(meta, indent=2))

    tag = f"spacetime_{args.variant}_K{args.K}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model, run_dir = _compile_on_scratch(args.stan, tag=tag)

    try:
        fit = model.sample(
            data=stan_data,
            chains=args.chains,
            parallel_chains=args.chains,
            iter_warmup=args.warmup,
            iter_sampling=args.draws,
            seed=args.seed,
            adapt_delta=args.adapt_delta,
            max_treedepth=args.max_treedepth,
            show_console=False,
            output_dir=str(run_dir),
        )

        idata = az.from_cmdstanpy(posterior=fit, log_likelihood="log_lik")
        summ = az.summary(
            idata,
            var_names=[
                "mu_log_a", "sigma_log_a",
                "mu_w_logit", "sigma_w_logit",
                "mu_v0", "sigma_v0",
                "mu_eta_t0", "sigma_eta_t0",
                "b_v", "b_t0", "b_a",
            ],
            hdi_prob=0.95,
        )

        nc_path = outdir / f"ddm_spacetime_{args.variant}_K{args.K}.nc"
        csv_path = outdir / f"ddm_spacetime_{args.variant}_K{args.K}_summary.csv"
        az.to_netcdf(idata, nc_path)
        summ.to_csv(csv_path)

        try:
            loo = az.loo(idata)
            (outdir / f"ddm_spacetime_{args.variant}_K{args.K}_loo.json").write_text(loo.to_json())
        except Exception:
            pass

        print("\n=== Key slopes ===")
        print(summ.filter(like="b_").to_string())
        print(f"\nSaved:\n  {nc_path}\n  {csv_path}")

    finally:
        _cleanup_run_dir(run_dir)


if __name__ == "__main__":
    main()
