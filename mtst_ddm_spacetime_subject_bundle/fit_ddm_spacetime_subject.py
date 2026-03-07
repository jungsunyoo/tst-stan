#!/usr/bin/env python3
"""
fit_ddm_spacetime_subject.py

Per-subject DDM for the MTST space–time tradeoff.

This is the non-hierarchical counterpart to the pooled/hierarchical modelset.
Use it when you want to:
- fit subjects independently,
- parallelize naturally on SLURM,
- then compare coefficients across SSC (S=3,4,5).

Important identifiability note
------------------------------
Within a single subject × SSC fit, menu-space size M is constant, so `logM`
and `x * logM` interactions are not separately identifiable. Therefore this
subject-level model only includes *time/cache* variables directly:
  - full_hit
  - neighbor_hit
  - menu_dist_lag1
  - log1p_menu_lag
  - diversity_prop

The "space" part of the space–time tradeoff is then tested by comparing the
same subject-level coefficients across SSC (e.g., S=3 vs 4 vs 5).

Recommended variants
--------------------
- baseline
- t0_cache
- v_cache
- t0v_cache
- t0va_cache

Main interpretation
-------------------
Support for the space–time hypothesis is strongest if:
1) t0_cache beats baseline and v_cache in fit / predictive checks, and
2) its t0 coefficients have expected signs:
   - full_hit < 0
   - neighbor_hit < 0
   - log1p_menu_lag > 0
   - diversity_prop > 0
   - menu_dist_lag1 > 0
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

from mtst_ddm_spacetime_subject_covariates import load_mtst_csv, build_subject_spacetime_covariates


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
    ap.add_argument("--stan", type=str, default="ddm_spacetime_subject.stan")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--states", type=int, required=True)
    ap.add_argument("--subj", type=int, required=True)
    ap.add_argument("--variant", type=str, default="t0_cache",
                    choices=["baseline", "t0_cache", "v_cache", "t0v_cache", "t0va_cache"])
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="ddm_spacetime_subject_out")

    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--adapt_delta", type=float, default=0.98)
    ap.add_argument("--max_treedepth", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2027)

    ap.add_argument("--print_names", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_mtst_csv(Path(args.csv))
    subjdat = build_subject_spacetime_covariates(
        df_all=df,
        subj=args.subj,
        S=args.states,
        K=args.K,
        variant=args.variant,
    )

    if args.print_names:
        print("Drift predictors:", subjdat.drift_names)
        print("t0 predictors:", subjdat.t0_names)
        print("a predictors:", subjdat.a_names)
        print("Meta:", json.dumps(subjdat.meta, indent=2))

    stan_data = {
        "N": subjdat.N,
        "rt": subjdat.rt.astype(float),
        "choice": subjdat.choice.astype(int),
        "K_v": subjdat.x_drift.shape[1],
        "X_v": subjdat.x_drift.astype(float),
        "K_t0": subjdat.x_t0.shape[1],
        "X_t0": subjdat.x_t0.astype(float),
        "K_a": subjdat.x_a.shape[1],
        "X_a": subjdat.x_a.astype(float),
        "t0_lower": float(subjdat.t0_lower),
        "t0_upper": float(subjdat.t0_upper),
    }

    meta = {
        "subj": subjdat.subj,
        "S": subjdat.S,
        "K": subjdat.K,
        "variant": args.variant,
        "drift_names": subjdat.drift_names,
        "t0_names": subjdat.t0_names,
        "a_names": subjdat.a_names,
        "meta": subjdat.meta,
    }
    (outdir / f"ddm_spacetime_subject{subjdat.subj}_S{args.states}_{args.variant}_K{args.K}_predictors.json").write_text(
        json.dumps(meta, indent=2)
    )

    tag = f"spacetime_subj{subjdat.subj}_S{args.states}_{args.variant}_K{args.K}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            var_names=["a0", "w", "v0", "b_v", "eta_t0_0", "b_t0", "b_a"],
            hdi_prob=0.95,
        )

        nc_path = outdir / f"ddm_spacetime_subject{subjdat.subj}_S{args.states}_{args.variant}_K{args.K}.nc"
        csv_path = outdir / f"ddm_spacetime_subject{subjdat.subj}_S{args.states}_{args.variant}_K{args.K}_summary.csv"
        az.to_netcdf(idata, nc_path)
        summ.to_csv(csv_path)

        print("\n=== Key slopes ===")
        print(summ.to_string())
        print(f"\nSaved:\n  {nc_path}\n  {csv_path}\n")

    finally:
        _cleanup_run_dir(run_dir)


if __name__ == "__main__":
    main()
