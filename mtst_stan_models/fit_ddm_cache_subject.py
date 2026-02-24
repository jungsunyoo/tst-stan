
#!/usr/bin/env python3
"""
fit_ddm_cache_subject.py

Per-subject DDM regression in Stan (Wiener likelihood) that decomposes
menu-graph effects into:
  - drift (evidence/value) vs
  - nondecision time t0 (retrieval/caching) vs
  - boundary a (caution)

This is designed to complement your behavioral menu-graph analyses.

Usage example:
  python fit_ddm_cache_subject.py \
    --csv /path/to/hddm2_fixed_final_5states.csv \
    --states 5 --subj 12 \
    --stan ddm_cache_regression_subject.stan \
    --outdir stan_ddm_cache_out

Outputs:
  - CmdStan CSVs (in a scratch run dir)
  - ArviZ netcdf + summary CSV in outdir
"""

from __future__ import annotations
import argparse, os, sys, time, shutil, glob, stat
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import arviz as az
from cmdstanpy import CmdStanModel

from mtst_covariates import load_mtst_csv, build_subject_covariates


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
                for attempt in (1, 2, 3):
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
    ap.add_argument("--stan", type=str, default="ddm_cache_regression_subject.stan")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--states", type=int, required=True)
    ap.add_argument("--subj", type=int, required=True)
    ap.add_argument("--outdir", type=str, default="stan_ddm_cache_out")

    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--adapt_delta", type=float, default=0.95)
    ap.add_argument("--max_treedepth", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2027)

    ap.add_argument("--print_names", action="store_true", help="Print predictor names (drift/t0/a).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_mtst_csv(Path(args.csv))
    subjdat = build_subject_covariates(df, subj=args.subj, S=args.states)

    if args.print_names:
        print("Drift predictors:", subjdat.drift_names)
        print("t0 predictors:", subjdat.tau_names)
        print("a predictors:", subjdat.a_names)
        print("transition predictors:", subjdat.trans_names)

    stan_data = {
        "N": subjdat.N,
        "rt": subjdat.rt.astype(float),
        "choice": subjdat.choice.astype(int),

        "K_v": subjdat.x_drift.shape[1],
        "X_v": subjdat.x_drift.astype(float),

        "K_t0": subjdat.x_tau.shape[1],
        "X_t0": subjdat.x_tau.astype(float),

        "K_a": subjdat.x_a.shape[1],
        "X_a": subjdat.x_a.astype(float),

        "t0_lower": float(subjdat.t0_lower),
        "t0_upper": float(subjdat.t0_upper),
    }

    tag = f"subj{args.subj}_S{args.states}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

        # save
        nc_path = outdir / f"ddm_subject{subjdat.subj}_S{args.states}.nc"
        csv_path = outdir / f"ddm_subject{subjdat.subj}_S{args.states}_summary.csv"
        az.to_netcdf(idata, nc_path)
        summ.to_csv(csv_path)

        print("\n=== Posterior summary (head) ===")
        print(summ.head(30).to_string())
        print(f"\nSaved:\n  {nc_path}\n  {csv_path}\n")

    finally:
        _cleanup_run_dir(run_dir)


if __name__ == "__main__":
    main()
