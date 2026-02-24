
#!/usr/bin/env python3
"""
fit_iohmm_ddm_cache_subject.py

Per-subject 2-state IOHMM with DDM (Wiener) emissions in Stan.

State 1 (MENU) vs State 2 (OPTION), with:
  - state-specific drift regression (X_v)
  - state-specific t0 regression (X_t0)
  - state-specific boundary regression (X_a)
  - covariate-dependent transitions via X_tr

This is the DDM analogue of your earlier choice+Gaussian-RT IOHMM.
It lets you say *which DDM component* changes when switching cache modes.

Usage:
  python fit_iohmm_ddm_cache_subject.py \
    --csv hddm2_fixed_final_5states.csv --states 5 --subj 12 \
    --stan iohmm_ddm_cache_subject.stan --outdir stan_iohmm_ddm_out
"""

from __future__ import annotations
import argparse, os, sys, time, shutil, stat
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
    ap.add_argument("--stan", type=str, default="iohmm_ddm_cache_subject.stan")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--states", type=int, required=True)
    ap.add_argument("--subj", type=int, required=True)
    ap.add_argument("--outdir", type=str, default="stan_iohmm_ddm_out")

    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--adapt_delta", type=float, default=0.95)
    ap.add_argument("--max_treedepth", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2027)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_mtst_csv(Path(args.csv))
    sd = build_subject_covariates(df, subj=args.subj, S=args.states)

    stan_data = {
        "N": sd.N,
        "rt": sd.rt.astype(float),
        "choice": sd.choice.astype(int),

        "K_v": sd.x_drift.shape[1],
        "X_v": sd.x_drift.astype(float),

        "K_t0": sd.x_tau.shape[1],
        "X_t0": sd.x_tau.astype(float),

        "K_a": sd.x_a.shape[1],
        "X_a": sd.x_a.astype(float),

        "K_tr": sd.x_trans.shape[1],
        "X_tr": sd.x_trans.astype(float),

        "t0_lower": float(sd.t0_lower),
        "t0_upper": float(sd.t0_upper),
    }

    # save predictor names for reference
    (outdir / f"predictor_names_subject{sd.subj}_S{args.states}.json").write_text(
        pd.Series({
            "drift": sd.drift_names,
            "t0": sd.tau_names,
            "a": sd.a_names,
            "trans": sd.trans_names,
        }).to_json()
    )

    tag = f"iohmm_subj{sd.subj}_S{args.states}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        # focus on transition-rule parameters
        summ = az.summary(
            idata,
            var_names=[
                "pi0_logit",
                "delta_from_menu",
                "delta_from_opt",
                "log_a0", "w_logit",
                "v0", "b_v",
                "eta_t0_0", "b_t0",
            ],
            hdi_prob=0.95,
        )

        nc_path = outdir / f"iohmm_ddm_subject{sd.subj}_S{args.states}.nc"
        csv_path = outdir / f"iohmm_ddm_subject{sd.subj}_S{args.states}_summary.csv"
        az.to_netcdf(idata, nc_path)
        summ.to_csv(csv_path)

        print("\n=== Transition rule (delta) summary ===")
        print(summ.filter(like="delta_").to_string())
        print(f"\nSaved:\n  {nc_path}\n  {csv_path}\n")

    finally:
        _cleanup_run_dir(run_dir)


if __name__ == "__main__":
    main()
