
#!/usr/bin/env python3
"""
fit_ddm_cache_hier.py

Hierarchical DDM regression (Stan Wiener likelihood) across subjects.
This is the cleanest "mechanism" check to support your behavioral results:

- drift contains both separated/component signals and menu/conjunctive cache signals
- t0 regression contains menu-graph distance / menu-lag (retrieval/caching)
- boundary regression can capture caution/conflict

This model partial-pools subject baselines (a0, w, v0, t0 intercept) while keeping slopes shared.

Because hierarchical Wiener models can be slow, start with:
  --chains 4 --warmup 500 --draws 500
and/or fit a subset of subjects via --max_subjects.

Usage:
  python fit_ddm_cache_hier.py --csv hddm2_fixed_final_5states.csv --states 5 \
    --stan ddm_cache_regression_hier.stan --outdir stan_ddm_hier_out
"""

from __future__ import annotations
import argparse, os, time, shutil, stat
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
    ap.add_argument("--stan", type=str, default="ddm_cache_regression_hier.stan")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--states", type=int, required=True)
    ap.add_argument("--outdir", type=str, default="stan_ddm_hier_out")
    ap.add_argument("--max_subjects", type=int, default=0, help="0=all subjects; else take first N.")

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
    subj_ids = sorted(df["participant_id"].unique().tolist())
    if args.max_subjects and args.max_subjects > 0:
        subj_ids = subj_ids[: args.max_subjects]

    # build and stack
    rt_all, choice_all, subj_index_all = [], [], []
    Xv_all, Xt0_all, Xa_all = [], [], []
    t0_upper = []

    # first pass to ensure consistent predictor sizes
    first = build_subject_covariates(df, subj=subj_ids[0], S=args.states)
    K_v = first.x_drift.shape[1]
    K_t0 = first.x_tau.shape[1]
    K_a = first.x_a.shape[1]

    drift_names = first.drift_names
    tau_names = first.tau_names
    a_names = first.a_names

    for j, sid in enumerate(subj_ids, start=1):
        sd = build_subject_covariates(df, subj=sid, S=args.states)
        assert sd.x_drift.shape[1] == K_v
        assert sd.x_tau.shape[1] == K_t0
        assert sd.x_a.shape[1] == K_a

        rt_all.append(sd.rt.astype(float))
        choice_all.append(sd.choice.astype(int))
        subj_index_all.append(np.full(sd.N, j, dtype=int))
        Xv_all.append(sd.x_drift.astype(float))
        Xt0_all.append(sd.x_tau.astype(float))
        Xa_all.append(sd.x_a.astype(float))

        t0_upper.append(sd.t0_upper)

    rt = np.concatenate(rt_all)
    choice = np.concatenate(choice_all)
    subj_id = np.concatenate(subj_index_all)
    X_v = np.vstack(Xv_all)
    X_t0 = np.vstack(Xt0_all)
    X_a = np.vstack(Xa_all)

    stan_data = {
        "N": int(len(rt)),
        "J": int(len(subj_ids)),
        "subj_id": subj_id.astype(int),

        "rt": rt.astype(float),
        "choice": choice.astype(int),

        "K_v": int(K_v),
        "X_v": X_v.astype(float),

        "K_t0": int(K_t0),
        "X_t0": X_t0.astype(float),

        "K_a": int(K_a),
        "X_a": X_a.astype(float),

        "t0_lower": float(first.t0_lower),
        "t0_upper": np.asarray(t0_upper, dtype=float),
    }

    # Save predictor names for reference
    (outdir / "predictor_names.json").write_text(
        pd.Series({
            "drift": drift_names,
            "t0": tau_names,
            "a": a_names,
        }).to_json()
    )

    tag = f"hier_S{args.states}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
                "mu_w", "sigma_w",
                "mu_v0", "sigma_v0",
                "mu_eta_t0", "sigma_eta_t0",
                "b_v", "b_t0", "b_a",
            ],
            hdi_prob=0.95,
        )

        nc_path = outdir / f"ddm_hier_S{args.states}.nc"
        csv_path = outdir / f"ddm_hier_S{args.states}_summary.csv"
        az.to_netcdf(idata, nc_path)
        summ.to_csv(csv_path)

        print("\n=== Posterior summary (key slopes) ===")
        print(summ.filter(like="b_").to_string())
        print(f"\nSaved:\n  {nc_path}\n  {csv_path}\n")

    finally:
        _cleanup_run_dir(run_dir)


if __name__ == "__main__":
    main()
