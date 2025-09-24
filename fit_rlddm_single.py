#!/usr/bin/env python3
import argparse, sys, shutil, glob, os
from pathlib import Path
from itertools import combinations
from datetime import datetime

import numpy as np
import pandas as pd
import arviz as az
from cmdstanpy import CmdStanModel


# ---------- helpers ----------

def print_latest_console_logs(tmpdir="cmdstan_tmp", max_bytes=200000):
    try:
        files = sorted(glob.glob(os.path.join(tmpdir, "*-stdout.txt")),
                       key=os.path.getmtime, reverse=True)
        if not files:
            print(f"[console] No *-stdout.txt files found in {tmpdir}",
                  file=sys.stderr)
            return
        log_path = files[0]
        with open(log_path, "r", errors="replace") as f:
            data = f.read()
        if len(data) > max_bytes:
            data = data[-max_bytes:]
        print(f"\n======= Stan console ({log_path}) =======\n{data}\n======= end console =======\n",
              file=sys.stderr)
    except Exception as e:
        print(f"[console] Could not read console log: {e}", file=sys.stderr)


def ensure_no_nan_inf(df: pd.DataFrame, cols):
    bad = {}
    for c in cols:
        v = df[c].to_numpy()
        if np.isnan(v).any() or np.isinf(v).any():
            bad[c] = int(np.isnan(v).sum() + np.isinf(v).sum())
    if bad:
        sample = df.loc[(df[cols].isna() | ~np.isfinite(df[cols])).any(axis=1)].head(5)
        raise SystemExit(f"Found NaN/Inf in {bad}. Example rows:\n{sample}")


# ---------- IO & preprocessing ----------

def load_subject_csv(csv_path: Path, subj: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Keep state1/state2 AS-IS; normalize the rest
    rename = {}
    if "subj_idx" in df.columns: rename["subj_idx"] = "participant_id"
    if "rt1" in df.columns: rename["rt1"] = "rt"
    if "response1" in df.columns: rename["response1"] = "choice"
    if "response2" in df.columns: rename["response2"] = "choice2"
    if "feedback" in df.columns: rename["feedback"] = "reward"
    df = df.rename(columns=rename)

    required = ["participant_id", "rt", "choice", "choice2", "state1", "state2", "reward"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing}\nColumns present: {list(df.columns)}")

    sdf = df[df["participant_id"].astype(int) == int(subj)].copy()
    if sdf.empty:
        raise SystemExit(f"No rows for participant_id={subj} in {csv_path}")

    # RTs
    sdf["rt"] = sdf["rt"].astype(float)
    if np.nanmedian(sdf["rt"].to_numpy()) > 5.0:
        print("NOTE: RTs look like milliseconds; converting to seconds.", file=sys.stderr)
        sdf["rt"] = sdf["rt"] / 1000.0
    if (sdf["rt"] <= 0).any():
        bad = sdf[sdf["rt"] <= 0].head(5)
        raise SystemExit(f"Non-positive RT detected. Examples:\n{bad[['rt']]}")

    # Binary recodes to 0/1 for choice, choice2
    for col in ["choice", "choice2"]:
        vals = np.unique(sdf[col].to_numpy())
        if set(vals).issubset({0, 1}):
            sdf[col] = sdf[col].astype(int)
        elif set(vals) == {-1, 1}:
            sdf[col] = ((sdf[col] + 1) // 2).astype(int)
        elif set(vals) == {1, 2}:
            sdf[col] = (sdf[col] - 1).astype(int)
        else:
            raise SystemExit(f"{col} must be coded as 0/1 (found {vals}).")

    # reward in [0,1]
    sdf["reward"] = np.clip(sdf["reward"].astype(float), 0.0, 1.0)

    # Leave state1/state2 AS-IS (just ints)
    sdf["state1"] = sdf["state1"].astype(int)
    sdf["state2"] = sdf["state2"].astype(int)

    ensure_no_nan_inf(sdf, ["rt","choice","choice2","state1","state2","reward"])
    return sdf.reset_index(drop=True)


def build_mb_columns(sdf: pd.DataFrame, S: int) -> pd.DataFrame:
    """
    ALWAYS:
      - Keep state1/state2 unchanged.
      - Build mb1/mb2 from the combination indexed by state1 (pair index).
      - Build s2 for Stan as 1..S, auto-detecting state2 coding (0/1 or 1/2).
    """
    sdf = sdf.copy()
    S = int(S)

    pairs = np.array(list(combinations(np.arange(1, S + 1), 2)))  # shape (C, 2)
    C = pairs.shape[0]
    idx = sdf["state1"].to_numpy()

    # coerce integer-like
    if not np.issubdtype(idx.dtype, np.integer):
        if np.allclose(idx, np.round(idx), atol=1e-8):
            idx = np.round(idx).astype(int)
        else:
            raise SystemExit("state1 must be an integer-like pair index when building mb1/mb2.")

    # treat as 1-based if all in [1..C]
    if idx.min() >= 1 and idx.max() <= C:
        idx = idx - 1
    if idx.min() < 0 or idx.max() >= C:
        raise SystemExit(f"state1 index out of bounds for S={S}. Valid range: 0..{C-1} or 1..{C}.")

    chosen = pairs[idx]  # (N, 2)
    sdf["mb1"] = chosen[:, 0].astype(int)
    sdf["mb2"] = chosen[:, 1].astype(int)

    # Build s2 as 1..S from state2 (auto-detect 0/1 vs 1/2)
    raw_s2 = sdf["state2"].astype(int).to_numpy()
    u = set(np.unique(raw_s2))
    if u.issubset({0, 1}):
        s2 = raw_s2 + 1
    elif u.issubset({1, 2}):
        s2 = raw_s2
    else:
        raise SystemExit(f"state2 must be 0/1 or 1/2 coded; found unique values {sorted(list(u))}.")
    sdf["s2"] = s2.astype(int)

    # Bounds checks
    for col in ["mb1","mb2","s2"]:
        if (sdf[col] < 1).any() or (sdf[col] > S).any():
            bad = sdf[(sdf[col] < 1) | (sdf[col] > S)][["state1","state2","mb1","mb2","s2"]].head(6)
            raise SystemExit(f"{col} out of bounds 1..S (S={S}). Examples:\n{bad}")

    return sdf


def rt_upper_limit(sdf: pd.DataFrame) -> float:
    """Strict upper bound for t0: ALWAYS < min(rt)."""
    rt_min = float(np.min(sdf["rt"].to_numpy()))
    upper = rt_min - 1e-3  # 1 ms below the minimum observed RT
    # keep it above the parameter lower bound (0.03), but NEVER above rt_min
    upper = max(0.031, min(upper, rt_min - 1e-6))
    if not (upper < rt_min):
        raise SystemExit(f"rt_upper_t0={upper:.6f} is not < min(rt)={rt_min:.6f}. Check RTs.")
    return upper


def make_inits(stan_data: dict, chains: int) -> list[dict]:
    rng = np.random.default_rng(2027)
    inits = []
    upper = float(stan_data["rt_upper_t0"])
    # choose a conservative t0 far from min(rt)
    lo = 0.06
    hi = max(0.0305, upper - 0.002)
    if hi <= lo:
        hi = lo + 0.001  # tiny range, still valid
    for _ in range(chains):
        alpha = float(np.clip(rng.normal(0.30, 0.08), 0.02, 0.98))
        a     = float(np.clip(rng.normal(1.20, 0.25), 0.40, 2.30))
        t0    = float(np.clip(rng.uniform(lo, hi), lo, upper - 1e-6))
        scaler= float(np.clip(rng.lognormal(mean=0.0, sigma=0.25), 0.1, 5.0))
        inits.append({"alpha": alpha, "a": a, "t0": t0, "scaler": scaler})
    return inits


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stan", type=str, default="rlddm_single_subject.stan", help="Path to Stan model")
    ap.add_argument("--csv", type=str, required=True, help="Data CSV")
    ap.add_argument("--subj", type=int, required=True, help="participant_id to fit")
    ap.add_argument("--states", type=int, default=2, help="Number of second-stage states S (usually 2)")
    ap.add_argument("--outdir", type=str, default="stan_out", help="Directory to copy per-chain CSVs")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--adapt_delta", type=float, default=0.95)
    ap.add_argument("--max_treedepth", type=int, default=12)
    ap.add_argument("--metric", type=str, default="diag_e", choices=["diag_e", "dense_e"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--show-console", action="store_true", help="Show Stan console while sampling")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    Path("cmdstan_tmp").mkdir(exist_ok=True)

    # Load & prep
    sdf = load_subject_csv(csv_path, args.subj)
    sdf = build_mb_columns(sdf, args.states)
    rt_upper_t0 = rt_upper_limit(sdf)

    # Stan data
    stan_data = {
        "N": len(sdf),
        "rt": sdf["rt"].astype(float).to_numpy(),
        "choice": sdf["choice"].astype(int).to_numpy(),
        "mb1": sdf["mb1"].astype(int).to_numpy(),
        "mb2": sdf["mb2"].astype(int).to_numpy(),
        "S": int(args.states),
        "s2": sdf["s2"].astype(int).to_numpy(),
        "choice2": sdf["choice2"].astype(int).to_numpy(),
        "reward": sdf["reward"].astype(float).to_numpy(),
        "rt_upper_t0": float(rt_upper_t0),
    }

    # Echo (preflight)
    print(f"\nData summary → N={len(sdf)}, S={args.states}")
    print("mb1 range:", int(sdf["mb1"].min()), "…", int(sdf["mb1"].max()))
    print("mb2 range:", int(sdf["mb2"].min()), "…", int(sdf["mb2"].max()))
    print("s2 unique:", sorted(sdf["s2"].unique().tolist()))
    print("choice counts:", sdf["choice"].value_counts().to_dict())
    print("choice2 counts:", sdf["choice2"].value_counts().to_dict(), "\n")

    # Compile & sample
    model = CmdStanModel(stan_file=str(Path(args.stan)))
    init_list = make_inits(stan_data, args.chains)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
            metric=args.metric,
            # inits=init_list,
            output_dir="cmdstan_tmp",
            show_console=args.show_console,
        )
    except Exception:
        print("\nSampling failed — showing latest Stan console log:\n", file=sys.stderr)
        print_latest_console_logs("cmdstan_tmp")
        raise

    # Copy per-chain CSVs
    for i, src in enumerate(fit.runset.csv_files, start=1):
        if src and Path(src).exists():
            dest = outdir / f"{Path(args.stan).stem}-{timestamp}_chain{i}.csv"
            shutil.copyfile(src, dest)
            print(f"Saved: {dest}")
        else:
            print(f"WARNING: CSV for chain {i} not found at {src}", file=sys.stderr)

    # Diagnostics + ArviZ
    try:
        import numpy as np
        print("Step sizes per chain:", np.array(fit.step_size()))
        print("\n" + fit.diagnose())
    except Exception as e:
        print(f"diagnose() unavailable: {e}", file=sys.stderr)

    try:
        idata = az.from_cmdstanpy(posterior=fit)
        summ = az.summary(idata, var_names=["alpha", "a", "t0", "scaler"], hdi_prob=0.95)
        print("\n", summ.to_string())
        az.to_netcdf(idata, outdir / f"{Path(args.stan).stem}-{timestamp}.nc")
        summ.to_csv(outdir / f"{Path(args.stan).stem}-{timestamp}-summary.csv")
        print(f"\nSaved outputs under: {outdir}")
    except Exception as e:
        print(f"NOTE: Could not build ArviZ InferenceData: {e}", file=sys.stderr)
        print(f"Raw CSVs are in: {outdir}")


if __name__ == "__main__":
    main()
