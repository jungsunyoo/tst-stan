#!/usr/bin/env python3
import argparse, sys, shutil, glob, os
from pathlib import Path
from itertools import combinations
from datetime import datetime

import numpy as np
import pandas as pd
import arviz as az
import pdb
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
    
    # Filter out problematically fast RTs that cause numerical issues
    n_before = len(sdf)
    sdf = sdf[sdf["rt"] >= 0.15].copy()  # Remove RTs < 150ms
    n_after = len(sdf)
    if n_before != n_after:
        print(f"NOTE: Filtered {n_before - n_after} trials with RT < 0.15s to avoid numerical issues.", file=sys.stderr)

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
    sdf = sdf.reset_index(drop=True)
    sdf["trial"] = np.arange(1, len(sdf)+1)
    return sdf


def build_mb_columns(sdf: pd.DataFrame, S: int) -> pd.DataFrame:
    """
    ALWAYS:
      - Keep state1/state2 unchanged.
      - Build mb1/mb2 from the combination indexed by state1 (pair index).
      - Build s2 for Stan as 1..S, auto-detecting state2 coding (0..S-1 or 1..S).
    """
    sdf = sdf.copy()
    S = int(S)

    pairs = np.array(list(combinations(np.arange(1, S + 1), 2)))  # (C, 2)
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

    # Build s2 as 1..S from state2 (support 0..S-1 and 1..S)
    raw_s2 = sdf["state2"].astype(int).to_numpy()
    if raw_s2.min() == 0 and raw_s2.max() == (S - 1):
        s2 = raw_s2 + 1
    elif raw_s2.min() == 1 and raw_s2.max() == S:
        s2 = raw_s2
    else:
        raise SystemExit(f"state2 must be 0..{S-1} or 1..{S}; found {sorted(np.unique(raw_s2))}.")
    sdf["s2"] = s2.astype(int)
    return sdf


def rt_upper_limit(sdf: pd.DataFrame) -> float:
    """Strict upper bound for t0: ALWAYS < min(rt), with extra safety margin."""
    rt_min = float(np.min(sdf["rt"].to_numpy()))
    safety_margin = max(0.01, rt_min * 0.1)  # 10ms or 10% of min RT, whichever is larger
    upper = rt_min - safety_margin
    upper = max(0.031, min(upper, rt_min - 1e-6))
    if not (upper < rt_min):
        raise SystemExit(f"rt_upper_t0={upper:.6f} is not < min(rt)={rt_min:.6f}. Check RTs.")
    print(f"NOTE: t0 upper bound set to {upper:.4f}s (min RT: {rt_min:.4f}s, safety margin: {safety_margin:.4f}s)", file=sys.stderr)
    return upper


# ---------- NEW: trial-wise covariates for t0 & boundary (minimal) ----------

def build_covariates(sdf: pd.DataFrame, S: int, p_common: float = 0.7) -> pd.DataFrame:
    """
    Adds:
      - U_chosen (planning uncertainty) then z-score -> U_chosen_z
      - Johnson distance d in {0,1,2} from (mb1, mb2) between trials -> d1,d2 dummies
      - Recency lags (ship overlap, exact pair) -> log1p then z-score
      - trial_scaled in [0,1] for between-trials boundary decline
    """
    sdf = sdf.copy()

    # Johnson distance & recency using mb1/mb2 (already built)
    johnson, lag_min_ship, pair_lag = [], [], []
    last_seen_ship = {i: None for i in range(1, S+1)}
    last_seen_pair = {}
    prev = None
    for t in range(len(sdf)):
        cur = (int(sdf.loc[t, "mb1"]), int(sdf.loc[t, "mb2"]))
        if prev is None:
            d = np.nan
        else:
            overlap = len(set(cur).intersection(prev))
            d = 2 - overlap
        johnson.append(d)

        # recency: closest overlap with either ship
        lmin = np.nan
        for s in cur:
            last = last_seen_ship.get(s, None)
            if last is not None:
                lmin = (t - last) if np.isnan(lmin) else min(lmin, t - last)
        lag_min_ship.append(lmin)

        pkey = tuple(sorted(cur))
        last_p = last_seen_pair.get(pkey, None)
        pair_lag.append(np.nan if last_p is None else (t - last_p))

        last_seen_ship[cur[0]] = t; last_seen_ship[cur[1]] = t
        last_seen_pair[pkey] = t
        prev = cur

    sdf["johnson_d"] = johnson
    sdf["d1"] = (sdf["johnson_d"] == 1).astype(float).fillna(0.0)
    sdf["d2"] = (sdf["johnson_d"] == 2).astype(float).fillna(0.0)

    # Recency: log(1+lag); fill first occurrences with 0
    sdf["log1p_ship_lag"] = np.log1p(pd.Series(lag_min_ship)).fillna(0.0)
    sdf["log1p_pair_lag"] = np.log1p(pd.Series(pair_lag)).fillna(0.0)

    # Planning-uncertainty (planet value variance, Beta–Bernoulli)
    alpha = np.ones(S); beta = np.ones(S)
    Uc = []
    s2_zero = sdf["s2"].astype(int).to_numpy() - 1  # s2 is 1..S, convert to 0..S-1
    for t in range(len(sdf)):
        i = int(sdf.loc[t, "mb1"]) - 1
        j = int(sdf.loc[t, "mb2"]) - 1
        chosen = i if int(sdf.loc[t, "choice"]) == 0 else j

        V_mean = alpha/(alpha+beta)
        V_var  = (alpha*beta)/(((alpha+beta)**2)*(alpha+beta+1))

        U_i = (p_common**2)*V_var[i] + ((1-p_common)**2)*V_var[j]
        U_j = (p_common**2)*V_var[j] + ((1-p_common)**2)*V_var[i]
        Uc.append(U_i if chosen == i else U_j)

        # update after outcome
        planet = int(s2_zero[t])
        rew    = float(sdf.loc[t, "reward"])
        alpha[planet] += rew
        beta[planet]  += (1.0 - rew)

    sdf["U_chosen"] = Uc

    # z-scores (robust to constant inputs)
    def z(x):
        x = x.astype(float)
        mu = float(np.mean(x)); sd = float(np.std(x))
        return (x - mu) / (sd if sd > 0 else 1.0)

    sdf["U_chosen_z"]       = z(sdf["U_chosen"])
    sdf["log1p_ship_lag_z"] = z(sdf["log1p_ship_lag"])
    sdf["log1p_pair_lag_z"] = z(sdf["log1p_pair_lag"])

    # for boundary decline (between-trials)
    nT = len(sdf)
    sdf["trial_scaled"] = (sdf["trial"] - 1) / max(1, (nT - 1))
    return sdf


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stan", type=str, default="rlddm_single_subject.stan", help="Path to Stan model")
    ap.add_argument("--csv", type=str, required=True, help="Data CSV")
    ap.add_argument("--subj", type=int, required=True, help="participant_id to fit")
    ap.add_argument("--states", type=int, default=2, help="Number of second-stage states S (2/3/4/5)")
    ap.add_argument("--outdir", type=str, default="stan_out", help="Directory to copy per-chain CSVs")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--adapt_delta", type=float, default=0.95)
    ap.add_argument("--max_treedepth", type=int, default=12)
    ap.add_argument("--metric", type=str, default="diag_e", choices=["diag_e", "dense_e"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--show-console", action="store_true", help="Show Stan console while sampling")
    ap.add_argument("--p_common", type=float, default=0.7, help="Common transition probability (usually 0.7)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    Path("cmdstan_tmp").mkdir(exist_ok=True)

    # Load & prep
    sdf = load_subject_csv(csv_path, args.subj)
    sdf = build_mb_columns(sdf, args.states)

    # --- NEW (1 line): build trial-wise covariates ---
    sdf = build_covariates(sdf, S=args.states, p_common=args.p_common)

    rt_upper_t0 = rt_upper_limit(sdf)

    # Stan data (ONLY the extra fields below are new)
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

        # --- NEW fields passed to Stan ---
        "U_chosen_z":        sdf["U_chosen_z"].astype(float).to_numpy(),
        "d1":                sdf["d1"].astype(float).to_numpy(),
        "d2":                sdf["d2"].astype(float).to_numpy(),
        "log1p_ship_lag_z":  sdf["log1p_ship_lag_z"].astype(float).to_numpy(),
        "log1p_pair_lag_z":  sdf["log1p_pair_lag_z"].astype(float).to_numpy(),
        "trial_scaled":      sdf["trial_scaled"].astype(float).to_numpy(),
        "t0_lower":          0.03,
        "p_common":          float(args.p_common),
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subject_tmp_dir = Path("cmdstan_tmp") / f"subject_{args.subj}_{timestamp}"
    subject_tmp_dir.mkdir(parents=True, exist_ok=True)
    
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
            output_dir=str(subject_tmp_dir),
            show_console=args.show_console,
        )
    except Exception:
        print("\nSampling failed — showing latest Stan console log:\n", file=sys.stderr)
        print_latest_console_logs("cmdstan_tmp")
        raise

    # Copy per-chain CSVs with subject ID in filename
    for i, src in enumerate(fit.runset.csv_files, start=1):
        if src and Path(src).exists():
            dest = outdir / f"subject{args.subj}_chain{i}.csv"
            # shutil.copyfile(src, dest)
            # print(f"Saved: {dest}")
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
        # Minimal, safe set of parameters (adjust if you like)
        summ = az.summary(
            idata,
            var_names=["alpha","a","t0","w","scaler","k_decline","b_U","b_d1","b_d2","b_ship","b_pair"],
            hdi_prob=0.95
        )
        print("\n", summ.to_string())
        az.to_netcdf(idata, outdir / f"subject{args.subj}.nc")
        # summ.to_csv(outdir / f"subject{args.subj}-summary.csv")
        # print(f"\nSaved outputs under: {outdir}")
    except Exception as e:
        print(f"NOTE: Could not build ArviZ InferenceData: {e}", file=sys.stderr)
        print(f"Raw CSVs are in: {outdir}")
    
    # Robust cleanup of temporary directory
    try:
        import time
        time.sleep(0.1)  # Brief pause to let file handles close
        if subject_tmp_dir.exists():
            shutil.rmtree(subject_tmp_dir, ignore_errors=True)
    except Exception as e:
        print(f"NOTE: Temp directory cleanup issue (non-fatal): {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
