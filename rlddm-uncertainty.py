#!/usr/bin/env python3
"""
Drop-in minimal changes:
- Adds two alternative uncertainty models:
  (A) Kalman filter over second-stage options (per planet, 2 actions) -> U_val_z
  (B) Menu-graph uncertainty (Markov entropy or node-frequency variance) -> U_menu_z
- Chooses via --uncertainty_model {kalman,menu,both,beta}; unused covariates passed as zeros.
- Everything else (CLI, IO, drift definition, boundary decline, t0 regression structure) stays intact.
"""
import argparse, os, sys, shutil, glob
from pathlib import Path
from itertools import combinations
from datetime import datetime

import numpy as np
import pandas as pd
import arviz as az
from cmdstanpy import CmdStanModel


# ---------------- helpers (unchanged) ----------------

def print_latest_console_logs(tmpdir="cmdstan_tmp", max_bytes=200000):
    try:
        files = sorted(glob.glob(os.path.join(tmpdir, "*-stdout.txt")),
                       key=os.path.getmtime, reverse=True)
        if not files:
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


# ---------------- IO & preprocessing (unchanged) ----------------

def load_subject_csv(csv_path: Path, subj: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rename = {}
    if "subj_idx"  in df.columns: rename["subj_idx"]  = "participant_id"
    if "rt1"       in df.columns: rename["rt1"]       = "rt"
    if "response1" in df.columns: rename["response1"] = "choice"
    if "response2" in df.columns: rename["response2"] = "choice2"
    if "feedback"  in df.columns: rename["feedback"]  = "reward"
    df = df.rename(columns=rename)

    required = ["participant_id", "rt", "choice", "choice2", "state1", "state2", "reward"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing}\nColumns present: {list(df.columns)}")

    sdf = df[df["participant_id"].astype(int) == int(subj)].copy()
    if sdf.empty:
        raise SystemExit(f"No rows for participant_id={subj} in {csv_path}")

    sdf["rt"] = sdf["rt"].astype(float)
    if np.nanmedian(sdf["rt"].to_numpy()) > 5.0:
        print("NOTE: RTs look like ms; converting to seconds.", file=sys.stderr)
        sdf["rt"] = sdf["rt"] / 1000.0
    if (sdf["rt"] <= 0).any():
        raise SystemExit("Non-positive RT detected.")

    # Avoid Wiener numerical issues
    n_before = len(sdf)
    sdf = sdf[sdf["rt"] >= 0.15].copy()
    n_after = len(sdf)
    if n_before != n_after:
        print(f"NOTE: Filtered {n_before - n_after} trials with RT < 0.15s.", file=sys.stderr)

    # Recode choices to 0/1
    for col in ["choice","choice2"]:
        vals = np.unique(sdf[col].to_numpy())
        if set(vals).issubset({0,1}):
            sdf[col] = sdf[col].astype(int)
        elif set(vals) == {-1,1}:
            sdf[col] = ((sdf[col]+1)//2).astype(int)
        elif set(vals) == {1,2}:
            sdf[col] = (sdf[col]-1).astype(int)
        else:
            raise SystemExit(f"{col} must be 0/1-coded (found {vals}).")

    sdf["reward"] = np.clip(sdf["reward"].astype(float), 0.0, 1.0)
    sdf["state1"] = sdf["state1"].astype(int)
    sdf["state2"] = sdf["state2"].astype(int)

    ensure_no_nan_inf(sdf, ["rt","choice","choice2","state1","state2","reward"])
    sdf = sdf.reset_index(drop=True)
    sdf["trial"] = np.arange(1, len(sdf)+1)
    return sdf


def build_mb_columns(sdf: pd.DataFrame, S: int) -> pd.DataFrame:
    """
    Use state1 (pair index) to build mb1/mb2 (1..S) and s2 (1..S).
    """
    sdf = sdf.copy()
    S = int(S)

    pairs = np.array(list(combinations(np.arange(1, S + 1), 2)))  # (C, 2)
    C = pairs.shape[0]
    idx = sdf["state1"].to_numpy()

    # integer-like
    if not np.issubdtype(idx.dtype, np.integer):
        if np.allclose(idx, np.round(idx), atol=1e-8):
            idx = np.round(idx).astype(int)
        else:
            raise SystemExit("state1 must be an integer-like pair index.")

    # treat as 1-based if all in [1..C]
    if idx.min() >= 1 and idx.max() <= C:
        idx = idx - 1
    if idx.min() < 0 or idx.max() >= C:
        raise SystemExit(f"state1 index out of bounds for S={S}. Valid 0..{C-1}/1..{C}.")

    chosen = pairs[idx]  # (N, 2)
    sdf["mb1"] = chosen[:, 0].astype(int)
    sdf["mb2"] = chosen[:, 1].astype(int)

    # s2 as 1..S (support 0..S-1 or 1..S in raw)
    raw_s2 = sdf["state2"].astype(int).to_numpy()
    if raw_s2.min() == 0 and raw_s2.max() == (S - 1):
        s2 = raw_s2 + 1
    elif raw_s2.min() == 1 and raw_s2.max() == S:
        s2 = raw_s2
    else:
        raise SystemExit(f"state2 must be 0..{S-1} or 1..{S}; got {sorted(np.unique(raw_s2))}")
    sdf["s2"] = s2.astype(int)
    return sdf


def rt_upper_limit(sdf: pd.DataFrame) -> float:
    rt_min = float(np.min(sdf["rt"].to_numpy()))
    safety_margin = max(0.01, 0.1 * rt_min)
    upper = max(0.031, min(rt_min - 1e-6, rt_min - safety_margin))
    print(f"NOTE: t0 upper bound set to {upper:.4f}s (min RT: {rt_min:.4f}s)", file=sys.stderr)
    return upper


# ---------------- Shared covariates (Johnson distance & recency) ----------------

def build_reconfig_recency(sdf: pd.DataFrame, S: int) -> pd.DataFrame:
    sdf = sdf.copy()
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
                lag = t - last
                lmin = lag if np.isnan(lmin) else min(lmin, lag)
        lag_min_ship.append(lmin)

        pkey = tuple(sorted(cur))
        last_p = last_seen_pair.get(pkey, None)
        pair_lag.append(np.nan if last_p is None else (t - last_p))

        last_seen_ship[cur[0]] = t; last_seen_ship[cur[1]] = t
        last_seen_pair[pkey] = t
        prev = cur

    sdf["d1"] = (pd.Series(johnson) == 1).astype(float).fillna(0.0)
    sdf["d2"] = (pd.Series(johnson) == 2).astype(float).fillna(0.0)
    sdf["log1p_ship_lag"] = np.log1p(pd.Series(lag_min_ship)).fillna(0.0)
    sdf["log1p_pair_lag"] = np.log1p(pd.Series(pair_lag)).fillna(0.0)

    # z-scores
    def z(x):
        x = x.astype(float)
        mu = float(np.mean(x)); sd = float(np.std(x))
        return (x - mu) / (sd if sd > 0 else 1.0)
    sdf["log1p_ship_lag_z"] = z(sdf["log1p_ship_lag"])
    sdf["log1p_pair_lag_z"] = z(sdf["log1p_pair_lag"])

    # trial scaling
    nT = len(sdf)
    sdf["trial_scaled"] = (sdf["trial"] - 1) / max(1, (nT - 1))
    return sdf


# ---------------- (A) Kalman filter value-uncertainty ----------------

def softmax_weights(mu_a, mu_b, beta2):
    ea = np.exp(beta2 * mu_a)
    eb = np.exp(beta2 * mu_b)
    Z = ea + eb
    wa = ea / Z
    wb = eb / Z
    return wa, wb

def build_uncertainty_kalman(sdf: pd.DataFrame, S: int, p_common: float, beta2: float,
                             q: float, r: float, p0: float) -> pd.Series:
    """
    Kalman filter per planet and action (2 actions) with random-walk process (q) and obs noise (r).
    Rewards in [0,1] are treated with constant observation variance r (assumed-density KF).

    Returns U_val_z (z-scored): planning-uncertainty at Stage-1 from KF variances:
      Var_planet(s) = w_a^2 * P[s,0] + w_b^2 * P[s,1]  (independence assumption)
      U_i = 0.7^2 Var_planet(i) + 0.3^2 Var_planet(j)
      U_chosen = U_i if left chosen else symmetric for right.
    """
    sdf = sdf.copy()
    # mean and variance for each planet s and action a∈{0,1}
    MU = np.full((S, 2), 0.5, dtype=float)
    P  = np.full((S, 2), p0,  dtype=float)

    U_val = []

    # action coding: choice2 ∈ {0,1} indicates which 2nd-stage action was taken
    for t in range(len(sdf)):
        i = sdf.loc[t, "mb1"] - 1
        j = sdf.loc[t, "mb2"] - 1
        chosen_first = i if sdf.loc[t, "choice"] == 0 else j

        # Softmax policy at each planet from current KF means
        wa_i, wb_i = softmax_weights(MU[i,0], MU[i,1], beta2)
        wa_j, wb_j = softmax_weights(MU[j,0], MU[j,1], beta2)

        # Planet value variance (softmax mixture, ignore covariance)
        Var_planet_i = wa_i**2 * P[i,0] + wb_i**2 * P[i,1]
        Var_planet_j = wa_j**2 * P[j,0] + wb_j**2 * P[j,1]

        # Planning uncertainty for each first-stage option through 70/30
        U_left  = (p_common**2) * Var_planet_i + ((1 - p_common)**2) * Var_planet_j
        U_right = (p_common**2) * Var_planet_j + ((1 - p_common)**2) * Var_planet_i
        U_val.append(U_left if chosen_first == i else U_right)

        # ====== After outcome: update KF for the visited planet & chosen action ======
        s2   = int(sdf.loc[t, "s2"]) - 1          # 0..S-1
        a2   = int(sdf.loc[t, "choice2"])         # 0 or 1
        rew  = float(sdf.loc[t, "reward"])

        # Predict
        P[s2, a2] = P[s2, a2] + q

        # Update
        K = P[s2, a2] / (P[s2, a2] + r)
        MU[s2, a2] = MU[s2, a2] + K * (rew - MU[s2, a2])
        P[s2, a2]  = (1 - K) * P[s2, a2]

    # z-score
    U_val = pd.Series(U_val, index=sdf.index, name="U_val")
    mu = float(U_val.mean()); sd = float(U_val.std())
    U_val_z = (U_val - mu) / (sd if sd > 0 else 1.0)
    return U_val_z


# ---------------- (B) Menu-graph uncertainty ----------------

def entropy(p):
    p = np.asarray(p, dtype=float)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = np.clip(p / s, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())

def build_uncertainty_menu(sdf: pd.DataFrame, S: int, mode: str, alpha0: float) -> pd.Series:
    """
    Menu-graph uncertainty.
      - mode='markov': Dirichlet row for current node's outgoing next-menu distribution,
                       U_menu[t] = entropy( row_alpha / sum(row_alpha) ) at the *current* node.
      - mode='nodefreq': Dirichlet over global node frequencies,
                         U_menu[t] = posterior variance of p(current node).
    Robust to 0-based or 1-based `state1` indices.
    """
    sdf = sdf.copy()

    # Build the lexicographic list of menu nodes (pairs), 1-based for readability
    pairs = list(combinations(range(1, S + 1), 2))
    C = len(pairs)  # number of menu nodes = S*(S-1)/2

    # Normalize state1 to 0-based node indices
    idx_raw = sdf["state1"].astype(int).to_numpy()
    if idx_raw.min() >= 1 and idx_raw.max() <= C:
        idx0 = idx_raw - 1                       # 1..C  ->  0..C-1
    elif idx_raw.min() >= 0 and idx_raw.max() < C:
        idx0 = idx_raw                           # 0..C-1 already
    else:
        raise SystemExit(f"state1 out of bounds for S={S} "
                         f"(expected 0..{C-1} or 1..{C}, got min={idx_raw.min()}, max={idx_raw.max()})")

    U_vec = []

    if mode == "markov":
        # Dirichlet counts over transitions between nodes (C x C)
        A = np.full((C, C), float(alpha0), dtype=float)
        for t in range(len(sdf)):
            cur = int(idx0[t])
            # Uncertainty of the *current* node's outgoing distribution
            U_vec.append(entropy(A[cur, :]))
            # After scoring uncertainty at t, update the transition from previous node → current
            if t > 0:
                prev = int(idx0[t - 1])
                A[prev, cur] += 1.0

    elif mode == "nodefreq":
        # Dirichlet over node frequencies (C-vector)
        alpha = np.full(C, float(alpha0), dtype=float)
        alpha_sum = float(alpha.sum())
        for t in range(len(sdf)):
            cur = int(idx0[t])
            a_i = alpha[cur]
            # Var[p_i] = (α_i (α_sum - α_i)) / (α_sum^2 (α_sum + 1))
            var_i = (a_i * (alpha_sum - a_i)) / (alpha_sum**2 * (alpha_sum + 1.0))
            U_vec.append(float(var_i))
            # Update node count
            alpha[cur] += 1.0
            alpha_sum  += 1.0

    else:
        raise ValueError(f"Unknown menu uncertainty mode: {mode}")

    U_menu = pd.Series(U_vec, index=sdf.index, name="U_menu")
    # z-score
    mu = float(U_menu.mean()); sd = float(U_menu.std())
    U_menu_z = (U_menu - mu) / (sd if sd > 0 else 1.0)
    return U_menu_z


# ---------------- Legacy Beta-Bernoulli value-uncertainty (optional) ----------------

def build_uncertainty_beta(sdf: pd.DataFrame, S: int, p_common: float) -> pd.Series:
    alpha = np.ones(S); beta = np.ones(S)
    U = []
    for t in range(len(sdf)):
        i = sdf.loc[t, "mb1"] - 1
        j = sdf.loc[t, "mb2"] - 1
        chosen = i if sdf.loc[t, "choice"] == 0 else j
        V_var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1))
        U_i = (p_common**2) * V_var[i] + ((1 - p_common)**2) * V_var[j]
        U_j = (p_common**2) * V_var[j] + ((1 - p_common)**2) * V_var[i]
        U.append(U_i if chosen == i else U_j)
        s2 = int(sdf.loc[t, "s2"]) - 1
        r  = float(sdf.loc[t, "reward"])
        alpha[s2] += r
        beta[s2]  += (1 - r)
    U = pd.Series(U, index=sdf.index, name="U_beta")
    mu = float(U.mean()); sd = float(U.std())
    return (U - mu) / (sd if sd > 0 else 1.0)


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ap.add_argument("--stan", type=str, default="rlddm_covariates_single_subject.stan")
    ap.add_argument("--stan", type=str, default="rlddm_uncertainty.stan")
    ap.add_argument("--csv",  type=str, required=True)
    ap.add_argument("--subj", type=int, required=True)
    ap.add_argument("--states", type=int, required=True)
    ap.add_argument("--outdir", type=str, default="stan_covs_out")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--draws",  type=int, default=1000)
    ap.add_argument("--adapt_delta", type=float, default=0.95)
    ap.add_argument("--max_treedepth", type=int, default=12)
    ap.add_argument("--metric", type=str, default="diag_e", choices=["diag_e","dense_e"])
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--p_common", type=float, default=0.7)

    # NEW: which uncertainty to use in t0
    ap.add_argument("--uncertainty_model", type=str, default="kalman",
                    choices=["kalman","menu","both","beta"],
                    help="Pick which uncertainty enters t0: Kalman, menu-graph, both, or legacy beta-variance.")

    # NEW: Kalman hyperparameters
    ap.add_argument("--kf_beta2", type=float, default=4.0, help="Stage-2 softmax inverse temperature")
    ap.add_argument("--kf_q", type=float, default=0.005, help="Process variance per trial")
    ap.add_argument("--kf_r", type=float, default=0.25, help="Observation variance")
    ap.add_argument("--kf_p0", type=float, default=0.25, help="Initial variance for each action")

    # NEW: Menu-graph hyperparameters
    ap.add_argument("--menu_mode", type=str, default="markov", choices=["markov","nodefreq"],
                    help="Menu uncertainty mode: Markov row entropy or node-frequency variance")
    ap.add_argument("--menu_alpha0", type=float, default=1.0, help="Dirichlet prior pseudo-count")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    Path("cmdstan_tmp").mkdir(exist_ok=True)

    sdf = load_subject_csv(csv_path, args.subj)
    sdf = build_mb_columns(sdf, args.states)
    sdf = build_reconfig_recency(sdf, args.states)

    # ----- Uncertainty channels (fill unused with zeros) -----
    zeros = pd.Series(np.zeros(len(sdf)), index=sdf.index)
    U_chosen_z = zeros.copy()  # legacy
    U_val_z    = zeros.copy()  # Kalman
    U_menu_z   = zeros.copy()  # menu graph

    if args.uncertainty_model == "kalman":
        U_val_z = build_uncertainty_kalman(
            sdf, S=args.states, p_common=args.p_common,
            beta2=args.kf_beta2, q=args.kf_q, r=args.kf_r, p0=args.kf_p0
        )
    elif args.uncertainty_model == "menu":
        U_menu_z = build_uncertainty_menu(
            sdf, S=args.states, mode=args.menu_mode, alpha0=args.menu_alpha0
        )
    elif args.uncertainty_model == "both":
        U_val_z  = build_uncertainty_kalman(
            sdf, S=args.states, p_common=args.p_common,
            beta2=args.kf_beta2, q=args.kf_q, r=args.kf_r, p0=args.kf_p0
        )
        U_menu_z = build_uncertainty_menu(
            sdf, S=args.states, mode=args.menu_mode, alpha0=args.menu_alpha0
        )
    else:  # "beta"
        U_chosen_z = build_uncertainty_beta(sdf, S=args.states, p_common=args.p_common)

    rt_upper_t0 = rt_upper_limit(sdf)

    stan_data = {
        "N": len(sdf),
        "rt": sdf["rt"].astype(float).to_numpy(),
        "choice": sdf["choice"].astype(int).to_numpy(),
        "S": int(args.states),
        "mb1": sdf["mb1"].astype(int).to_numpy(),
        "mb2": sdf["mb2"].astype(int).to_numpy(),
        "s2":  sdf["s2"].astype(int).to_numpy(),
        "choice2": sdf["choice2"].astype(int).to_numpy(),
        "reward": sdf["reward"].astype(float).to_numpy(),
        "rt_upper_t0": float(rt_upper_t0),
        "t0_lower": 0.03,

        # core covariates
        "U_chosen_z": U_chosen_z.astype(float).to_numpy(),
        "d1": sdf["d1"].astype(float).to_numpy(),
        "d2": sdf["d2"].astype(float).to_numpy(),
        "log1p_ship_lag_z": sdf["log1p_ship_lag_z"].astype(float).to_numpy(),
        "log1p_pair_lag_z": sdf["log1p_pair_lag_z"].astype(float).to_numpy(),
        "trial_scaled": sdf["trial_scaled"].astype(float).to_numpy(),

        # NEW: alternative uncertainty covariates
        "U_val_z":  U_val_z.astype(float).to_numpy(),
        "U_menu_z": U_menu_z.astype(float).to_numpy(),

        "p_common": float(args.p_common),
    }

    print(f"\nData summary → N={len(sdf)}, S={args.states}, uncertainty={args.uncertainty_model}")
    print("mb1 range:", int(sdf["mb1"].min()), "…", int(sdf["mb1"].max()))
    print("mb2 range:", int(sdf["mb2"].min()), "…", int(sdf["mb2"].max()))
    print("s2 unique:", sorted(sdf["s2"].unique().tolist()))
    print("choice counts:", sdf["choice"].value_counts().to_dict())
    print("choice2 counts:", sdf["choice2"].value_counts().to_dict(), "\n")

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
            show_console=False
        )
    except Exception:
        print("\nSampling failed — showing latest Stan console log:\n", file=sys.stderr)
        print_latest_console_logs("cmdstan_tmp")
        raise

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(fit.runset.csv_files, start=1):
        if src and Path(src).exists():
            shutil.copyfile(src, outdir / f"subject{args.subj}_chain{i}.csv")

    try:
        idata = az.from_cmdstanpy(posterior=fit)
        summ = az.summary(
            idata,
            var_names=["alpha","a","t0","w","scaler","k_decline",
                       "b_U","b_Uval","b_Umenu","b_d1","b_d2","b_ship","b_pair"],
            hdi_prob=0.95
        )
        print("\n", summ.to_string())
        az.to_netcdf(idata, outdir / f"subject{args.subj}.nc")
        summ.to_csv(outdir / f"subject{args.subj}-summary.csv")
        print(f"Saved outputs under: {outdir}")
    except Exception as e:
        print(f"[NOTE] ArviZ failure (non-fatal): {e}", file=sys.stderr)

    # Cleanup
    try:
        shutil.rmtree(subject_tmp_dir, ignore_errors=True)
    except Exception as e:
        print(f"Temp cleanup issue: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
