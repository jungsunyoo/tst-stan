#!/usr/bin/env python3
# recover_rlddm_params.py
#
# Parameter recovery for the two-stage RL+DDM (stage-1 only) model.
# - Simulates subjects with known parameters
# - Fits each subject via your existing Stan model
# - Writes recovery diagnostics (truth vs posterior means)

import sys, math, argparse, time, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import os
import glob
import shutil

# ---------------------------
# DDM simulator (Euler scheme)
# ---------------------------
def simulate_ddm_trial(a, v, t0, w=0.5, dt=0.001, tmax=3.0, rng=None):
    """
    Simulate a single DDM first-passage time using simple Euler-Maruyama.
    Boundaries: 0 (upper/left), a (lower/right). Start at w*a (w in [0,1]).
    Returns (rt, choice) with choice coding: 0=upper/left, 1=lower/right.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = w * a
    t = 0.0
    s = 1.0  # diffusion std
    sqdt = math.sqrt(dt)

    while t < tmax:
        # increment
        x += v * dt + s * sqdt * rng.normal()
        t += dt
        if x <= 0.0:
            return (t0 + t, 0)  # hit upper/left -> choice 0
        if x >= a:
            return (t0 + t, 1)  # hit lower/right -> choice 1

    # If no boundary hit by tmax, force to nearest boundary (rare with reasonable v)
    # This keeps data valid for the Stan model (no missed RTs).
    if abs(x - 0.0) < abs(x - a):
        return (t0 + tmax, 0)
    else:
        return (t0 + tmax, 1)

# ---------------------------
# Two-stage task simulator
# ---------------------------
def simulate_subject(
    subj_id,
    N,
    alpha,
    a,
    t0,
    scaler,
    S=2,
    p_common=0.7,
    beta2=5.0,   # inverse temperature for stage-2 softmax
    q0=0.5,
    reward_drift_std=0.025,  # standard deviation for reward probability random walk
    rng=None,
):
    """
    Simulate one subject on the multi-planet two-stage task with model-based stage-1.
    - Supports 2-5 planets (states) with non-stationary reward probabilities
    - Stage-1 options: choose between pairs of planets (all combinations)
    - Stage-2 visit follows common/rare transition probabilities
    - Stage-2 choice from softmax over Q(s2, a)
    - Reward probabilities drift over time via Gaussian random walk
    - Update Q(s2, a2) with alpha
    - Stage-1 RT & choice from DDM comparing model-based Q-values

    Returns a DataFrame with columns expected by your fitter.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Initialize reward probabilities with opposing preferences for each planet
    # Each planet starts with [0.25, 0.75] or [0.75, 0.25] pattern
    reward_probs = np.zeros((S, 2))
    for s in range(S):
        if s % 2 == 0:
            reward_probs[s] = [0.25, 0.75]  # even planets: action1 low, action2 high
        else:
            reward_probs[s] = [0.75, 0.25]  # odd planets: action1 high, action2 low

    # Generate all possible planet pairs (combinations) for S planets
    from itertools import combinations
    planet_pairs = list(combinations(range(1, S+1), 2))  # [(1,2), (1,3), (2,3), ...] for 1-based indexing
    n_pairs = len(planet_pairs)
    
    # Q(s,a), s=1..S, a in {1,2}
    Q = np.full((S, 2), q0, dtype=float)

    rows = []
    for trial in range(N):
        # Update reward probabilities with Gaussian random walk
        if trial > 0:  # Don't drift on first trial
            drift = rng.normal(0, reward_drift_std, size=(S, 2))
            reward_probs += drift
            # Clip to valid probability range [0.05, 0.95] to avoid extreme values
            reward_probs = np.clip(reward_probs, 0.05, 0.95)
        
        # For multi-planet task, randomly select a pair of planets for this trial
        pair_idx = rng.integers(0, n_pairs)
        mb1, mb2 = planet_pairs[pair_idx]  # 1-based planet indices
        
        # stage-1 value difference using model-based Q-values
        qL = max(Q[mb1-1, 0], Q[mb1-1, 1])  # max Q-value for left planet
        qR = max(Q[mb2-1, 0], Q[mb2-1, 1])  # max Q-value for right planet
        dv = qL - qR
        v = scaler * dv

        # stage-1 DDM
        rt1, ch1 = simulate_ddm_trial(a=a, v=v, t0=t0, w=0.5, dt=0.001, tmax=3.0, rng=rng)
        # ch1: 0=left(mb1), 1=right(mb2)

        # transition to stage-2 based on common/rare transition structure
        if ch1 == 0:
            # chose left planet (mb1) -> common transition to mb1, rare to mb2
            s2 = mb1 if rng.uniform() < p_common else mb2
        else:
            # chose right planet (mb2) -> common transition to mb2, rare to mb1
            s2 = mb2 if rng.uniform() < p_common else mb1

        # stage-2 policy: softmax over Q[s2,*]
        q_s = Q[s2-1]  # [Q(s2,1), Q(s2,2)]
        p_a1 = math.exp(beta2 * q_s[0]) / (math.exp(beta2 * q_s[0]) + math.exp(beta2 * q_s[1]))
        a2 = 0 if rng.uniform() < p_a1 else 1  # 0->action1, 1->action2
        rew = 1.0 if rng.uniform() < reward_probs[s2-1, a2] else 0.0

        # RL update on visited (s2, a2)
        Q[s2-1, a2] += alpha * (rew - Q[s2-1, a2])

        rows.append({
            "participant_id": subj_id,
            "trial_id": trial+1,
            # Format matching the reference CSV (hddm2_fixed_final_2states.csv)
            # state1/state2 are 0/1 based, fit_rlddm_single.py will handle the conversion
            "state1": pair_idx,     # 0-based pair index (for S=2: always 0, for S=3: 0,1,2)  
            "state2": s2 - 1,       # 0-based planet index (convert 1-based s2 to 0-based)
            # stage-1 observed:
            "rt": rt1,              # seconds
            "choice": ch1,          # 0/1 (0=left planet, 1=right planet)
            # stage-2 observed (for updates only):
            "choice2": a2,          # 0/1 (0->action1, 1->action2)
            "reward": rew,          # 0/1
        })

    return pd.DataFrame(rows)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stan", type=str, default="rlddm_single_subject.stan", help="Path to Stan model")
    ap.add_argument("--outdir", type=str, default="recovery_out", help="Output directory")
    ap.add_argument("--subject", type=int, required=True, help="Subject index to fit (1-based)")
    ap.add_argument("--total_subjects", type=int, default=20, help="Total number of synthetic subjects")
    ap.add_argument("--trials", type=int, default=300, help="Trials per subject")
    ap.add_argument("--planets", type=int, default=2, choices=[2, 3, 4, 5], help="Number of planets/states (2-5)")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--sim_only", action="store_true", help="Only generate simulated data, don't fit")
    # sampler settings (parameter recovery uses chains=1, default warmup=2000, draws=1000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--adapt_delta", type=float, default=0.999)
    ap.add_argument("--max_treedepth", type=int, default=15)
    ap.add_argument("--metric", type=str, default="dense_e", choices=["diag_e","dense_e"])
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Draw true parameters per subject (broad but sane) ----
    def trunc_norm(mu, sd, lo, hi, n):
        x = rng.normal(mu, sd, size=n)
        # simple rejection to stay in bounds
        mask = (x >= lo) & (x <= hi)
        while not mask.all():
            k = (~mask).sum()
            x[~mask] = rng.normal(mu, sd, size=k)
            mask = (x >= lo) & (x <= hi)
        return x

    # Check if we're in simulation mode or fitting mode
    truth_file = outdir / "truth_params.csv" 
    sim_csv_path = outdir / f"simulated_data_{args.planets}planets.csv"
    
    if args.sim_only or not (truth_file.exists() and sim_csv_path.exists()):
        # SIMULATION MODE: Generate all subjects
        print(f"\n=== SIMULATION MODE: Generating {args.total_subjects} subjects ===")
        
        # Generate diverse true parameters across subjects for good recovery assessment
        alphas = rng.beta(2.0, 2.0, size=args.total_subjects)                   # learning rates: broad 0..1
        a_vals = trunc_norm(1.28, 0.15, 0.6, 2.0, args.total_subjects)          # boundary separation: wider range
        t0_vals = trunc_norm(0.25, 0.08, 0.06, 0.45, args.total_subjects)       # non-decision time: wider range
        log_scalers = rng.normal(np.log(0.15), 0.5, size=args.total_subjects)   # drift scale: wider range
        scalers = np.exp(log_scalers)
        
        print(f"\nTrue parameter ranges:")
        print(f"  α (learning rate): {alphas.min():.3f} - {alphas.max():.3f}")
        print(f"  a (boundary): {a_vals.min():.3f} - {a_vals.max():.3f}")  
        print(f"  t0 (non-decision): {t0_vals.min():.3f} - {t0_vals.max():.3f}")
        print(f"  scaler (drift): {scalers.min():.3f} - {scalers.max():.3f}")

        truth = pd.DataFrame({
            "participant_id": np.arange(1, args.total_subjects+1, dtype=int),
            "alpha_true": alphas,
            "a_true": a_vals,
            "t0_true": t0_vals,
            "scaler_true": scalers,
            "log_scaler_true": log_scalers,
        })

        truth.to_csv(truth_file, index=False)

        # ---- 2) Simulate all subjects into one CSV ----
        dfs = []
        for i in range(args.total_subjects):
            df_i = simulate_subject(
                subj_id=i+1,
                N=args.trials,
                alpha=alphas[i],
                a=a_vals[i],
                t0=t0_vals[i],
                scaler=scalers[i],
                S=args.planets,
                p_common=0.7,
                beta2=5.0,
                q0=0.5,
                reward_drift_std=0.025,
                rng=rng,
            )
            dfs.append(df_i)

        sim = pd.concat(dfs, ignore_index=True)
        sim_csv_path = outdir / f"simulated_data_{args.planets}planets.csv"
        sim.to_csv(sim_csv_path, index=False)
        print(f"Simulated data saved to {sim_csv_path} ({len(sim)} rows)")
        
        if args.sim_only:
            print("Simulation complete. Use --subject <N> to fit individual subjects.")
            return

    else:
        # ---- Single subject fitting mode ----
        print(f"\n=== FITTING MODE: Subject {args.subject} ===")
        
        # Load the previously generated simulation data
        sim_csv_path = outdir / f"simulated_data_{args.planets}planets.csv"
        if not sim_csv_path.exists():
            print(f"❌ ERROR: Simulation data not found: {sim_csv_path}")
            print("This usually means the recovery_out directory was deleted or corrupted.")
            print("Solution: Re-run the full parameter recovery with simulation generation.")
            raise ValueError(f"Simulation data not found: {sim_csv_path}. Run with --sim_only first.")
            
        # Load truth parameters
        truth_file = outdir / "truth_params.csv"
        if not truth_file.exists():
            print(f"❌ ERROR: Truth parameters not found: {truth_file}")
            print("This usually means the recovery_out directory was deleted or corrupted.")
            print("Solution: Re-run the full parameter recovery with simulation generation.")
            raise ValueError(f"Truth parameters not found: {truth_file}. Run with --sim_only first.")
            
        truth = pd.read_csv(truth_file)
        subj_truth = truth[truth['participant_id'] == args.subject]
        if len(subj_truth) == 0:
            print(f"❌ ERROR: No truth parameters found for subject {args.subject}")
            print(f"Available subjects in truth file: {sorted(truth['participant_id'].unique())}")
            print("This might indicate a subject ID mismatch or corrupted truth file.")
            raise ValueError(f"No truth parameters found for subject {args.subject}")
            
        true_params = subj_truth.iloc[0]
        print(f"True parameters for subject {args.subject}:")
        print(f"  α = {true_params['alpha_true']:.3f}")
        print(f"  a = {true_params['a_true']:.3f}")
        print(f"  t0 = {true_params['t0_true']:.3f}")
        print(f"  scaler = {true_params['scaler_true']:.3f}")

        # ---- 3) Fit the single subject ----
        subj_outdir = outdir / f"subject_{args.subject}"
        subj_outdir.mkdir(exist_ok=True)
        
        # Build command for fit_rlddm_single.py (using chains=1 for parameter recovery)
        cmd = [
            "/opt/anaconda3/envs/stan/bin/python", "fit_rlddm_single.py",
            "--csv", str(sim_csv_path),
            "--subj", str(args.subject),
            "--states", str(args.planets),
            "--chains", "1",  # Use single chain for parameter recovery
            "--warmup", str(args.warmup),
            "--draws", str(args.draws),
            "--adapt_delta", str(args.adapt_delta),
            "--max_treedepth", str(args.max_treedepth),
            "--metric", args.metric,
            "--seed", str(args.seed + int(args.subject)),
            "--outdir", str(subj_outdir)
        ]
        
        # Environment for isolation with enhanced temporary directory handling
        env = os.environ.copy()
        env["CMDSTAN_PARALLEL"] = "false"
        env["STAN_NUM_THREADS"] = "1"
        
        # Create unique temporary directory for each subject with timestamp and PID
        import time
        import os
        timestamp = int(time.time() * 1000000)  # microsecond precision
        pid = os.getpid()
        temp_dir = outdir / f"tmp_subject_{args.subject}_{timestamp}_{pid}"
        env["TMPDIR"] = str(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Additional isolation environment variables
        env["STAN_OPENCL"] = "FALSE"
        env["STAN_NO_RANGE_CHECKS"] = "1"
        
        print(f"\nFitting subject {args.subject}...")
        print(f"Command: {' '.join(cmd)}")
        
        # Run fitting with enhanced error handling
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        # Check if the error is a cleanup error but Stan sampling succeeded
        if result.returncode != 0:
            error_output = result.stderr
            # If it's a FileNotFoundError during cleanup but sampling completed
            if ("FileNotFoundError" in error_output and 
                "No such file or directory" in error_output and
                ("Sampling completed" in result.stdout or 
                 "Elapsed Time:" in result.stdout)):
                print(f"⚠️  Subject {args.subject}: Stan sampling completed but cleanup failed")
                print("Attempting to recover results from Stan output files...")
                # Continue to result parsing - the Stan files should exist
            else:
                print(f"❌ Fitting failed for subject {args.subject}")
                print(f"Error: {error_output}")
                return
        
        # Parse results with fallback to cmdstan_tmp files
        try:
            import glob
            summary_files = glob.glob(str(subj_outdir / "*-summary.csv"))
            
            # If no summary files in subject directory, look in cmdstan_tmp
            if not summary_files:
                cmdstan_pattern = f"cmdstan_tmp/subject_{args.subject}_*/rlddm_single_subject-*-summary.csv"
                summary_files = glob.glob(cmdstan_pattern)
                if summary_files:
                    print(f"Found summary file in cmdstan_tmp for subject {args.subject}")
            
            if summary_files:
                summary_file = max(summary_files, key=lambda f: os.path.getmtime(f))
                summary_df = pd.read_csv(summary_file, index_col=0)
                
                est_alpha = float(summary_df.loc["alpha", "mean"])
                est_a = float(summary_df.loc["a", "mean"])
                est_t0 = float(summary_df.loc["t0", "mean"])
                est_scaler = float(summary_df.loc["scaler", "mean"])
                est_log_scaler = float(summary_df.loc["log_scaler", "mean"])
                
                print(f"\n✅ Fitting successful!")
                print(f"Estimated parameters for subject {args.subject}:")
                print(f"  α = {est_alpha:.3f} (true: {true_params['alpha_true']:.3f})")
                print(f"  a = {est_a:.3f} (true: {true_params['a_true']:.3f})")
                print(f"  t0 = {est_t0:.3f} (true: {true_params['t0_true']:.3f})")
                print(f"  scaler = {est_scaler:.3f} (true: {true_params['scaler_true']:.3f})")
                print(f"  log_scaler = {est_log_scaler:.3f} (true: {true_params['log_scaler_true']:.3f})")
                
                # Save individual result
                result_df = pd.DataFrame([{
                    'participant_id': args.subject,
                    'alpha_true': true_params['alpha_true'],
                    'alpha_hat': est_alpha,
                    'a_true': true_params['a_true'],  
                    'a_hat': est_a,
                    't0_true': true_params['t0_true'],
                    't0_hat': est_t0,
                    'scaler_true': true_params['scaler_true'],
                    'scaler_hat': est_scaler,
                    'log_scaler_true': true_params['log_scaler_true'],
                    'log_scaler_hat': est_log_scaler,
                }])
                result_df.to_csv(subj_outdir / "recovery_result.csv", index=False)
                print(f"Results saved to {subj_outdir / 'recovery_result.csv'}")
                
            else:
                print("❌ No summary file found in subject directory or cmdstan_tmp")
                
        except Exception as e:
            print(f"❌ Error parsing results: {e}")

        # Enhanced cleanup with retry logic
        temp_dirs = glob.glob(str(outdir / f"tmp_subject_{args.subject}*"))
        for d in temp_dirs:
            for attempt in range(3):  # Try up to 3 times
                try:
                    if os.path.exists(d):
                        # Wait a bit before cleanup to ensure all file handles are closed
                        time.sleep(0.5)
                        shutil.rmtree(d)
                        break
                except (OSError, FileNotFoundError, PermissionError):
                    if attempt == 2:  # Last attempt
                        print(f"Warning: Could not clean up {d}, leaving for manual cleanup")
                    else:
                        time.sleep(1)  # Wait longer before retry


if __name__ == "__main__":
    main()
