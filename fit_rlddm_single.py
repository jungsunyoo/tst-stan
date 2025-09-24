#!/usr/bin/env python
# file: fit_rlddm_single.py
# End-to-end: read your CSV, prepare one subject, compile Stan, fit, save results.

import os, argparse, numpy as np, pandas as pd
import arviz as az
from cmdstanpy import CmdStanModel

def load_subject(csvfile: str, subj: int) -> pd.DataFrame:
  df = pd.read_csv(csvfile, index_col=0)
  df = df.rename(columns={
      "subj_idx":"participant_id",
      "trial":"trial_id",
      "rt1":"rt",
      "response1":"choice",
      "response2":"choice2"
  })
  # keep only one subject
  sdf = df[df["participant_id"].astype(int) == int(subj)].copy()
  if sdf.empty:
    raise SystemExit(f"No rows for participant_id={subj} in {csvfile}")

  # dtypes
  sdf["rt"] = sdf["rt"].astype(float)
  for c in ["choice","choice2","state1","state2"]:
    sdf[c] = sdf[c].astype(int)
  # choice must be 0/1
  if sdf["choice"].min() < 0 or sdf["choice"].max() > 1:
    raise SystemExit("choice must be coded 0/1 (lower/upper).")
  if sdf["choice2"].min() < 0 or sdf["choice2"].max() > 1:
    raise SystemExit("choice2 must be coded 0/1 (lower/upper).")

  # reward in [0,1]
#   if "feedback" in sdf.columns:
  r = sdf["feedback"].astype(float) #.clip(0.0, 1.0).to_numpy()
#   else:
#     # fallback: treat upper choice as reward=1 (you can replace with your true feedback)
#     r = (sdf["choice"].to_numpy() == 1).astype(float)

  # Generate all possible combinations (0-based)
  from itertools import combinations
  S = int(max(sdf["state1"].max(), sdf["state2"].max())) + 1
  all_combos = list(combinations(range(S), 2))
  
  # Map combination indices to actual state pairs
  actual_state1 = []
  actual_state2 = []
  
  for combo_idx in sdf["state1"].to_numpy().astype(int):
    s1, s2 = all_combos[combo_idx]  # Get the state pair for this combination
    actual_state1.append(s1 + 1)    # Convert to 1-based for Stan
    actual_state2.append(s2 + 1)    # Convert to 1-based for Stan
  
  data = dict(
    N = len(sdf),
    rt = sdf["rt"].to_numpy(),
    choice = sdf["choice"].to_numpy().astype(int),
    choice2 = sdf["choice2"].to_numpy().astype(int),
    state1 = np.array(actual_state1),
    state2 = np.array(actual_state2),
    S = S,
    reward = r,
  )
  return sdf, data

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--stan", default="rlddm_single_subject.stan", help="Stan model file")
  ap.add_argument("--csv", required=True, help="CSV path (e.g., hddm2_fixed_final_2states.csv)")
  ap.add_argument("--subj", type=int, required=True, help="participant_id")
  ap.add_argument("--outdir", default="stan_out")
  ap.add_argument("--chains", type=int, default=4)
  ap.add_argument("--warmup", type=int, default=1500)
  ap.add_argument("--draws", type=int, default=1000)
  ap.add_argument("--adapt_delta", type=float, default=0.95)
  ap.add_argument("--max_treedepth", type=int, default=12)
  ap.add_argument("--seed", type=int, default=2025)
  args = ap.parse_args()

  os.makedirs(args.outdir, exist_ok=True)
  sdf, data = load_subject(args.csv, args.subj)

  # compile (reuses the exe if unchanged)
  model = CmdStanModel(stan_file=args.stan)
  inits = [
    {"alpha":0.2, "scaler":1.0, "a":1.0, "z":0.5, "t0":0.25, "theta":1.0}
    for _ in range(args.chains)
    ]
  fit = model.sample(
    data=data,
    seed=args.seed + args.subj,
    chains=args.chains,
    parallel_chains=args.chains,
    iter_warmup=args.warmup,
    iter_sampling=args.draws,
    adapt_delta=args.adapt_delta,
    inits=inits,
    max_treedepth=args.max_treedepth,
    show_progress=True
  )


#   opt = model.optimize(data=stan_data, algorithm="lbfgs")
# inits = [opt.optimized_params_dict] * chains
# fit = model.sample(..., inits=inits, adapt_delta=0.95, max_treedepth=12)
  print("step sizes per chain:", fit.step_size)



  print(fit.diagnose())  # prints divergences, treedepth saturations, step size, E-BFMI, etc.
  # Save CmdStan outputs and an ArviZ file for convenience
  base = os.path.join(args.outdir, f"stan_ssc_{os.path.basename(args.csv).split('_')[-1].replace('.csv','')}_s{args.subj:04d}")
#   fit.save_csvfiles(dir=args.outdir, basename=os.path.basename(base))
  fit.save_csvfiles(dir=args.outdir)  # new API
  # If you want to see the actual files:
  for f in fit.runset.csv_files:
      print("Saved:", f)

  try:
    idata = az.from_cmdstanpy(posterior=fit)
    az.to_netcdf(idata, base + ".nc")
    # summ = az.summary(idata, var_names=["alpha","scaler","a","z","t0","theta"], hdi_prob=0.95)
    summ = az.summary(idata, var_names=["alpha","scaler","a","t0",], hdi_prob=0.95)
    summ.to_csv(base + "_summary.csv")
  except Exception as e:
    print("ArviZ export failed (ok to ignore):", e)

  print("Saved outputs under:", args.outdir)

if __name__ == "__main__":
  main()
