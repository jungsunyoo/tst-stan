
# mtst_covariates.py
# Utilities to load MTST CSVs and build menu-graph / caching covariates for Stan DDM models.

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros_like(x, dtype=float)
    z = (x - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return z


def load_mtst_csv(csv_path: Path) -> pd.DataFrame:
    """
    Loads an MTST CSV and standardizes column names to:
      participant_id, rt1, rt2, choice1, choice2, state1, state2, reward, trial
    Leaves extra columns intact.
    """
    df = pd.read_csv(csv_path)

    rename = {}
    # accept multiple variants
    if "subj_idx" in df.columns:
        rename["subj_idx"] = "participant_id"
    if "participant" in df.columns:
        rename["participant"] = "participant_id"

    if "rt1" in df.columns:
        rename["rt1"] = "rt1"
    if "rt" in df.columns and "rt1" not in df.columns:
        rename["rt"] = "rt1"
    if "rt2" in df.columns:
        rename["rt2"] = "rt2"

    if "response1" in df.columns:
        rename["response1"] = "choice1"
    if "choice" in df.columns and "choice1" not in df.columns:
        rename["choice"] = "choice1"
    if "response2" in df.columns:
        rename["response2"] = "choice2"

    if "feedback" in df.columns:
        rename["feedback"] = "reward"

    df = df.rename(columns=rename)

    required = ["participant_id", "rt1", "choice1", "choice2", "state1", "state2", "reward"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {csv_path} missing required columns: {missing}. "
            f"Columns present: {list(df.columns)}"
        )

    if "trial" not in df.columns:
        df["trial"] = np.nan

    df["participant_id"] = df["participant_id"].astype(int)
    df["rt1"] = df["rt1"].astype(float)
    if "rt2" in df.columns:
        df["rt2"] = df["rt2"].astype(float)

    # RT unit check: if median > 5, likely ms
    if np.nanmedian(df["rt1"].to_numpy()) > 5.0:
        df["rt1"] = df["rt1"] / 1000.0
        if "rt2" in df.columns:
            df["rt2"] = df["rt2"] / 1000.0

    def _fix_choice(col: str) -> None:
        v = df[col].dropna().unique()
        s = set(v.tolist())
        if s.issubset({0, 1}):
            df[col] = df[col].astype(int)
        elif s == {-1, 1}:
            df[col] = ((df[col] + 1) // 2).astype(int)
        elif s == {1, 2}:
            df[col] = (df[col] - 1).astype(int)
        else:
            raise ValueError(f"{col} must be 0/1-coded; found unique={sorted(s)}")

    _fix_choice("choice1")
    _fix_choice("choice2")

    df["state1"] = df["state1"].astype(int)
    df["state2"] = df["state2"].astype(int)
    df["reward"] = np.clip(df["reward"].astype(float), 0.0, 1.0)

    # basic RT filter
    df = df[df["rt1"] >= 0.15].copy()
    if (df["rt1"] <= 0).any():
        raise ValueError("Non-positive RT1 values detected after filtering.")

    return df


def decode_menu_options(state1: np.ndarray, S: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode state1 index to (mb1, mb2) option IDs in 1..S using lexicographic combinations.
    Accepts 0..C-1 or 1..C indexing.
    """
    S = int(S)
    pairs = np.array(list(combinations(np.arange(1, S + 1), 2)))
    C = pairs.shape[0]
    idx = np.asarray(state1, dtype=int).copy()

    if idx.min() >= 1 and idx.max() <= C:
        idx = idx - 1

    if idx.min() < 0 or idx.max() >= C:
        raise ValueError(
            f"state1 out of bounds for S={S}. Valid 0..{C-1} or 1..{C}. "
            f"Got range {idx.min()}..{idx.max()}"
        )

    chosen = pairs[idx]
    return chosen[:, 0].astype(int), chosen[:, 1].astype(int)


def normalize_state2_to_1S(state2: np.ndarray, S: int) -> np.ndarray:
    state2 = np.asarray(state2, dtype=int)
    if state2.min() == 0 and state2.max() == (S - 1):
        return state2 + 1
    if state2.min() == 1 and state2.max() == S:
        return state2
    raise ValueError(
        f"state2 must be 0..{S-1} or 1..{S}; got {sorted(np.unique(state2))}"
    )


@dataclass
class SubjectDDMData:
    subj: int
    S: int
    N: int
    rt: np.ndarray
    choice: np.ndarray
    mb1: np.ndarray
    mb2: np.ndarray
    s2: np.ndarray
    reward: np.ndarray
    trial: np.ndarray
    rt_signed: np.ndarray
    trial_scaled: np.ndarray

    x_drift: np.ndarray
    x_tau: np.ndarray
    x_a: np.ndarray
    x_trans: np.ndarray

    drift_names: List[str]
    tau_names: List[str]
    a_names: List[str]
    trans_names: List[str]

    t0_lower: float
    t0_upper: float


def build_subject_covariates(
    df_all: pd.DataFrame,
    subj: int,
    S: int,
    t0_lower: float = 0.03,
    rt_min_margin: float = 0.02,
) -> SubjectDDMData:
    """
    Build per-subject predictors for DDM regression / IOHMM-DDM.

    Drift predictors (direction toward mb2):
      - diff_lastR_state
      - diff_loglag_state
      - menu_pref
      - menu_pref_win
      - menu_pref_lag

    Tau predictors (retrieval/caching):
      - menu_dist_lag1
      - log1p_menu_lag
      - trial_scaled

    Boundary predictors (optional):
      - abs_diff_lastR_state
      - menu_dist_lag1

    Transition predictors (IOHMM):
      - menu_dist_lag1
      - log1p_menu_lag
      - log1p_lag_pred_option
      - prev_reward
    """
    sdf = df_all[df_all["participant_id"].astype(int) == int(subj)].copy()
    if sdf.empty:
        raise ValueError(f"No rows for participant_id={subj}")

    # enforce trial order; create trial index
    sdf = sdf.sort_values(["trial"]).copy() if sdf["trial"].notna().any() else sdf.copy()
    sdf = sdf.reset_index(drop=True)
    sdf["trial"] = np.arange(1, len(sdf) + 1)

    # decode menus
    mb1, mb2 = decode_menu_options(sdf["state1"].to_numpy(), S=S)
    sdf["mb1"] = mb1
    sdf["mb2"] = mb2
    sdf["s2"] = normalize_state2_to_1S(sdf["state2"].to_numpy(), S=S)

    rt = sdf["rt1"].astype(float).to_numpy()
    choice = sdf["choice1"].astype(int).to_numpy()  # 0 -> mb1, 1 -> mb2
    reward = sdf["reward"].astype(float).to_numpy()

    # signed RT for wiener: + for choice==1 (mb2), - for choice==0 (mb1)
    rt_signed = rt.copy()
    rt_signed[choice == 0] *= -1.0

    # per-subject t0 upper bound
    rt_min = float(np.min(rt))
    t0_upper = max(t0_lower + 1e-4, rt_min - rt_min_margin)
    if t0_upper <= t0_lower:
        t0_upper = max(t0_lower + 1e-4, rt_min - 1e-4)

    trial = sdf["trial"].to_numpy()
    trial_scaled = (trial - 1) / max(1, (len(trial) - 1))

    # --- trackers ---
    last_menu_t: Dict[Tuple[int, int], int] = {}
    last_menu_choice: Dict[Tuple[int, int], int] = {}
    last_menu_reward: Dict[Tuple[int, int], float] = {}

    last_visit_t: Dict[int, int] = {s: -10**9 for s in range(1, S + 1)}
    last_reward_state: Dict[int, float] = {s: 0.0 for s in range(1, S + 1)}
    visited_state: Dict[int, bool] = {s: False for s in range(1, S + 1)}

    menu_dist_lag1 = np.zeros(len(sdf), dtype=float)
    menu_lag = np.zeros(len(sdf), dtype=float)
    menu_pref = np.zeros(len(sdf), dtype=float)
    menu_prev_rew = np.zeros(len(sdf), dtype=float)

    lastR_mb1 = np.zeros(len(sdf), dtype=float)
    lastR_mb2 = np.zeros(len(sdf), dtype=float)
    lag_visit_mb1 = np.full(len(sdf), np.nan, dtype=float)
    lag_visit_mb2 = np.full(len(sdf), np.nan, dtype=float)

    lag_pred_opt = np.full(len(sdf), np.nan, dtype=float)
    prev_reward = np.zeros(len(sdf), dtype=float)

    prev_menu: Optional[Tuple[int, int]] = None

    for t_idx in range(len(sdf)):
        i = int(sdf.loc[t_idx, "mb1"])
        j = int(sdf.loc[t_idx, "mb2"])
        key = (min(i, j), max(i, j))

        # menu distance from previous menu
        if prev_menu is None:
            menu_dist_lag1[t_idx] = 0.0
        else:
            overlap = len(set(key).intersection(prev_menu))
            menu_dist_lag1[t_idx] = float(2 - overlap)

        # menu lag and menu memory
        if key in last_menu_t:
            menu_lag[t_idx] = float(t_idx - last_menu_t[key])
            c_prev = int(last_menu_choice[key])  # 0/1 for (mb1/mb2) on the last encounter
            menu_pref[t_idx] = float(2 * c_prev - 1)  # +1 if mb2, -1 if mb1
            menu_prev_rew[t_idx] = float(last_menu_reward[key])
        else:
            menu_lag[t_idx] = 0.0
            menu_pref[t_idx] = 0.0
            menu_prev_rew[t_idx] = 0.0

        # component last reward at each state
        lastR_mb1[t_idx] = float(last_reward_state[i]) if visited_state[i] else 0.0
        lastR_mb2[t_idx] = float(last_reward_state[j]) if visited_state[j] else 0.0

        # visit lags (based on visiting state2)
        if visited_state[i]:
            lag_visit_mb1[t_idx] = float(t_idx - last_visit_t[i])
        if visited_state[j]:
            lag_visit_mb2[t_idx] = float(t_idx - last_visit_t[j])

        # component-predicted best option
        # heuristic: higher last reward; tie -> more recent visit; if never visited -> other
        pred = j
        if visited_state[i] and not visited_state[j]:
            pred = i
        elif visited_state[j] and not visited_state[i]:
            pred = j
        elif visited_state[i] and visited_state[j]:
            if lastR_mb1[t_idx] > lastR_mb2[t_idx]:
                pred = i
            elif lastR_mb2[t_idx] > lastR_mb1[t_idx]:
                pred = j
            else:
                li = lag_visit_mb1[t_idx] if np.isfinite(lag_visit_mb1[t_idx]) else 1e9
                lj = lag_visit_mb2[t_idx] if np.isfinite(lag_visit_mb2[t_idx]) else 1e9
                pred = i if li < lj else j

        if visited_state[pred]:
            lag_pred_opt[t_idx] = float(t_idx - last_visit_t[pred])
        else:
            lag_pred_opt[t_idx] = float(t_idx + 1)

        prev_reward[t_idx] = reward[t_idx - 1] if t_idx > 0 else 0.0

        # update menu trackers at end of trial
        last_menu_t[key] = t_idx
        last_menu_choice[key] = int(choice[t_idx])
        last_menu_reward[key] = float(reward[t_idx])

        # update visited planet trackers
        s2 = int(sdf.loc[t_idx, "s2"])
        visited_state[s2] = True
        last_visit_t[s2] = t_idx
        last_reward_state[s2] = float(reward[t_idx])

        prev_menu = key

    diff_lastR_state = lastR_mb2 - lastR_mb1
    diff_loglag_state = (
        np.log1p(np.nan_to_num(lag_visit_mb2, nan=0.0))
        - np.log1p(np.nan_to_num(lag_visit_mb1, nan=0.0))
    )

    log1p_menu_lag = np.log1p(menu_lag)
    menu_pref_win = menu_pref * menu_prev_rew
    menu_pref_lag = menu_pref * log1p_menu_lag

    drift_names = [
        "diff_lastR_state",
        "diff_loglag_state",
        "menu_pref",
        "menu_pref_win",
        "menu_pref_lag",
    ]
    Xv = np.column_stack(
        [
            _zscore(diff_lastR_state),
            _zscore(diff_loglag_state),
            _zscore(menu_pref),
            _zscore(menu_pref_win),
            _zscore(menu_pref_lag),
        ]
    )

    tau_names = ["menu_dist_lag1", "log1p_menu_lag", "trial_scaled"]
    Xt = np.column_stack(
        [
            _zscore(menu_dist_lag1),
            _zscore(log1p_menu_lag),
            _zscore(trial_scaled),
        ]
    )

    a_names = ["abs_diff_lastR_state", "menu_dist_lag1"]
    Xa = np.column_stack([_zscore(np.abs(diff_lastR_state)), _zscore(menu_dist_lag1)])

    trans_names = [
        "menu_dist_lag1",
        "log1p_menu_lag",
        "log1p_lag_pred_option",
        "prev_reward",
    ]
    Xx = np.column_stack(
        [
            _zscore(menu_dist_lag1),
            _zscore(log1p_menu_lag),
            _zscore(np.log1p(lag_pred_opt)),
            _zscore(prev_reward),
        ]
    )

    return SubjectDDMData(
        subj=int(subj),
        S=int(S),
        N=int(len(sdf)),
        rt=rt,
        choice=choice,
        mb1=mb1,
        mb2=mb2,
        s2=sdf["s2"].astype(int).to_numpy(),
        reward=reward,
        trial=trial,
        rt_signed=rt_signed,
        trial_scaled=trial_scaled.astype(float),
        x_drift=Xv.astype(float),
        x_tau=Xt.astype(float),
        x_a=Xa.astype(float),
        x_trans=Xx.astype(float),
        drift_names=drift_names,
        tau_names=tau_names,
        a_names=a_names,
        trans_names=trans_names,
        t0_lower=float(t0_lower),
        t0_upper=float(t0_upper),
    )
