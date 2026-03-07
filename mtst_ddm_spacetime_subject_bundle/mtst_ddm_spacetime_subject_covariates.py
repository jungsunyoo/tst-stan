from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 0:
        out = np.zeros_like(x, dtype=float)
    else:
        out = (x - mu) / sd
    out[~np.isfinite(out)] = 0.0
    return out


def load_mtst_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    rename = {}
    if "subj_idx" in df.columns:
        rename["subj_idx"] = "participant_id"
    if "participant" in df.columns:
        rename["participant"] = "participant_id"

    if "response1" in df.columns:
        rename["response1"] = "choice1"
    if "response2" in df.columns:
        rename["response2"] = "choice2"
    if "feedback" in df.columns:
        rename["feedback"] = "reward"
    if "rt" in df.columns and "rt1" not in df.columns:
        rename["rt"] = "rt1"
    df = df.rename(columns=rename)

    required = ["participant_id", "rt1", "choice1", "choice2", "state1", "state2", "reward"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {csv_path} missing required columns: {missing}. Present: {list(df.columns)}")

    if "trial" not in df.columns:
        df["trial"] = np.nan

    df["participant_id"] = df["participant_id"].astype(int)
    df["rt1"] = df["rt1"].astype(float)

    # ms -> s heuristic
    if np.nanmedian(df["rt1"].to_numpy()) > 5.0:
        df["rt1"] = df["rt1"] / 1000.0

    def _fix_choice(col: str) -> None:
        vals = df[col].dropna().unique().tolist()
        s = set(vals)
        if s.issubset({0, 1}):
            df[col] = df[col].astype(int)
        elif s == {-1, 1}:
            df[col] = ((df[col] + 1) // 2).astype(int)
        elif s == {1, 2}:
            df[col] = (df[col] - 1).astype(int)
        else:
            raise ValueError(f"{col} must be 0/1-coded; found {sorted(s)}")

    _fix_choice("choice1")
    _fix_choice("choice2")

    df["state1"] = df["state1"].astype(int)
    df["state2"] = df["state2"].astype(int)
    df["reward"] = np.clip(df["reward"].astype(float), 0.0, 1.0)

    df = df[df["rt1"] >= 0.15].copy()
    if (df["rt1"] <= 0).any():
        raise ValueError("Non-positive RT1 values detected after filtering.")

    return df


def decode_menu_options(state1: np.ndarray, S: int) -> Tuple[np.ndarray, np.ndarray]:
    pairs = np.array(list(combinations(np.arange(1, S + 1), 2)))
    C = pairs.shape[0]
    idx = np.asarray(state1, dtype=int).copy()
    if idx.min() >= 1 and idx.max() <= C:
        idx = idx - 1
    if idx.min() < 0 or idx.max() >= C:
        raise ValueError(f"state1 out of bounds for S={S}; got {idx.min()}..{idx.max()} valid 0..{C-1} or 1..{C}")
    chosen = pairs[idx]
    return chosen[:, 0].astype(int), chosen[:, 1].astype(int)


def normalize_state2_to_1S(state2: np.ndarray, S: int) -> np.ndarray:
    state2 = np.asarray(state2, dtype=int)
    if state2.min() == 0 and state2.max() == (S - 1):
        return state2 + 1
    if state2.min() == 1 and state2.max() == S:
        return state2
    raise ValueError(f"state2 must be 0..{S-1} or 1..{S}; got {sorted(np.unique(state2))}")


@dataclass
class SubjectSpaceTimeDDM:
    subj: int
    S: int
    K: int
    N: int
    rt: np.ndarray
    choice: np.ndarray
    x_drift: np.ndarray
    x_t0: np.ndarray
    x_a: np.ndarray
    drift_names: List[str]
    t0_names: List[str]
    a_names: List[str]
    t0_lower: float
    t0_upper: float
    meta: dict


def _preprocess_single_subject(df_subj: pd.DataFrame, S: int, K: int) -> pd.DataFrame:
    d = df_subj.sort_values("trial").reset_index(drop=True).copy()
    d["trial"] = np.arange(1, len(d) + 1)

    mb1, mb2 = decode_menu_options(d["state1"].to_numpy(), S=S)
    d["mb1"] = mb1
    d["mb2"] = mb2
    d["s2"] = normalize_state2_to_1S(d["state2"].to_numpy(), S=S)

    last_visit_state = {s: None for s in range(1, S + 1)}
    last_reward_state = {s: 0.0 for s in range(1, S + 1)}

    last_menu_t = {}
    last_menu_choice = {}
    last_menu_reward = {}

    cache = OrderedDict()

    diff_lastR_state = np.zeros(len(d), dtype=float)
    diff_loglag_state = np.zeros(len(d), dtype=float)
    abs_diff_lastR_state = np.zeros(len(d), dtype=float)
    menu_pref = np.zeros(len(d), dtype=float)
    menu_pref_win = np.zeros(len(d), dtype=float)
    menu_pref_lag = np.zeros(len(d), dtype=float)
    menu_dist_lag1 = np.zeros(len(d), dtype=float)
    log1p_menu_lag = np.zeros(len(d), dtype=float)
    diversity_prop = np.zeros(len(d), dtype=float)
    full_hit = np.zeros(len(d), dtype=float)
    neighbor_hit = np.zeros(len(d), dtype=float)

    prev_menu = None
    menu_options = {int(m): tuple(sorted((int(a), int(b)))) for m, a, b in zip(d["state1"], d["mb1"], d["mb2"])}

    for t in range(len(d)):
        menu_id = int(d.loc[t, "state1"])
        cur_pair = menu_options[menu_id]
        mb1_t = int(d.loc[t, "mb1"])
        mb2_t = int(d.loc[t, "mb2"])

        diff_lastR_state[t] = float(last_reward_state[mb2_t] - last_reward_state[mb1_t])
        abs_diff_lastR_state[t] = abs(diff_lastR_state[t])

        lag1 = np.nan if last_visit_state[mb1_t] is None else (t - last_visit_state[mb1_t])
        lag2 = np.nan if last_visit_state[mb2_t] is None else (t - last_visit_state[mb2_t])
        diff_loglag_state[t] = np.log1p(0.0 if np.isnan(lag2) else lag2) - np.log1p(0.0 if np.isnan(lag1) else lag1)

        if menu_id in last_menu_t:
            lag = t - last_menu_t[menu_id]
            log1p_menu_lag[t] = np.log1p(lag)
            prev_choice = int(last_menu_choice[menu_id])
            prev_rew = float(last_menu_reward[menu_id])
            menu_pref[t] = 1.0 if prev_choice == 1 else -1.0
            menu_pref_win[t] = menu_pref[t] * (1.0 if prev_rew == 1.0 else -1.0)
            menu_pref_lag[t] = menu_pref[t] * log1p_menu_lag[t]

            j = last_menu_t[menu_id]
            if t - j > 1:
                intervening = d.loc[j + 1 : t - 1, "state1"].astype(int).tolist()
                n_unique = len(set(intervening))
                diversity_prop[t] = n_unique / max(1, (t - j - 1))
            else:
                diversity_prop[t] = 0.0
        else:
            log1p_menu_lag[t] = 0.0
            diversity_prop[t] = 0.0
            menu_pref[t] = 0.0
            menu_pref_win[t] = 0.0
            menu_pref_lag[t] = 0.0

        if prev_menu is None:
            menu_dist_lag1[t] = 0.0
        else:
            if cur_pair == prev_menu:
                menu_dist_lag1[t] = 0.0
            else:
                overlap = len(set(cur_pair).intersection(prev_menu))
                menu_dist_lag1[t] = 1.0 if overlap == 1 else 2.0
        prev_menu = cur_pair

        if menu_id in cache:
            full_hit[t] = 1.0
            neighbor_hit[t] = 0.0
            cache.move_to_end(menu_id)
        else:
            nh = 0.0
            for cached_m in cache.keys():
                overlap = len(set(cur_pair).intersection(menu_options[cached_m]))
                if overlap == 1:
                    nh = 1.0
                    break
            neighbor_hit[t] = nh
            full_hit[t] = 0.0
            cache[menu_id] = True
            cache.move_to_end(menu_id)
            while len(cache) > K:
                cache.popitem(last=False)

        s2_t = int(d.loc[t, "s2"])
        last_visit_state[s2_t] = t
        last_reward_state[s2_t] = float(d.loc[t, "reward"])
        last_menu_t[menu_id] = t
        last_menu_choice[menu_id] = int(d.loc[t, "choice1"])
        last_menu_reward[menu_id] = float(d.loc[t, "reward"])

    d["diff_lastR_state"] = diff_lastR_state
    d["diff_loglag_state"] = diff_loglag_state
    d["abs_diff_lastR_state"] = abs_diff_lastR_state
    d["menu_pref"] = menu_pref
    d["menu_pref_win"] = menu_pref_win
    d["menu_pref_lag"] = menu_pref_lag
    d["menu_dist_lag1"] = menu_dist_lag1
    d["log1p_menu_lag"] = log1p_menu_lag
    d["diversity_prop"] = diversity_prop
    d["full_hit"] = full_hit
    d["neighbor_hit"] = neighbor_hit
    d["trial_scaled"] = (d["trial"].to_numpy() - 1) / max(1, (len(d) - 1))
    d["log_rt1"] = np.log(d["rt1"].astype(float))
    return d


def get_variant_designs_subject(variant: str) -> tuple[List[str], List[str], List[str]]:
    """
    Subject-level variants for within-SSC fits.

    Important: within a single subject/SSC fit, menu-space size M is constant, so
    logM and logM interactions are not separately identifiable. To test the
    space-time hypothesis without a hierarchical model, fit the same variant
    separately by SSC (e.g., S=3,4,5) and compare the recovered coefficients
    across SSC.
    """
    base_v = [
        "diff_lastR_state_z",
        "diff_loglag_state_z",
        "menu_pref",
        "menu_pref_win",
        "menu_pref_lag_z",
    ]
    base_t0 = ["trial_scaled_z"]
    base_a = ["abs_diff_lastR_state_z"]

    cache_time = [
        "full_hit",
        "neighbor_hit",
        "menu_dist_lag1_z",
        "log1p_menu_lag_z",
        "diversity_prop_z",
    ]

    if variant == "baseline":
        return base_v, base_t0, base_a
    if variant == "t0_cache":
        return base_v, base_t0 + cache_time, base_a
    if variant == "v_cache":
        return base_v + cache_time, base_t0, base_a
    if variant == "t0v_cache":
        return base_v + cache_time, base_t0 + cache_time, base_a
    if variant == "t0va_cache":
        return base_v + cache_time, base_t0 + cache_time, base_a + ["menu_dist_lag1_z"]
    raise ValueError(f"Unknown variant: {variant}")


def build_subject_spacetime_covariates(
    df_all: pd.DataFrame,
    subj: int,
    S: int,
    K: int = 1,
    variant: str = "t0_cache",
    t0_lower: float = 0.03,
    rt_min_margin: float = 0.02,
) -> SubjectSpaceTimeDDM:
    dsub = df_all[df_all["participant_id"].astype(int) == int(subj)].copy()
    if dsub.empty:
        raise ValueError(f"No rows for participant_id={subj}")

    dsub = _preprocess_single_subject(dsub, S=S, K=K)

    z_cols = [
        "diff_lastR_state",
        "diff_loglag_state",
        "menu_pref_lag",
        "abs_diff_lastR_state",
        "menu_dist_lag1",
        "log1p_menu_lag",
        "diversity_prop",
        "trial_scaled",
    ]
    for c in z_cols:
        dsub[c + "_z"] = _z(dsub[c].to_numpy(dtype=float))

    v_cols, t0_cols, a_cols = get_variant_designs_subject(variant)

    rt = dsub["rt1"].astype(float).to_numpy()
    rt_min = float(np.min(rt))
    t0_upper = max(t0_lower + 1e-4, rt_min - rt_min_margin)
    if t0_upper <= t0_lower:
        t0_upper = max(t0_lower + 1e-4, rt_min - 1e-4)

    meta = {
        "variant": variant,
        "K": int(K),
        "menu_space_M": int(len(set(dsub["state1"].astype(int).tolist()))),
        "theoretical_M": int(len(list(combinations(range(1, S + 1), 2)))),
        "cache_coverage_K_over_M": float(K / len(list(combinations(range(1, S + 1), 2)))),
        "mean_full_hit": float(dsub["full_hit"].mean()),
        "mean_neighbor_hit": float(dsub["neighbor_hit"].mean()),
    }

    return SubjectSpaceTimeDDM(
        subj=int(subj),
        S=int(S),
        K=int(K),
        N=int(len(dsub)),
        rt=rt,
        choice=dsub["choice1"].astype(int).to_numpy(),
        x_drift=dsub[v_cols].astype(float).to_numpy(),
        x_t0=dsub[t0_cols].astype(float).to_numpy(),
        x_a=dsub[a_cols].astype(float).to_numpy(),
        drift_names=v_cols,
        t0_names=t0_cols,
        a_names=a_cols,
        t0_lower=float(t0_lower),
        t0_upper=float(t0_upper),
        meta=meta,
    )
