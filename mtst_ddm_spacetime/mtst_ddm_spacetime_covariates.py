from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import comb, log
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from mtst_covariates import load_mtst_csv, decode_menu_options, normalize_state2_to_1S


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


@dataclass
class PooledDDMStage1:
    df: pd.DataFrame
    subj_lookup: Dict[int, int]
    t0_upper_by_subj: np.ndarray


def _menu_pair_from_state1(state1: np.ndarray, S: int) -> tuple[np.ndarray, np.ndarray]:
    mb1, mb2 = decode_menu_options(state1, S=S)
    return mb1.astype(int), mb2.astype(int)


def _preprocess_single_subject(df_subj: pd.DataFrame, S: int, K: int) -> pd.DataFrame:
    d = df_subj.sort_values("trial").reset_index(drop=True).copy()
    d["trial"] = np.arange(1, len(d) + 1)

    mb1, mb2 = _menu_pair_from_state1(d["state1"].to_numpy(), S=S)
    d["mb1"] = mb1
    d["mb2"] = mb2
    d["s2"] = normalize_state2_to_1S(d["state2"].to_numpy(), S=S)
    d["log_rt1"] = np.log(d["rt1"].astype(float))

    # trackers for separated/component signals
    last_visit_state = {s: None for s in range(1, S + 1)}
    last_reward_state = {s: 0.0 for s in range(1, S + 1)}

    # trackers for menu memory
    last_menu_t: Dict[int, int] = {}
    last_menu_choice: Dict[int, int] = {}
    last_menu_reward: Dict[int, float] = {}

    # LRU cache of menu_ids (distinct menus)
    cache: OrderedDict[int, bool] = OrderedDict()

    # outputs
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
    prev_menu_id = np.full(len(d), -1, dtype=int)

    prev_menu = None

    # Precompute option sets for menu graph neighbor checks
    menu_options = {int(m): tuple(sorted((int(a), int(b)))) for m, a, b in zip(d["state1"], d["mb1"], d["mb2"])}

    for t in range(len(d)):
        menu_id = int(d.loc[t, "state1"])
        cur_pair = menu_options[menu_id]
        mb1_t = int(d.loc[t, "mb1"])
        mb2_t = int(d.loc[t, "mb2"])

        # component signals based on state values mapped to the two offered options
        diff_lastR_state[t] = float(last_reward_state[mb2_t] - last_reward_state[mb1_t])
        abs_diff_lastR_state[t] = abs(diff_lastR_state[t])

        lag1 = np.nan if last_visit_state[mb1_t] is None else (t - last_visit_state[mb1_t])
        lag2 = np.nan if last_visit_state[mb2_t] is None else (t - last_visit_state[mb2_t])
        diff_loglag_state[t] = np.log1p(0.0 if np.isnan(lag2) else lag2) - np.log1p(0.0 if np.isnan(lag1) else lag1)

        # menu memory / menu staleness
        if menu_id in last_menu_t:
            lag = t - last_menu_t[menu_id]
            log1p_menu_lag[t] = np.log1p(lag)
            prev_choice = int(last_menu_choice[menu_id])  # 0/1 on prior encounter of this exact menu
            prev_rew = float(last_menu_reward[menu_id])
            # orient menu pref toward current choice coding: +1 favors high option, -1 favors low option
            menu_pref[t] = 1.0 if prev_choice == 1 else -1.0
            menu_pref_win[t] = menu_pref[t] * (1.0 if prev_rew == 1 else -1.0)
            menu_pref_lag[t] = menu_pref[t] * log1p_menu_lag[t]

            # interference / diversity between repeats
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

        # distance from immediately previous menu
        if prev_menu is None:
            menu_dist_lag1[t] = 0.0
        else:
            if cur_pair == prev_menu:
                menu_dist_lag1[t] = 0.0
            else:
                overlap = len(set(cur_pair).intersection(prev_menu))
                menu_dist_lag1[t] = 1.0 if overlap == 1 else 2.0
        prev_menu = cur_pair
        prev_menu_id[t] = -1 if t == 0 else int(d.loc[t - 1, "state1"])

        # hierarchical cache hit logic BEFORE current menu is updated into cache
        if menu_id in cache:
            full_hit[t] = 1.0
            neighbor_hit[t] = 0.0
            cache.move_to_end(menu_id)
        else:
            # neighbor partial hit if any cached menu shares one option
            nh = 0.0
            for cached_m in cache.keys():
                pair_cached = menu_options.get(cached_m)
                if pair_cached is None:
                    continue
                overlap = len(set(cur_pair).intersection(pair_cached))
                if overlap == 1:
                    nh = 1.0
                    break
            neighbor_hit[t] = nh
            full_hit[t] = 0.0
            cache[menu_id] = True
            cache.move_to_end(menu_id)
            while len(cache) > K:
                cache.popitem(last=False)

        # update trackers after observing trial outcome
        chosen_firststage = mb2_t if int(d.loc[t, "choice1"]) == 1 else mb1_t
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
    d["prev_menu_id"] = prev_menu_id
    d["trial_scaled"] = (d["trial"].to_numpy() - 1) / max(1, (len(d) - 1))
    d["logM"] = log(comb(S, 2))
    d["coverage"] = K / comb(S, 2)
    d["full_hit_x_logM"] = d["full_hit"] * d["logM"]
    d["neighbor_hit_x_logM"] = d["neighbor_hit"] * d["logM"]
    d["diversity_x_logM"] = d["diversity_prop"] * d["logM"]
    d["menulag_x_logM"] = d["log1p_menu_lag"] * d["logM"]
    return d


def build_pooled_stage1_ddm_data(csv_by_S: Dict[int, str | Path], states: List[int], K: int = 1,
                                 t0_lower: float = 0.03, rt_min_margin: float = 0.02) -> PooledDDMStage1:
    dfs = []
    subj_lookup: Dict[int, int] = {}
    subj_counter = 1
    t0_upper_list: List[float] = []

    for S in states:
        path = Path(csv_by_S[S])
        df = load_mtst_csv(path)
        # keep first-stage relevant columns standardized
        rename = {}
        if "choice1" not in df.columns and "response1" in df.columns:
            rename["response1"] = "choice1"
        if "rt1" not in df.columns and "rt" in df.columns:
            rename["rt"] = "rt1"
        df = df.rename(columns=rename)

        for subj in sorted(df["participant_id"].unique().tolist()):
            dsub = df[df["participant_id"] == subj].copy()
            dsub = _preprocess_single_subject(dsub, S=S, K=K)
            rt_min = float(np.min(dsub["rt1"].astype(float).to_numpy()))
            t0_upper = max(t0_lower + 1e-4, rt_min - rt_min_margin)
            if t0_upper <= t0_lower:
                t0_upper = max(t0_lower + 1e-4, rt_min - 1e-4)

            dsub["S"] = int(S)
            dsub["subject_original"] = int(subj)
            dsub["subj_global"] = subj_counter
            subj_lookup[subj_counter] = int(subj)
            subj_counter += 1
            dfs.append(dsub)
            t0_upper_list.append(t0_upper)

    pooled = pd.concat(dfs, ignore_index=True)

    # global z-scoring of continuous predictors across pooled data
    z_cols = [
        "diff_lastR_state", "diff_loglag_state", "menu_pref_lag", "abs_diff_lastR_state",
        "menu_dist_lag1", "log1p_menu_lag", "diversity_prop", "logM", "coverage",
        "full_hit_x_logM", "neighbor_hit_x_logM", "diversity_x_logM", "menulag_x_logM",
        "trial_scaled",
    ]
    for c in z_cols:
        pooled[c + "_z"] = _z(pooled[c].to_numpy(dtype=float))

    return PooledDDMStage1(df=pooled, subj_lookup=subj_lookup, t0_upper_by_subj=np.asarray(t0_upper_list, dtype=float))


def get_variant_designs(df: pd.DataFrame, variant: str) -> tuple[list[str], list[str], list[str]]:
    """
    Returns (v_cols, t0_cols, a_cols) for a named space-time DDM variant.

    Key idea:
    - drift carries policy content / value content
    - t0 carries retrieval / recompute / memory-space burden
    - a carries generic caution/conflict only
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

    time_cache = [
        "full_hit",
        "neighbor_hit",
        "menu_dist_lag1_z",
        "log1p_menu_lag_z",
        "diversity_prop_z",
    ]
    space_terms = [
        "logM_z",
        "full_hit_x_logM_z",
        "neighbor_hit_x_logM_z",
        "diversity_x_logM_z",
        "menulag_x_logM_z",
    ]

    if variant == "baseline":
        return base_v, base_t0, base_a
    if variant == "t0_time":
        return base_v, base_t0 + time_cache, base_a
    if variant == "t0_space_time":
        return base_v, base_t0 + time_cache + space_terms, base_a
    if variant == "v_space_time":
        return base_v + time_cache + space_terms, base_t0, base_a
    if variant == "t0v_space_time":
        return base_v + time_cache, base_t0 + time_cache + space_terms, base_a
    if variant == "t0va_space_time":
        return base_v + time_cache, base_t0 + time_cache + space_terms, base_a + ["menu_dist_lag1_z"]
    raise ValueError(f"Unknown variant: {variant}")


def make_stan_data(pooled: PooledDDMStage1, variant: str, t0_lower: float = 0.03) -> dict:
    df = pooled.df.copy()
    v_cols, t0_cols, a_cols = get_variant_designs(df, variant)

    stan_data = {
        "N": int(len(df)),
        "J": int(df["subj_global"].nunique()),
        "subj_id": df["subj_global"].astype(int).to_numpy(),
        "rt": df["rt1"].astype(float).to_numpy(),
        "choice": df["choice1"].astype(int).to_numpy(),
        "K_v": len(v_cols),
        "X_v": df[v_cols].astype(float).to_numpy(),
        "K_t0": len(t0_cols),
        "X_t0": df[t0_cols].astype(float).to_numpy(),
        "K_a": len(a_cols),
        "X_a": df[a_cols].astype(float).to_numpy(),
        "t0_lower": float(t0_lower),
        "t0_upper": pooled.t0_upper_by_subj.astype(float),
    }
    return stan_data
