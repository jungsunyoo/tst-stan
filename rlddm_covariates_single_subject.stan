functions {
  // Map a linear predictor into (low, high) using logistic link
  real logit_to_range(real x, real low, real high) {
    return low + inv_logit(x) * (high - low);
  }
}

data {
  int<lower=1> N;                    // trials
  vector<lower=0>[N] rt;             // Stage-1 RT (s)

  // NEW-STYLE array declarations
  array[N] int<lower=0, upper=1> choice;   // 1 = upper boundary, 0 = lower (drift sign flip)

  // First-stage menu & second-stage outcomes
  int<lower=1> S;                            // number of planets
  array[N] int<lower=1, upper=S> mb1;        // left option ship (1..S)
  array[N] int<lower=1, upper=S> mb2;        // right option ship (1..S)
  array[N] int<lower=1, upper=S> s2;         // visited planet (1..S)
  array[N] int<lower=0, upper=1> choice2;    // (kept for compatibility; unused here)
  vector<lower=0, upper=1>[N] reward;        // second-stage reward (0/1)

  // Bounds for non-decision time
  real<lower=0> rt_upper_t0;                 // strict upper bound for t0 (below min RT)
  real<lower=0> t0_lower;                    // hard lower bound (e.g., 0.03)

  // Trial-wise covariates for t0 & boundary decline
  vector[N] U_chosen_z;        // z-scored planning uncertainty
  vector[N] d1;                // 1 if Johnson distance == 1
  vector[N] d2;                // 1 if Johnson distance == 2
  vector[N] log1p_ship_lag_z;  // z-scored recency (ship overlap)
  vector[N] log1p_pair_lag_z;  // z-scored recency (exact pair)
  vector[N] trial_scaled;      // [0,1] for boundary decline

  // Fixed transition probability for EV computation
  real<lower=0, upper=1> p_common;           // usually 0.7
}

parameters {
  // RL parameters
  real<lower=0, upper=1> alpha;              // RW learning rate

  // DDM core
  real<lower=0.2, upper=3.5> a;              // baseline boundary separation
  real<lower=t0_lower, upper=rt_upper_t0> t0;// baseline non-decision time
  real<lower=0, upper=1> w;                  // starting point (0.5 ~ unbiased)
  real<lower=0> scaler;                      // drift scaling (Qdiff → drift)

  // NEW: between-trials boundary decline
  real<lower=0> k_decline;

  // NEW: coefficients for t0 regression (on logit scale)
  real b_U;
  real b_d1;
  real b_d2;
  real b_ship;
  real b_pair;
}

transformed parameters {
  // Online planet values (MB) updated via RW
  vector[S] V;
  for (s in 1:S) V[s] = 0.5;

  // Trial-wise drift, boundary, and t0
  vector[N] v_t;
  vector[N] a_t;
  vector[N] t0_t;

  // baseline t0 → logit space
  real eta0 = logit((t0 - t0_lower) / (rt_upper_t0 - t0_lower));

  for (t in 1:N) {
    // MB expected values
    real qL = p_common * V[mb1[t]] + (1 - p_common) * V[mb2[t]];
    real qR = p_common * V[mb2[t]] + (1 - p_common) * V[mb1[t]];
    real qdiff = qL - qR;

    // Drift (unchanged structure)
    v_t[t] = scaler * qdiff;

    // Boundary decline across trials
    a_t[t] = a * exp(-k_decline * trial_scaled[t]);

    // Non-decision time regression
    real t0_lin = eta0
                  + b_U    * U_chosen_z[t]
                  + b_d1   * d1[t]
                  + b_d2   * d2[t]
                  + b_ship * log1p_ship_lag_z[t]
                  + b_pair * log1p_pair_lag_z[t];
    t0_t[t] = logit_to_range(t0_lin, t0_lower, rt_upper_t0);

    // Online RW update
    V[s2[t]] = V[s2[t]] + alpha * (reward[t] - V[s2[t]]);
  }
}

model {
  // Priors (close to your vanilla defaults)
  alpha  ~ beta(1.5, 1.5);
  a      ~ normal(1.2, 0.5);
  t0     ~ normal(0.30, 0.10);
  w      ~ beta(2, 2);
  scaler ~ lognormal(0, 0.5);

  // New parameters
  k_decline ~ normal(0, 0.5);   // half-normal via <0
  b_U   ~ normal(0, 1);
  b_d1  ~ normal(0, 1);
  b_d2  ~ normal(0, 1);
  b_ship~ normal(0, 1);
  b_pair~ normal(0, 1);

  // Likelihood
  for (t in 1:N) {
    real v_use = (choice[t] == 1) ? v_t[t] : -v_t[t];
    target += wiener_lpdf(rt[t] | a_t[t], t0_t[t], w, v_use);
  }
}

generated quantities {
  // Simple posterior predictive check (approx)
  vector[N] rt_rep;
  for (t in 1:N) {
    real mu = log(rt[t] + 1e-3);
    real sd = 0.25;
    rt_rep[t] = exp(normal_rng(mu, sd));
  }
}
