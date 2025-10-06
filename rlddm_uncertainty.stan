functions {
  real logit_to_range(real x, real low, real high) {
    return low + inv_logit(x) * (high - low);
  }
}

data {
  int<lower=1> N;
  vector<lower=0>[N] rt;

  // new-style arrays
  array[N] int<lower=0, upper=1> choice;

  int<lower=1> S;
  array[N] int<lower=1, upper=S> mb1;
  array[N] int<lower=1, upper=S> mb2;
  array[N] int<lower=1, upper=S> s2;
  array[N] int<lower=0, upper=1> choice2;
  vector<lower=0, upper=1>[N] reward;

  real<lower=0> rt_upper_t0;
  real<lower=0> t0_lower;

  // --- core covariates already in your model ---
  vector[N] U_chosen_z;        // Beta-Bernoulli value-variance (legacy; can be zeros)
  vector[N] d1;
  vector[N] d2;
  vector[N] log1p_ship_lag_z;
  vector[N] log1p_pair_lag_z;
  vector[N] trial_scaled;

  // --- NEW: optional alternative covariates ---
  vector[N] U_val_z;           // Kalman-based value uncertainty (this request)
  vector[N] U_menu_z;          // Menu-graph uncertainty (this request)

  real<lower=0, upper=1> p_common; // 0.7
}

parameters {
  // RL / DDM core (unchanged)
  real<lower=0, upper=1> alpha;
  real<lower=0.2, upper=3.5> a;
  real<lower=t0_lower, upper=rt_upper_t0> t0;
  real<lower=0, upper=1> w;
  real<lower=0> scaler;

  // boundary decline (unchanged)
  real<lower=0> k_decline;

  // t0 regression (existing + NEW terms)
  real b_U;        // legacy Beta-Bernoulli uncertainty
  real b_d1;
  real b_d2;
  real b_ship;
  real b_pair;

  // --- NEW: coefficients for alternative uncertainties ---
  real b_Uval;     // KF value-uncertainty
  real b_Umenu;    // Menu-graph uncertainty
}

transformed parameters {
  vector[S] V;
  for (s in 1:S) V[s] = 0.5;

  vector[N] v_t;
  vector[N] a_t;
  vector[N] t0_t;

  real eta0 = logit((t0 - t0_lower) / (rt_upper_t0 - t0_lower));

  for (t in 1:N) {
    real qL = p_common * V[mb1[t]] + (1 - p_common) * V[mb2[t]];
    real qR = p_common * V[mb2[t]] + (1 - p_common) * V[mb1[t]];
    real qdiff = qL - qR;

    v_t[t] = scaler * qdiff;
    a_t[t] = a * exp(-k_decline * trial_scaled[t]);

    // t0 linear predictor: add NEW covariates (U_val_z, U_menu_z)
    real t0_lin =
        eta0
      + b_U    * U_chosen_z[t]
      + b_Uval * U_val_z[t]      // NEW
      + b_Umenu* U_menu_z[t]     // NEW
      + b_d1   * d1[t]
      + b_d2   * d2[t]
      + b_ship * log1p_ship_lag_z[t]
      + b_pair * log1p_pair_lag_z[t];

    t0_t[t] = logit_to_range(t0_lin, t0_lower, rt_upper_t0);

    V[s2[t]] = V[s2[t]] + alpha * (reward[t] - V[s2[t]]);
  }
}

model {
  // weakly-informative priors
  alpha  ~ beta(1.5, 1.5);
  a      ~ normal(1.2, 0.5);
  t0     ~ normal(0.30, 0.10);
  w      ~ beta(2, 2);
  scaler ~ lognormal(0, 0.5);

  k_decline ~ normal(0, 0.5);

  b_U    ~ normal(0, 1);
  b_Uval ~ normal(0, 1);   // NEW
  b_Umenu~ normal(0, 1);   // NEW
  b_d1   ~ normal(0, 1);
  b_d2   ~ normal(0, 1);
  b_ship ~ normal(0, 1);
  b_pair ~ normal(0, 1);

  for (t in 1:N) {
    real v_use = (choice[t] == 1) ? v_t[t] : -v_t[t];
    target += wiener_lpdf(rt[t] | a_t[t], t0_t[t], w, v_use);
  }
}

generated quantities {
  vector[N] rt_rep;
  for (t in 1:N) {
    real mu = log(rt[t] + 1e-3);
    real sd = 0.25;
    rt_rep[t] = exp(normal_rng(mu, sd));
  }
}
