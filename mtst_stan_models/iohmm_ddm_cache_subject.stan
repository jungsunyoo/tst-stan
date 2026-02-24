
functions {
  real logit_to_range(real x, real low, real high) {
    return low + inv_logit(x) * (high - low);
  }
}

data {
  int<lower=1> N;
  vector<lower=0>[N] rt;
  array[N] int<lower=0, upper=1> choice; // 1=mb2 (upper)

  // emission predictors
  int<lower=1> K_v;
  matrix[N, K_v] X_v;

  int<lower=1> K_t0;
  matrix[N, K_t0] X_t0;

  int<lower=1> K_a;
  matrix[N, K_a] X_a;

  // transition predictors (use t=2..N; row 1 ignored)
  int<lower=1> K_tr;
  matrix[N, K_tr] X_tr;

  real<lower=0> t0_lower;
  real<lower=t0_lower> t0_upper;
}

parameters {
  // initial probability of OPTION state (state=2)
  real pi0_logit;

  // state-specific parameters (state 1 = MENU, state 2 = OPTION)
  vector[2] log_a0;
  vector[2] w_logit;
  vector[2] v0;
  matrix[2, K_v] b_v;

  vector[2] eta_t0_0;
  matrix[2, K_t0] b_t0;

  matrix[2, K_a] b_a;

  // transitions: probability(next=OPTION) depends on previous state
  // delta_from_menu: used when prev state = MENU (1)
  // delta_from_opt:  used when prev state = OPTION (2)
  vector[K_tr] delta_from_menu;
  vector[K_tr] delta_from_opt;
}

transformed parameters {
  vector[2] a0;
  vector[2] w;
  for (k in 1:2) {
    a0[k] = exp(log_a0[k]);
    w[k]  = inv_logit(w_logit[k]);
  }
}

model {
  // priors (weakly informative)
  pi0_logit ~ normal(0, 1);

  log_a0 ~ normal(log(1.2), 0.5);
  w_logit ~ normal(0, 1);
  v0 ~ normal(0, 1);
  to_vector(b_v) ~ normal(0, 1);

  eta_t0_0 ~ normal(0, 1);
  to_vector(b_t0) ~ normal(0, 1);
  to_vector(b_a) ~ normal(0, 0.5);

  delta_from_menu ~ normal(0, 1);
  delta_from_opt  ~ normal(0, 1);

  // forward algorithm in log space
  vector[2] alpha;          // log p(y_{1:t}, z_t=k)
  vector[2] alpha_new;
  vector[2] ll;             // emission log-lik at time t

  // t=1 emissions
  for (k in 1:2) {
    real v_t = v0[k] + dot_product(row(X_v, 1), row(b_v, k));
    real v_use = (choice[1] == 1) ? v_t : -v_t;
    real a_t = fmax(a0[k] * exp(dot_product(row(X_a, 1), row(b_a, k))), 1e-6);
    real t0_t = logit_to_range(eta_t0_0[k] + dot_product(row(X_t0, 1), row(b_t0, k)),
                               t0_lower, t0_upper);
    ll[k] = wiener_lpdf(rt[1] | a_t, t0_t, w[k], v_use);
  }

  {
    real p_opt0 = inv_logit(pi0_logit);
    alpha[1] = log1m(p_opt0) + ll[1]; // MENU
    alpha[2] = log(p_opt0)   + ll[2]; // OPTION
  }

  // t>=2
  for (t in 2:N) {
    // emissions
    for (k in 1:2) {
      real v_t = v0[k] + dot_product(row(X_v, t), row(b_v, k));
      real v_use = (choice[t] == 1) ? v_t : -v_t;
      real a_t = fmax(a0[k] * exp(dot_product(row(X_a, t), row(b_a, k))), 1e-6);
      real t0_t = logit_to_range(eta_t0_0[k] + dot_product(row(X_t0, t), row(b_t0, k)),
                                 t0_lower, t0_upper);
      ll[k] = wiener_lpdf(rt[t] | a_t, t0_t, w[k], v_use);
    }

    // transitions (probability to be in OPTION at time t depends on z_{t-1})
    real p_opt_given_menu = inv_logit(dot_product(row(X_tr, t), delta_from_menu));
    real p_opt_given_opt  = inv_logit(dot_product(row(X_tr, t), delta_from_opt));

    // MENU at time t
    alpha_new[1] = ll[1] + log_sum_exp(
      alpha[1] + log1m(p_opt_given_menu),  // MENU->MENU
      alpha[2] + log1m(p_opt_given_opt)    // OPT->MENU
    );

    // OPTION at time t
    alpha_new[2] = ll[2] + log_sum_exp(
      alpha[1] + log(p_opt_given_menu),    // MENU->OPT
      alpha[2] + log(p_opt_given_opt)      // OPT->OPT
    );

    alpha = alpha_new;
  }

  target += log_sum_exp(alpha); // marginal likelihood
}

generated quantities {
  vector[N] log_lik;
  // compute incremental log likelihoods: log p(y_t | y_{1:t-1})
  {
    vector[2] alpha;
    vector[2] alpha_new;
    vector[2] ll;
    vector[N] prefix;
    prefix[1] = negative_infinity();

    // t=1
    for (k in 1:2) {
      real v_t = v0[k] + dot_product(row(X_v, 1), row(b_v, k));
      real v_use = (choice[1] == 1) ? v_t : -v_t;
      real a_t = fmax(a0[k] * exp(dot_product(row(X_a, 1), row(b_a, k))), 1e-6);
      real t0_t = logit_to_range(eta_t0_0[k] + dot_product(row(X_t0, 1), row(b_t0, k)),
                                 t0_lower, t0_upper);
      ll[k] = wiener_lpdf(rt[1] | a_t, t0_t, w[k], v_use);
    }
    {
      real p_opt0 = inv_logit(pi0_logit);
      alpha[1] = log1m(p_opt0) + ll[1];
      alpha[2] = log(p_opt0) + ll[2];
    }
    prefix[1] = log_sum_exp(alpha);
    log_lik[1] = prefix[1];

    for (t in 2:N) {
      for (k in 1:2) {
        real v_t = v0[k] + dot_product(row(X_v, t), row(b_v, k));
        real v_use = (choice[t] == 1) ? v_t : -v_t;
        real a_t = fmax(a0[k] * exp(dot_product(row(X_a, t), row(b_a, k))), 1e-6);
        real t0_t = logit_to_range(eta_t0_0[k] + dot_product(row(X_t0, t), row(b_t0, k)),
                                   t0_lower, t0_upper);
        ll[k] = wiener_lpdf(rt[t] | a_t, t0_t, w[k], v_use);
      }

      real p_opt_given_menu = inv_logit(dot_product(row(X_tr, t), delta_from_menu));
      real p_opt_given_opt  = inv_logit(dot_product(row(X_tr, t), delta_from_opt));

      alpha_new[1] = ll[1] + log_sum_exp(
        alpha[1] + log1m(p_opt_given_menu),
        alpha[2] + log1m(p_opt_given_opt)
      );
      alpha_new[2] = ll[2] + log_sum_exp(
        alpha[1] + log(p_opt_given_menu),
        alpha[2] + log(p_opt_given_opt)
      );

      alpha = alpha_new;
      prefix[t] = log_sum_exp(alpha);
      log_lik[t] = prefix[t] - prefix[t-1];
    }
  }
}
