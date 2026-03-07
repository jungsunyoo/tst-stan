functions {
  real logit_to_range(real x, real low, real high) {
    return low + inv_logit(x) * (high - low);
  }
}

data {
  int<lower=1> N;
  vector<lower=0>[N] rt;
  array[N] int<lower=0, upper=1> choice;

  int<lower=1> K_v;
  matrix[N, K_v] X_v;

  int<lower=1> K_t0;
  matrix[N, K_t0] X_t0;

  int<lower=1> K_a;
  matrix[N, K_a] X_a;

  real<lower=0> t0_lower;
  real<lower=t0_lower> t0_upper;
}

parameters {
  real<lower=0.2, upper=4.0> a0;
  vector[K_a] b_a;

  real<lower=0, upper=1> w;

  real v0;
  vector[K_v] b_v;

  real eta_t0_0;
  vector[K_t0] b_t0;
}

transformed parameters {
  vector[N] v_t;
  vector[N] a_t;
  vector[N] t0_t;

  for (t in 1:N) {
    real eta_a;
    v_t[t] = v0 + dot_product(row(X_v, t), b_v);

    eta_a = dot_product(row(X_a, t), b_a);
    eta_a = fmin(eta_a, 20);
    eta_a = fmax(eta_a, -20);
    a_t[t] = fmax(a0 * exp(eta_a), 1e-6);

    t0_t[t] = logit_to_range(
      eta_t0_0 + dot_product(row(X_t0, t), b_t0),
      t0_lower, t0_upper
    );
  }
}

model {
  a0  ~ normal(1.2, 0.5);
  b_a ~ normal(0, 0.35);

  w   ~ beta(2, 2);

  v0  ~ normal(0, 1.0);
  b_v ~ normal(0, 1.0);

  eta_t0_0 ~ normal(0, 1.0);
  b_t0     ~ normal(0, 1.0);

  for (t in 1:N) {
    real v_use = (choice[t] == 1) ? v_t[t] : -v_t[t];
    target += wiener_lpdf(rt[t] | a_t[t], t0_t[t], w, v_use);
  }
}

generated quantities {
  vector[N] log_lik;
  for (t in 1:N) {
    real v_use = (choice[t] == 1) ? v_t[t] : -v_t[t];
    log_lik[t] = wiener_lpdf(rt[t] | a_t[t], t0_t[t], w, v_use);
  }
}
