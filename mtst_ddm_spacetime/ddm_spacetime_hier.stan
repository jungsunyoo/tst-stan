functions {
  real logit_to_range(real x, real low, real high) {
    return low + inv_logit(x) * (high - low);
  }
}

data {
  int<lower=1> N;
  int<lower=1> J;
  array[N] int<lower=1, upper=J> subj_id;

  vector<lower=0>[N] rt;
  array[N] int<lower=0, upper=1> choice;   // 0 = lower-coded option/action, 1 = upper-coded option/action

  int<lower=1> K_v;
  matrix[N, K_v] X_v;

  int<lower=1> K_t0;
  matrix[N, K_t0] X_t0;

  int<lower=1> K_a;
  matrix[N, K_a] X_a;

  real<lower=0> t0_lower;
  vector<lower=t0_lower>[J] t0_upper;
}

parameters {
  // subject baselines
  real mu_log_a;
  real<lower=0> sigma_log_a;
  vector[J] z_log_a;

  real mu_w_logit;
  real<lower=0> sigma_w_logit;
  vector[J] z_w_logit;

  real mu_v0;
  real<lower=0> sigma_v0;
  vector[J] z_v0;

  real mu_eta_t0;
  real<lower=0> sigma_eta_t0;
  vector[J] z_eta_t0;

  // shared slopes
  vector[K_v] b_v;
  vector[K_t0] b_t0;
  vector[K_a] b_a;
}

transformed parameters {
  vector[J] a0;
  vector[J] w;
  vector[J] v0;
  vector[J] eta_t0_0;

  for (j in 1:J) {
    a0[j] = exp(mu_log_a + sigma_log_a * z_log_a[j]);
    w[j]  = inv_logit(mu_w_logit + sigma_w_logit * z_w_logit[j]);
    v0[j] = mu_v0 + sigma_v0 * z_v0[j];
    eta_t0_0[j] = mu_eta_t0 + sigma_eta_t0 * z_eta_t0[j];
  }
}

model {
  // hyperpriors
  mu_log_a ~ normal(log(1.2), 0.5);
  sigma_log_a ~ normal(0, 0.5);

  mu_w_logit ~ normal(0, 1.0);
  sigma_w_logit ~ normal(0, 1.0);

  mu_v0 ~ normal(0, 1.0);
  sigma_v0 ~ normal(0, 1.0);

  mu_eta_t0 ~ normal(0, 1.0);
  sigma_eta_t0 ~ normal(0, 1.0);

  z_log_a ~ std_normal();
  z_w_logit ~ std_normal();
  z_v0 ~ std_normal();
  z_eta_t0 ~ std_normal();

  // slopes
  b_v  ~ normal(0, 1.0);
  b_t0 ~ normal(0, 1.0);
  b_a  ~ normal(0, 0.35);

  for (t in 1:N) {
    int j = subj_id[t];
    real v_t = v0[j] + dot_product(row(X_v, t), b_v);
    real v_use = (choice[t] == 1) ? v_t : -v_t;

    // safe boundary transform to avoid overflow
    real eta_a = dot_product(row(X_a, t), b_a);
    real a_t = fmax(a0[j] * exp(fmin(fmax(eta_a, -20), 20)), 1e-6);

    real t0_t = logit_to_range(eta_t0_0[j] + dot_product(row(X_t0, t), b_t0),
                               t0_lower, t0_upper[j]);

    target += wiener_lpdf(rt[t] | a_t, t0_t, w[j], v_use);
  }
}

generated quantities {
  vector[N] log_lik;
  for (t in 1:N) {
    int j = subj_id[t];
    real v_t = v0[j] + dot_product(row(X_v, t), b_v);
    real v_use = (choice[t] == 1) ? v_t : -v_t;
    real eta_a = dot_product(row(X_a, t), b_a);
    real a_t = fmax(a0[j] * exp(fmin(fmax(eta_a, -20), 20)), 1e-6);
    real t0_t = logit_to_range(eta_t0_0[j] + dot_product(row(X_t0, t), b_t0),
                               t0_lower, t0_upper[j]);
    log_lik[t] = wiener_lpdf(rt[t] | a_t, t0_t, w[j], v_use);
  }
}
