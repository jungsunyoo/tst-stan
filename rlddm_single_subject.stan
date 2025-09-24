// Two-stage RL + DDM (single subject), model-based only on stage 1
// Stage 1 likelihood only; Stage 2 choices update Q(s,a)
// choice coding: 0 = upper/left (mb1), 1 = lower/right (mb2)




functions {

  real softmax_max2(real q1, real q2, real tau) {
    // numerically stable: returns approx max(q1,q2)
    real m = fmax(q1, q2);
    return m + tau * log(exp((q1 - m)/tau) + exp((q2 - m)/tau));
  }
  
  // Small-time series (Navarro-Fuss style), simple and fast
  real wiener_pdf_small(real t, real a, real v, real w, int K) {
    real sum = 0;
    for (k in 0:K) {
      real mpos = 2.0 * k + w;
      real npos = a * mpos - v * t;
      sum += mpos * exp(- square(npos) / (2.0 * t));

      if (k > 0) {
        real mneg = -2.0 * k + w;
        real nneg = a * mneg - v * t;
        sum += mneg * exp(- square(nneg) / (2.0 * t));
      }
    }
    return (a / sqrt(2.0 * pi() * pow(t, 3))) * exp(-0.5 * square(v) * t) * sum;
  }

  // User-defined _lpdf supports conditional notation
  real wiener_lpdf(real rt, int choice, real a, real v, real t0) {
    if (rt <= t0) return negative_infinity();
    real t = rt - t0;
    real w = 0.5;  // fixed start bias

    // reflect for upper boundary (choice==0)
    real w_eff = (choice == 1) ? w : (1.0 - w);
    real v_eff = (choice == 1) ? v : -v;

    int K = 20;
    real dens = wiener_pdf_small(t, a, v_eff, w_eff, K);
    if (!(dens > 0)) return negative_infinity();
    return log(dens);
  }
}

data {
  int<lower=1> N;                          // trials
  vector<lower=1e-6>[N] rt;                // stage-1 RT (seconds)
  array[N] int<lower=0,upper=1> choice;    // stage-1 choice: 0=left(mb1), 1=right(mb2)

  // First-stage options mapped to second-stage states
  array[N] int<lower=1> mb1;               // left option (1..S)
  array[N] int<lower=1> mb2;               // right option (1..S)
  int<lower=2> S;                          // number of second-stage states (often 2)

  // Observed second-stage visit & action (for updates only)
  array[N] int<lower=1,upper=S> s2;        // visited second-stage state (1..S)
  array[N] int<lower=0,upper=1> choice2;   // chosen action at s2: 0=action1, 1=action2

  vector<lower=0,upper=1>[N] reward;       // reward in [0,1]
  real<lower=0.03> rt_upper_t0;            // strict upper bound for t0 (< min(rt))
}

// parameters {
//   real<lower=0,upper=1> alpha;             // learning rate
//   real<lower=0.35, upper=2.5> a;           // boundary separation
//   real<lower=0.03, upper=rt_upper_t0> t0;  // non-decision time
//   real<lower=0> scaler;                    // drift scale
// }

parameters {
  real<lower=0,upper=1> alpha;             // learning rate
  real<lower=0.5, upper=2.2> a;           // boundary separation
//   real<lower=0.6, upper=1.5> a;           // boundary separation
  real<lower=0.06, upper=rt_upper_t0> t0;  // non-decision time
//   real<lower=0> scaler;                    // drift scale
  real log_scaler; // unconstrained
}
transformed parameters {
  real<lower=0> scaler = exp(log_scaler);   // <-- move it here (not transformed data)
}

model {
  // Priors
//   alpha  ~ beta(2, 2);
//   a      ~ normal(1.2, 0.35);
//   t0     ~ normal(0.20, 0.06);
//   scaler ~ normal(1.0, 0.5);
  alpha  ~ beta(5, 5);
  a      ~ normal(1.2, 0.1);
  t0     ~ normal(0.25, 0.05);
//   scaler ~ lognormal(log(0.30), 0.40);
  log_scaler ~ normal(log(0.12), 0.35);  // ≈ lognormal(mean ~0.12)


  // Q(s,a): second-stage action-values
  array[S] vector[2] Q;
  for (s in 1:S) Q[s] = rep_vector(0.0, 2);

  // Likelihood + RL updates
  for (n in 1:N) {
    int sL = mb1[n];
    int sR = mb2[n];

    // model-based first-stage value difference: max_a Q[sL,a] - max_a Q[sR,a]
    // real qL = (Q[sL][1] > Q[sL][2]) ? Q[sL][1] : Q[sL][2];
    // real qR = (Q[sR][1] > Q[sR][2]) ? Q[sR][1] : Q[sR][2];
    real qL = softmax_max2(Q[sL][1], Q[sL][2], 0.03); // tau=0.03 works well
    real qR = softmax_max2(Q[sR][1], Q[sR][2], 0.03);

    real dv = qL - qR;
    real v  = scaler * dv;

    target += wiener_lpdf(rt[n] | choice[n], a, v, t0);

    // Second-stage update from visited state and chosen action (0->1, 1->2)
    int ss  = s2[n];
    int a2  = (choice2[n] == 1) ? 2 : 1;
    real pe = reward[n] - Q[ss][a2];
    Q[ss][a2] += alpha * pe;
  }
}

generated quantities {
  vector[N] log_lik;
  {
    array[S] vector[2] Qg;
    for (s in 1:S) Qg[s] = rep_vector(0.5, 2);

    for (n in 1:N) {
      int sL = mb1[n];
      int sR = mb2[n];
      real qL = (Qg[sL][1] > Qg[sL][2]) ? Qg[sL][1] : Qg[sL][2];
      real qR = (Qg[sR][1] > Qg[sR][2]) ? Qg[sR][1] : Qg[sR][2];
      real dv = qL - qR;
      real v  = scaler * dv;

      log_lik[n] = wiener_lpdf(rt[n] | choice[n], a, v, t0);

      int ss  = s2[n];
      int a2  = (choice2[n] == 1) ? 2 : 1;
      real pe = reward[n] - Qg[ss][a2];
      Qg[ss][a2] += alpha * pe;
    }
  }
}
