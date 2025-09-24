// file: rlddm_single_subject.stan
functions {
  real PI() { return 3.1415926535897932384626433832795; }

  // Small-time expansion (Navarro–Fuss style). Sum k = -K..K without negative ranges.
  real wiener_pdf_small(real t, real a, real v, real w, int K) {
    real sum = 0;
    // k = 0..K (positive & zero terms)
    for (k in 0:K) {
      real m_pos  = 2.0 * k + w;
      real num_pos = a * m_pos - v * t;
      sum += m_pos * exp( - square(num_pos) / (2.0 * t) );

      // k = -1..-K (mirror), avoid double-counting k=0
      if (k > 0) {
        real m_neg  = 2.0 * (-k) + w;
        real num_neg = a * m_neg - v * t;
        sum += m_neg * exp( - square(num_neg) / (2.0 * t) );
      }
    }
    return (a / sqrt(2.0 * PI() * pow(t, 3))) * exp( -0.5 * square(v) * t ) * sum;
  }

  // Large-time expansion (Navarro–Fuss style).
  real wiener_pdf_large(real t, real a, real v, real w, int K) {
    real sum = 0;
    for (k in 1:K) {
      real lam = k * PI();
      sum += lam * sin(lam * w) * exp( -0.5 * (square(lam) * t) / square(a) );
    }
    return (2.0 / square(a)) * exp( v * a * w - 0.5 * square(v) * t ) * sum;
  }

  // Wiener log-PDF wrapper (reflect to upper boundary when choice==0)
  real wiener_lpdf(real rt, int choice, real a, real v, real w, real t0) {
    if (rt <= t0) return negative_infinity();
    real t = rt - t0;
    real w_eff = choice == 0 ? w : 1.0 - w;
    real v_eff = choice == 0 ? v : -v;
    real thresh = 0.1 * square(a);
    int  K      = 10;  // your speed tweak
    real dens = (t < thresh)
                ? wiener_pdf_small(t, a, v_eff, w_eff, K)
                : wiener_pdf_large(t, a, v_eff, w_eff, K);
    if (!(dens > 0)) return negative_infinity();
    return log(dens);
  }
}

data {
  int<lower=1> N;                             // trials
  vector<lower=1e-6>[N] rt;                   // RT (s)
  array[N] int<lower=0,upper=1> choice;       // 0/1 (lower/upper)
  array[N] int<lower=0,upper=1> choice2;      // 0/1 (lower/upper action)
  array[N] int<lower=1> state1;               // 1..S
  array[N] int<lower=1> state2;               // 1..S
  int<lower=2> S;                             // # states
  vector<lower=0,upper=1>[N] reward;          // reward in [0,1]
}

transformed data {
  // Make constants visible to all blocks (model + generated quantities)
  real z = 0.5;
  real theta = 1.0;
}

parameters {
  real<lower=0,upper=1> alpha;                // learning rate
  real<lower=0,upper=4>         scaler;               // drift scale
  real<lower=0.3>       a;                    // boundary separation
  real<lower=0.05>      t0;                   // non-decision time
}

model {
  // Priors (adjust as needed)
//   alpha  ~ beta(2, 2);
//   scaler ~ normal(1.5, 0.5);
//   a      ~ normal(1.2, 0.4);
//   t0     ~ normal(0.30, 0.12);



  // tighter priors to improve sampling
  alpha  ~ beta(5,5);
  scaler ~ normal(1.0, 0.3);  // and keep parameter <upper=4>
  a      ~ normal(1.0, 0.25);
  t0     ~ normal(0.30, 0.08);


  // RL values (per-state-action)
  matrix[S, 2] Q = rep_matrix(0.5, S, 2);

  // Likelihood with sequential RL updates
  for (n in 1:N) {
    int sL = state1[n];
    int sR = state2[n];
    
    // Model-based Q-values with transition probabilities (70% common, 30% rare)
    real prob_common = 0.7;
    real prob_rare = 1.0 - prob_common;
    
    // For each first-stage state, compute expected value considering transition probabilities
    real mb_q_sL = prob_common * max(Q[sL, :]) + prob_rare * max(Q[sR, :]);
    real mb_q_sR = prob_common * max(Q[sR, :]) + prob_rare * max(Q[sL, :]);

    real dv = mb_q_sL - mb_q_sR;  // Compare model-based Q-values
    real v  = scaler * dv;
    target += wiener_lpdf(rt[n] | choice[n], a, v, z, t0);  
    // real dv = Q[sL, 2] - Q[sR, 1];  // Compare upper choice for sL vs lower choice for sR
    // real v  = scaler * dv;
    // target += wiener_lpdf(rt[n] | choice[n], a, v, z, t0);

    int sC = choice[n] == 0 ? sL : sR;       // chosen state (sL if choice=0 (upper), sR if choice=1 (lower))
    int aC = choice2[n] + 1;                  // chosen action (convert 0/1 to 1/2)
    real pe = reward[n] - Q[sC, aC];
    Q[sC, aC]  += alpha * theta * pe;
  }
}

generated quantities {
  vector[N] log_lik;
  {
    matrix[S, 2] Qg = rep_matrix(0.5, S, 2);
    for (n in 1:N) {
      int sL = state1[n];
      int sR = state2[n];
      
      // Model-based Q-values with transition probabilities (70% common, 30% rare)
      real prob_common = 0.7;
      real prob_rare = 1.0 - prob_common;
      
      real mb_q_sL = prob_common * max(Qg[sL, :]) + prob_rare * max(Qg[sR, :]);
      real mb_q_sR = prob_common * max(Qg[sR, :]) + prob_rare * max(Qg[sL, :]);
      real dv = mb_q_sL - mb_q_sR;
      real v  = scaler * dv;
      log_lik[n] = wiener_lpdf(rt[n] | choice[n], a, v, z, t0);
      int sC = choice[n] == 0 ? sL : sR;       // chosen state (sL if choice=0 (upper), sR if choice=1 (lower))
      int aC = choice2[n] + 1;                  // chosen action (convert 0/1 to 1/2)
      real pe = reward[n] - Qg[sC, aC];
      Qg[sC, aC] += alpha * theta * pe;
    }
  }
}
