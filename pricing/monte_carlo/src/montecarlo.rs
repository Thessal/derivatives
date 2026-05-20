use crate::definitions::*;
use rand::prelude::*;

struct MarketState {
    pub x1: f64, // log S1(t)
    pub x2: f64, // log S2(t)
    pub hit: bool,
    pub prob_weight: f64,
}

#[allow(unused_parens)]
fn sample(rng: &mut StdRng, p: &Params, s: &mut MarketState) {
    // correlated standard normal.
    let (dz1, dz2) = sample_correlated_normal(rng, p.corr);
    // risk-free measure
    let drift1 = (p.rf - 0.5 * p.sigm1 * p.sigm1) * p.T;
    let drift2 = (p.rf - 0.5 * p.sigm2 * p.sigm2) * p.T;
    let (dz1, dz2) = (drift1 + dz1 * p.sigm1 * p.dw, drift2 + dz2 * p.sigm2 * p.dw);

    // assert!(!s.hit); // No need to sample if already hit.
    // assert!(s.x2 < p.m);

    s.x1 += dz1;
    s.x2 += dz2;

    if (s.x2 >= p.m || s.hit) {
        // barrier hit by X2, terminate.
        s.hit = true;
        s.prob_weight = 0.0;
    } else {
        // barrier hit by brownian bridge, given X1, X2 < m. Terminate p fraction of paths.
        let prob = intraday_hit_prob(s.x2 - dz2, s.x2, p.T, p.m, p.sigm2);
        // non-terminating probability is 1 - p
        s.prob_weight *= (1.0 - prob);
    }
}

pub fn pricing(rng: &mut StdRng, num_simulations: u32, param: &Params) -> f64 {
    let mut prob_times_payoff_sum: f64 = 0.0; // prob * payoff
    for _i in 0..num_simulations {
        let mut state = MarketState {
            x1: param.X0,
            x2: param.X0,
            hit: param.X0 >= param.m,
            prob_weight: if param.X0 >= param.m { 0.0 } else { 1.0 },
        };
        sample(rng, &param, &mut state);
        // expected payoff
        prob_times_payoff_sum += state.prob_weight * (state.x1.exp() - param.K.exp()).max(0.0);
    }
    prob_times_payoff_sum /= num_simulations as f64;
    let option_value = prob_times_payoff_sum * ((-param.rf * param.T).exp());
    option_value
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::rngs::SysRng;

    #[test]
    fn test_random_two_samples() {
        let mut rng: StdRng = StdRng::try_from_rng(&mut SysRng).unwrap();
        let n = 1_000_000;
        for corr in [-1.0, -0.4, 0.0, 0.4, 1.0] {
            let samples: Vec<(f64, f64)> = (0..n)
                .map(|_| sample_correlated_normal(&mut rng, corr))
                .collect();
            let cov = samples.iter().map(|(x, y)| x * y).sum::<f64>() / n as f64;
            assert_eq!((cov * 100.0).round(), (corr * 100.0));
        }
    }
}
