use crate::definitions::*;
use rand::prelude::*;

struct MarketState {
    pub x1: f64, // log S1(t)
    pub x2: f64, // log S2(t)
    pub hit: bool,
    pub adj: f64,
    pub hit_a: bool, // antithetic path hits boundary
    pub adj_a: f64,
}

pub fn pricing(rng: &mut StdRng, num_simulations: u32, p: &Params) -> f64 {
    let drift1 = (p.rf - 0.5 * p.sigm1 * p.sigm1) * (p.T);
    let drift2 = (p.rf - 0.5 * p.sigm2 * p.sigm2) * (p.T);
    let mut history = Vec::with_capacity(num_simulations as usize);
    for _i in 0..num_simulations {
        let s: MarketState = MarketState {
            x1: p.X0,
            x2: p.X0,
            hit: p.X0 >= p.m,
            adj: 1.0,
            hit_a: p.X0 >= p.m,
            adj_a: 1.0,
        };

        // correlated standard normal.
        let (dz1, dz2) = sample_correlated_normal(rng, p.corr);
        let (dz1, dz2) = (dz1 * p.sigm1 * p.dw, dz2 * p.sigm2 * p.dw);

        let payoff = {
            let (x1, x2) = (s.x1 + drift1 + dz1, s.x2 + drift2 + dz2);
            let (adj, _hit):(f64, bool) = 
                // Upper barrier hit by X2
                if (x2 >= p.m) || s.hit{
                    (0., true)
                } else {
                    // Probability of brownian bridge hit, for given X1 and X2.
                    let prob = intraday_hit_prob(s.x2, x2, p.T, p.m, p.sigm2);
                    (s.adj * (1.0 - prob), false)
            };
            let payoff = (x1.exp() - p.K.exp()).max(0.0);
            payoff * adj
        };

        //payoff of antithetic path
        let payoff_a ={
            let (x1, x2) = (s.x1 + drift1 - dz1, s.x2 + drift2 - dz2);
            // Antithetic X2 hits barrier
            let (adj_a, _hit_a) = if (x2 >= p.m) || s.hit_a {
                (0., true)
            } else {
                // Antithetic path hits brownian bridge
                let prob =
                    intraday_hit_prob(s.x2, x2, p.T, p.m, p.sigm2);
                (s.adj_a * (1.0-prob), false)
            };
            let payoff_a = (x1.exp() - p.K.exp()).max(0.0);
            payoff_a * adj_a
        };
        
        history.push(0.5*(payoff + payoff_a));
    }
    let discount = (-p.rf * p.T).exp();
    history.iter().sum::<f64>() * discount / (history.len() as f64)
}
