use rand::{prelude::*, rngs::SysRng};
mod antithetic;
mod definitions;
mod montecarlo;

use definitions::*;

fn main() {
    let num_simulations = 10_000_000; // Error of correlation of sampled data is less than 0.01.

    let start_price = 100.0;
    let exercise = 105.0;
    let barrier = 130.0;
    let ttm = 1.0;
    let rf = 0.03;
    let sigm1 = 0.2;
    let sigm2 = 0.3;
    let corr = 0.25;

    let param = Params::new(ttm, start_price, exercise, barrier, rf, sigm1, sigm2, corr);
    let mut rng: StdRng = StdRng::try_from_rng(&mut SysRng).unwrap();

    let price_mc = montecarlo::pricing(&mut rng, num_simulations, &param);
    println!("MC: {}", price_mc);
    let price_at = antithetic::pricing(&mut rng, num_simulations, &param);
    println!("AT: {}", price_at);
}

#[cfg(test)]
mod test {
    use super::*;
    use std::thread;
    use std::time::Instant;

    #[test]
    fn s0_scan() {
        println!("S0, MC, AT, Avg, Diff");

        let handles: Vec<_> = (60 * 4..=140 * 4)
            .map(|i| {
                thread::spawn(move || {
                    let num_simulations = 100_000_000;

                    let start_price = i as f64 / 4.0;
                    let exercise = 105.0;
                    let barrier = 130.0;
                    let ttm = 1.0;
                    let rf = 0.03;
                    let sigm1 = 0.2;
                    let sigm2 = 0.3;
                    let corr = 0.25;

                    let param =
                        Params::new(ttm, start_price, exercise, barrier, rf, sigm1, sigm2, corr);
                    let mut rng: StdRng = StdRng::try_from_rng(&mut SysRng).unwrap();

                    let price_mc = montecarlo::pricing(&mut rng, num_simulations, &param);
                    let price_at = antithetic::pricing(&mut rng, num_simulations, &param);
                    // assert!((price_mc - price_at).abs() < 1e-3);
                    println!(
                        "{}, {}, {}, {}, {}",
                        start_price,
                        price_mc,
                        price_at,
                        0.5 * (price_mc + price_at),
                        (price_mc - price_at).abs()
                    );
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn without_barrier() {
        let num_simulations = 10_000_000; // Error of correlation of sampled data is less than 0.01.

        let start_price = 100.0;
        let exercise = 100.0;
        let barrier = f64::MAX;
        let ttm = 1.0;
        let rf = 0.05;
        let sigm1 = 0.2;
        let sigm2 = 0.2;
        let corr = 0.0;

        let param = Params::new(ttm, start_price, exercise, barrier, rf, sigm1, sigm2, corr);
        let mut rng: StdRng = StdRng::try_from_rng(&mut SysRng).unwrap();

        let price_mc = montecarlo::pricing(&mut rng, num_simulations, &param);
        println!("MC: {}", price_mc);
        let price_at = antithetic::pricing(&mut rng, num_simulations, &param);
        println!("AT: {}", price_at);
        let theoretical = 10.4506;
        println!("BS: {}", theoretical);
        assert!((price_mc - theoretical).abs() < 1e-2);
        assert!((price_at - theoretical).abs() < 1e-2);
    }

    fn mean(data: &[f64]) -> Option<f64> {
        let sum = data.iter().sum::<f64>() as f64;
        let count = data.len();

        match count {
            positive if positive > 0 => Some(sum / count as f64),
            _ => None,
        }
    }

    fn std_deviation(data: &[f64]) -> Option<f64> {
        match (mean(data), data.len()) {
            (Some(data_mean), count) if count > 0 => {
                let variance = data
                    .iter()
                    .map(|value| {
                        let diff = data_mean - (*value as f64);

                        diff * diff
                    })
                    .sum::<f64>()
                    / count as f64;

                Some(variance.sqrt())
            }
            _ => None,
        }
    }

    #[test]
    fn convergence_comparison() {
        let start_price = 100.0;
        let exercise = 105.0;
        let barrier = 130.0;
        let ttm = 1.0;
        let rf = 0.03;
        let sigm1 = 0.2;
        let sigm2 = 0.3;
        let corr = 0.25;

        println!("Total step, MC, AT");
        for num_sim in (50..=1000).step_by(25) {
            let num_simulations = num_sim as u32;
            let param = Params::new(ttm, start_price, exercise, barrier, rf, sigm1, sigm2, corr);
            let mut rng: StdRng = StdRng::try_from_rng(&mut SysRng).unwrap();

            let price_mc: Vec<f64> = (0..300)
                .map(|_| montecarlo::pricing(&mut rng, num_simulations, &param))
                .collect();
            let price_at: Vec<f64> = (0..300)
                .map(|_| antithetic::pricing(&mut rng, num_simulations, &param))
                .collect();
            println!(
                "{}, {}, {}",
                num_sim,
                std_deviation(&price_mc).unwrap(),
                std_deviation(&price_at).unwrap()
            );
        }
    }

    #[test]
    fn speed_test() {
        let num_simulations = 10_000_000; // Error of correlation of sampled data is less than 0.01.

        let start_price = 100.0;
        let exercise = 105.0;
        let barrier = 130.0;
        let ttm = 1.0;
        let rf = 0.03;
        let sigm1 = 0.2;
        let sigm2 = 0.3;
        let corr = 0.25;

        let param = Params::new(ttm, start_price, exercise, barrier, rf, sigm1, sigm2, corr);
        let mut rng: StdRng = StdRng::try_from_rng(&mut SysRng).unwrap();

        println!("Testing simulation speed");
        println!("number of simulations: {}", num_simulations);

        let now = Instant::now();
        let _price_mc = montecarlo::pricing(&mut rng, num_simulations, &param);
        let new_now = Instant::now();
        println!("Monte Carlo: {:?}", new_now.duration_since(now));

        let now = Instant::now();
        let _price_at = antithetic::pricing(&mut rng, num_simulations, &param);
        let new_now = Instant::now();
        println!("Antithetic: {:?}", new_now.duration_since(now));
    }
}
