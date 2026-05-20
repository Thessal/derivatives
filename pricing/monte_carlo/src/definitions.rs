// Definitions
use rand::{RngExt, prelude::*};

pub fn box_muller(rng: &mut StdRng) -> (f64, f64) {
    // Box-Muller transform
    let u1: f64 = rng.random_range(0.0..=1.0);
    let u2: f64 = rng.random_range(0.0..=2.0 * std::f64::consts::PI);
    let r: f64 = (-2.0f64 * u1.ln()).sqrt();
    let z1: f64 = r * (u2.cos());
    let z2: f64 = r * (u2.sin());
    (z1, z2)
}

pub fn sample_correlated_normal(rng: &mut StdRng, corr: f64) -> (f64, f64) {
    // Generates correlated RVs.
    let (z1, z2) = box_muller(rng);
    (z1, corr * z1 + (1.0 - corr * corr).sqrt() * z2)
}

pub fn intraday_hit_prob(x1: f64, x2: f64, dt: f64, b: f64, sigm: f64) -> f64 {
    // Brownian bridge
    // prob that the log price hits barrier during the time step
    // To prove this, Brownian bridge in no-drift measure was used (reflection principle)
    // Girsanov factor cancels out.
    assert!(x1 < b);
    assert!(x2 < b);
    (-(2.0 * (b - x1) * (b - x2)) / (sigm * sigm * dt)).exp()
}

#[allow(non_snake_case)]
pub struct Params {
    pub(crate) X0: f64,   // Initial Log price
    pub(crate) K: f64,    // Strike price (log)
    pub(crate) m: f64,    // barrier level (log)
    pub(crate) T: f64,    // year
    pub(crate) rf: f64,   // annually, continuously compounded
    pub(crate) corr: f64, // correlation between two assets
    pub(crate) sigm1: f64,
    pub(crate) sigm2: f64,
    pub(crate) dw: f64, // scale for dw
}
#[allow(non_snake_case)]
impl Params {
    pub fn new(
        T: f64,
        S0: f64,
        exercise: f64, // not log price
        barrier: f64,  // not log price
        rf: f64,
        sigm1: f64,
        sigm2: f64,
        corr: f64,
    ) -> Self {
        Self {
            X0: S0.ln(),
            K: exercise.ln(),
            m: barrier.ln(),
            T,
            rf,
            corr,
            sigm1,
            sigm2,
            dw: (T).sqrt(),
        }
    }
}
