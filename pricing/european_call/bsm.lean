/-
I'm trying to formalize Black-Scholes model in Lean 4.
But it seems that I have to learn other things first...
After looking at Line 80 - 100, I realized that formalization can be non-trivial, especially when the complexity is not in the type, but in the lengthy equation.
-/

import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.MeasureTheory.Measure.MeasureSpace
import Mathlib.MeasureTheory.Integral.Lebesgue.Basic
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic
import Mathlib.MeasureTheory.Measure.Decomposition.RadonNikodym
import Mathlib.Analysis.Calculus.Deriv.Basic

open Real MeasureTheory Set Filter -- ProbabilityTheory -- Let's just define some axioms instead

noncomputable section BlackScholesFormalization
-- Real numbers (R) and Integrals (∫) are noncomputable

/-!
# Black-Scholes Formalization
-/

variable (S_t E r_f r_fin σ x t T : ℝ)
variable (T_t : ℝ) -- Represents (T - t)
variable (μ : Measure ℝ)
-- variable (N : Measure ℝ) [IsProbabilityMeasure N] -- \integral 1 dN = 1
-- variable (hμ : μ = volume) -- \integral f(x) dμ = \integral f(x) dx
variable (hσ : 0 < σ)

/-!
### 0. Standard normal distribution
We define the distribution explicitly instead of importing mathlib's probability spaces.
-/

def n (x : ℝ) : ℝ := (1 / sqrt (2 * π)) * exp (- (x^2 / 2))
def N (x : ℝ) : ℝ := ∫ t in Iic x, n t

-- We axiomatize the core properties of the Standard Normal distribution
-- We are treating LEAN like a calculator, but we really shouldn't. Why not just use sympy?
axiom hN_deri : ∀ x, deriv N x = n x
axiom hN_rlim : ∫ x, n x = 1
axiom hN_llim : Filter.Tendsto (fun a ↦ ∫ x in Iic a, n x) Filter.atBot (nhds 0)
axiom hN_symm : ∀ x, 1 - N (-x) = N x


/-!
### 1. Definition of d1 and d2
We explicitly write `d1` and `d2` formulas as they appear in closed form.
We derived it by hand, so let's check if there's no problem
-/
def d1 (S_t E r_f σ T_t : ℝ) : ℝ := (log (S_t / E) + (r_f + σ^2 / 2) * T_t) / (σ * sqrt T_t)
def d2 (S_t E r_f σ T_t : ℝ) : ℝ := (log (S_t / E) + (r_f - σ^2 / 2) * T_t) / (σ * sqrt T_t)


/-!
### 2. Call option price
We formulate the Random Variable `S(T)` as a function of the variable `x`
X is supposed to have standard normal distribution.
However is is not formally described here because I don't know how to describe path integral in LEAN.
-/
def S_T (S_t r_f σ T_t x : ℝ) : ℝ := S_t * exp ((r_f - σ^2 / 2) * T_t + σ * sqrt T_t * x)

def payoff (S_T E : ℝ) : ℝ := if S_T - E > 0 then (S_T - E) else 0

def call_option_price (μ : Measure ℝ) (S_t E r_f r_fin σ T_t : ℝ) : ℝ :=
  exp (- r_fin * T_t) * ∫ x, payoff (S_T S_t r_f σ T_t x) E * n x ∂μ

-- What we derived by hand
def call_option_solution (S_t E r_f r_fin σ T_t : ℝ) : ℝ :=
  S_t * exp ((r_f - r_fin) * T_t) * N (d1 S_t E r_f σ T_t) - E * exp (- r_fin * T_t) * N (d2 S_t E r_f σ T_t)


/-!
### 3. Verify the calculation
-/

theorem black_scholes_verification
  (h_S : 0 < S_t) (h_E : 0 < E) (h_σ : 0 < σ) (h_T : 0 < T_t) (h_μ : μ = volume) : -- assumptions
  -- (h_S : 0 < S_t) (h_E : 0 < E) (h_σ : 0 < σ) (h_T : 0 < T_t) : -- assumptions required
  call_option_price μ S_t E r_f r_fin σ T_t = call_option_solution S_t E r_f r_fin σ T_t := by
  -- 1. Expand definitions
  unfold call_option_price call_option_solution
  rw [h_μ] -- Essential: use the volume measure -- I dont understand really

  -- calc
  --   exp (-r_fin * T_t) * ∫ x, payoff (S_T S_t r_f σ T_t x) E * n x =
  --     exp (-r_fin * T_t) * ∫ x in Ici (-(d2 S_t E r_f σ T_t)), (S_T S_t r_f σ T_t x - E) * n x := by
  --     -- This is the 'indicator' logic we discussed earlier
  --     -- integral_congr_fn

  --   _ = exp (-r_fin * T_t) * (∫ x in Ici (-d2 ...), S_T ... * n x - ∫ x in Ici (-d2 ...), E * n x) := by
  --     -- Linearity of integration
  --     rw [integral_sub]
  --     sorry

  --   _ = S_t * exp ((r_f - r_fin) * T_t) * N (d1 ...) - E * exp (-r_fin * T_t) * N (d2 ...) := by
  --     -- The heavy lifting: Gaussian substitution and using hN_symm
  --     sorry
  -- apply?
  sorry

end BlackScholesFormalization
