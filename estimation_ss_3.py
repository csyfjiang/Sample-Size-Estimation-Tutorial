"""
estimation_ss_3.py

DEVELOPMENT-stage sample size reverse-lookup for a binary outcome
prediction model (e.g. pCR prediction).

Based on:
Riley RD, Ensor J, Snell KIE, Harrell FE Jr, Martin GP, Reitsma JB,
Moons KGM, Collins G, van Smeden M. Calculating the sample size required
for developing a clinical prediction model. BMJ 2020;368:m441.

This is DIFFERENT from estimation_ss_1.py and estimation_ss_2.py, which
both address EXTERNAL VALIDATION sample size (evaluating an already-built
model in new data). This script addresses the DEVELOPMENT-stage question:
"is Cohort 1 (my training set) big enough to build the model reliably?"

Four criteria for binary outcomes (Box 1 of the paper):
  B1: precisely estimate the overall outcome proportion (intercept)
  B2: target a small mean absolute prediction error (MAPE) -- via van Smeden
      et al's formula, valid for <=30 candidate predictor parameters
  B3: target a shrinkage factor S >= 0.9 (i.e. <=10% expected shrinkage),
      to minimise overfitting -- this is USUALLY the binding constraint
  B4: target small optimism (<=0.05) in apparent R2_Nagelkerke

All four use a REVERSE-LOOKUP mode here: given your actual N (e.g. Cohort 1
= 717), the script reports what each criterion implies is ACHIEVED at that N
(e.g. the achieved shrinkage factor, achieved MAPE, achieved optimism),
rather than only solving forward for the required N.

It also supports the forward direction (solve for minimum N), matching the
pmsampsize package's default behaviour, for cross-checking.
"""

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Helper: max(R2_cs) as a function of outcome proportion (binary outcome)
# Supplementary material S5 of Riley et al 2020 -- based on the null model
# log-likelihood under a Bernoulli outcome.
# ---------------------------------------------------------------------------
def max_r2cs(phi):
    """
    Maximum possible Cox-Snell R^2 for a binary outcome with event
    proportion phi, given a sample size n (cancels out in the standard
    formula since it depends only on phi):
        max(R2cs) = 1 - L0^(2/n) rearranges asymptotically to a function
        of phi alone:  max(R2cs) = 1 - {phi^phi * (1-phi)^(1-phi)}^2
    This matches the reference values reported in the paper (phi=0.5 -> 0.75,
    phi=0.3 -> 0.71, phi=0.2 -> 0.63, phi=0.1 -> 0.48, phi=0.05 -> 0.33).
    """
    L0_per_obs = (phi ** phi) * ((1 - phi) ** (1 - phi))
    return 1 - L0_per_obs ** 2


# ---------------------------------------------------------------------------
# Criterion B1: precision of overall outcome proportion estimate
# ---------------------------------------------------------------------------
def b1_margin_of_error(n, phi):
    """
    Given N and assumed true outcome proportion phi, return the achieved
    margin of error (half-width of 95% CI) around phi.
    Forward solve: n = (1.96/delta)^2 * phi*(1-phi)
    """
    margin = 1.96 * np.sqrt(phi * (1 - phi) / n)
    return margin


def b1_required_n(phi, target_margin=0.05):
    return (1.96 / target_margin) ** 2 * phi * (1 - phi)


# ---------------------------------------------------------------------------
# Criterion B2: Mean Absolute Prediction Error (MAPE), van Smeden et al formula
#   ln(MAPE) = -0.508 - 0.544*ln(n) + 0.259*ln(phi) + 0.504*ln(P)
#   valid for phi <= 0.5 and P <= 30
# ---------------------------------------------------------------------------
def b2_achieved_mape(n, phi, P):
    """
    Given N, outcome proportion phi (<=0.5; if >0.5 use 1-phi), and number
    of candidate predictor parameters P, return the expected MAPE achieved.
    """
    phi_use = phi if phi <= 0.5 else 1 - phi
    ln_mape = -0.508 - 0.544 * np.log(n) + 0.259 * np.log(phi_use) + 0.504 * np.log(P)
    return np.exp(ln_mape)


def b2_required_n(phi, P, target_mape=0.05):
    phi_use = phi if phi <= 0.5 else 1 - phi
    ln_n = (-0.508 + 0.259 * np.log(phi_use) + 0.504 * np.log(P) - np.log(target_mape)) / 0.544
    return np.exp(ln_n)


# ---------------------------------------------------------------------------
# Criterion B3: shrinkage factor S, Riley et al closed-form
#   n = P / [ (S-1) * ln(1 - R2cs/S) ]
# Reverse mode: given n, P, R2cs -> solve for achieved S
# ---------------------------------------------------------------------------
def b3_required_n(P, r2cs, target_S=0.9):
    """Forward: minimum N to achieve shrinkage factor >= target_S."""
    denom = (target_S - 1) * np.log(1 - r2cs / target_S)
    return P / denom


def b3_achieved_shrinkage(n, P, r2cs):
    """
    Reverse lookup: given N, P, and assumed R2cs, solve for the shrinkage
    factor S implied by that sample size. Solve n = P / [(S-1)ln(1-R2cs/S)]
    for S, numerically (S in (R2cs, 1), since need 1 - R2cs/S > 0 i.e. S>R2cs,
    and S<1 typically for the overfitting regime of interest but the
    function is defined for S>R2cs generally).
    """
    def f(S):
        if S <= r2cs or S <= 0:
            return np.inf
        denom = (S - 1) * np.log(1 - r2cs / S)
        if denom == 0:
            return np.inf
        return P / denom - n

    # Search over a plausible range of S, avoiding S=1 (division by zero in ln term at S->1 is fine,
    # but (S-1)->0 too; the function is continuous there in the limit). Use bounds carefully.
    lo, hi = max(r2cs + 1e-6, 1e-6), 0.999999
    try:
        # f is monotonic decreasing in S over this range typically; bracket search
        f_lo, f_hi = f(lo), f(hi)
        if np.sign(f_lo) == np.sign(f_hi):
            # try widening slightly or report boundary
            if f_hi > 0:
                return hi  # shrinkage very close to 1 (excellent, oversized N)
            else:
                return lo  # shrinkage very poor, near R2cs floor
        S_hat = brentq(f, lo, hi, xtol=1e-6)
        return S_hat
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Criterion B4: optimism in apparent R2_Nagelkerke
# ---------------------------------------------------------------------------
def b4_required_n(P, r2cs, max_r2cs_val, target_delta=0.05):
    """Forward: minimum N to achieve expected optimism <= target_delta."""
    S = r2cs / (r2cs + target_delta * max_r2cs_val)
    denom = (S - 1) * np.log(1 - r2cs / S)
    return P / denom, S


def b4_achieved_optimism(n, P, r2cs, max_r2cs_val):
    """
    Reverse lookup: given N, find the shrinkage factor S implied (via B3's
    reverse solve), then back out the implied optimism delta from:
        S = R2cs / (R2cs + delta * max(R2cs))
    =>  delta = R2cs * (1 - S) / (S * max_r2cs_val)
    """
    S_hat = b3_achieved_shrinkage(n, P, r2cs)
    if np.isnan(S_hat) or S_hat <= 0:
        return np.nan, S_hat
    delta = r2cs * (1 - S_hat) / (S_hat * max_r2cs_val)
    return delta, S_hat


# ---------------------------------------------------------------------------
# MASTER: reverse lookup all four criteria at a fixed N
# ---------------------------------------------------------------------------
def reverse_lookup_development(n, phi, P, r2cs, label="",
                                target_margin=0.05, target_mape=0.05,
                                target_S=0.9, target_optimism=0.05):
    max_r2 = max_r2cs(phi)
    print(f"\n{'='*72}")
    print(f"DEVELOPMENT-stage reverse lookup for N = {n}   "
          f"{'(' + label + ')' if label else ''}")
    print(f"Assumed: outcome proportion phi={phi}, candidate predictor "
          f"parameters P={P}, anticipated R2cs={r2cs}")
    print(f"max(R2cs) at this outcome proportion = {max_r2:.3f}   "
          f"(R2_Nagelkerke implied = {r2cs/max_r2:.3f})")
    print(f"{'='*72}")

    # B1
    margin = b1_margin_of_error(n, phi)
    n_b1 = b1_required_n(phi, target_margin)
    print(f"\n[B1] Precision of overall outcome proportion estimate")
    print(f"  Achieved margin of error at N={n}: +/-{margin:.4f}   "
          f"(target <= {target_margin})")
    print(f"  N required to hit target margin: {n_b1:.0f}")
    print(f"  Verdict: {'OK' if margin <= target_margin else 'INSUFFICIENT for B1'}")

    # B2
    if P <= 30 and phi <= 0.99:
        mape = b2_achieved_mape(n, phi, P)
        n_b2 = b2_required_n(phi, P, target_mape)
        print(f"\n[B2] Mean absolute prediction error (MAPE)")
        print(f"  Achieved MAPE at N={n}: {mape:.4f}   (target <= {target_mape})")
        print(f"  N required to hit target MAPE: {n_b2:.0f}")
        print(f"  Verdict: {'OK' if mape <= target_mape else 'INSUFFICIENT for B2'}")
    else:
        mape, n_b2 = np.nan, np.nan
        print(f"\n[B2] Skipped -- formula only valid for P<=30 candidate parameters "
              f"(P={P} given)")

    # B3
    S_hat = b3_achieved_shrinkage(n, P, r2cs)
    n_b3 = b3_required_n(P, r2cs, target_S)
    print(f"\n[B3] Shrinkage factor (overfitting control)")
    print(f"  Achieved shrinkage factor at N={n}: S = {S_hat:.3f}   "
          f"(target >= {target_S}, i.e. <=10% shrinkage)")
    print(f"  N required to hit target shrinkage: {n_b3:.0f}")
    print(f"  Verdict: {'OK' if S_hat >= target_S else 'INSUFFICIENT for B3 (likely the binding constraint)'}")

    # B4
    delta_hat, S_hat_b4 = b4_achieved_optimism(n, P, r2cs, max_r2)
    n_b4, S_b4 = b4_required_n(P, r2cs, max_r2, target_optimism)
    print(f"\n[B4] Optimism in apparent R2_Nagelkerke")
    print(f"  Achieved expected optimism at N={n}: delta = {delta_hat:.4f}   "
          f"(target <= {target_optimism})")
    print(f"  N required to hit target optimism: {n_b4:.0f}")
    print(f"  Verdict: {'OK' if delta_hat <= target_optimism else 'INSUFFICIENT for B4'}")

    print(f"\n{'-'*72}")
    all_n = [n_b1, n_b3, n_b4] + ([n_b2] if not np.isnan(n_b2) else [])
    n_required_overall = max(all_n)
    print(f"Overall minimum N required (max across all criteria): {n_required_overall:.0f}")
    print(f"Your actual N: {n}")
    if n >= n_required_overall:
        print(f"  ==> Your sample size APPEARS SUFFICIENT for all criteria "
              f"under these assumptions.")
    else:
        print(f"  ==> Your sample size is BELOW the requirement "
              f"(short by {n_required_overall - n:.0f} participants) "
              f"under these assumptions.")
        binding = {"B1": n_b1, "B2": n_b2, "B3": n_b3, "B4": n_b4}
        binding = {k: v for k, v in binding.items() if not np.isnan(v)}
        worst = max(binding, key=binding.get)
        print(f"  Binding constraint: {worst} (requires N={binding[worst]:.0f})")

    return {
        "n": n, "phi": phi, "P": P, "r2cs": r2cs,
        "B1_margin": margin, "B1_required_n": n_b1,
        "B2_mape": mape, "B2_required_n": n_b2,
        "B3_shrinkage": S_hat, "B3_required_n": n_b3,
        "B4_optimism": delta_hat, "B4_required_n": n_b4,
        "overall_required_n": n_required_overall,
    }


# ---------------------------------------------------------------------------
# INPUTS -- edit to match your actual Cohort 1 development setting
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Placeholder assumptions -- REPLACE with your real values:
    #   PHI: your actual / anticipated pCR proportion in Cohort 1
    #   P:   number of CANDIDATE predictor parameters considered during
    #        development (not just those in the final model -- this matters
    #        a lot for multimodal fusion models like BC-BioMIXER, where the
    #        effective parameter count for overfitting purposes is a much
    #        harder question than in classic regression; you may need a
    #        conservative proxy, e.g. count of distinct fusion/attention
    #        components tuned against outcome, or consult supplementary
    #        material S5-style guidance in the paper for how to think about
    #        this in non-regression settings)
    #   R2CS: anticipated Cox-Snell R^2 for the model (conservative estimate,
    #        e.g. from pilot results, prior published models in the same
    #        clinical area, or the "15% of max(R2cs)" default rule)
    PHI = 0.30
    P = 20          # <-- placeholder; replace with your actual candidate parameter count
    R2CS_GUESS = 0.15 * max_r2cs(PHI)  # default conservative rule from Fig 4 guidance

    print(f"\nDefault conservative R2cs guess (15% of max(R2cs) at phi={PHI}): "
          f"{R2CS_GUESS:.4f}")

    cohort1_n = 717

    result = reverse_lookup_development(
        n=cohort1_n,
        phi=PHI,
        P=P,
        r2cs=R2CS_GUESS,
        label="Cohort 1 (development)",
    )

    # --- Sensitivity check: repeat with a more optimistic R2cs assumption ---
    print("\n" + "#" * 72)
    print("# SENSITIVITY CHECK: more optimistic R2cs (30% of max(R2cs))")
    print("#" * 72)
    R2CS_OPTIMISTIC = 0.30 * max_r2cs(PHI)
    result_opt = reverse_lookup_development(
        n=cohort1_n,
        phi=PHI,
        P=P,
        r2cs=R2CS_OPTIMISTIC,
        label="Cohort 1 (development, optimistic R2cs)",
    )

    # --- Sensitivity check: fewer candidate parameters ---
    print("\n" + "#" * 72)
    print("# SENSITIVITY CHECK: fewer candidate parameters (P=10)")
    print("#" * 72)
    result_fewer_p = reverse_lookup_development(
        n=cohort1_n,
        phi=PHI,
        P=10,
        r2cs=R2CS_GUESS,
        label="Cohort 1 (development, P=10)",
    )
