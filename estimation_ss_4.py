"""
estimation_ss_4.py

Sample size calculation for COMPARING TWO CORRELATED AUCs (head-to-head
model comparison, e.g. BC-BioMIXER vs a baseline/competitor model on the
same patients).

Based on:
Jung SH. Sample size calculation for comparing two ROC curves.
Pharmaceutical Statistics. 2024;23(4):557-569.

This is a DIFFERENT question from estimation_ss_1/2/3:
  - ss_1, ss_2: precision of a SINGLE model's external validation performance
  - ss_3: development-stage sample size for a SINGLE model
  - ss_4 (this script): sample size to detect a DIFFERENCE in AUC between
    TWO models evaluated on the SAME patients (paired / correlated design),
    asymptotically equivalent to a DeLong test.

Core idea (log-transform + standardization trick from the paper):
  After a monotone transformation (e.g. log) that maps each biomarker's
  case/control distributions to approximately Normal with equal variance,
  we can write:
      X_k ~ N(delta_k, 1)   (biomarker k, case/event group)
      Y_k ~ N(0, 1)         (biomarker k, control/non-event group)
  where delta_k is a standardized effect size that maps 1:1 to AUC theta_k:
      theta_k = P(X_k > Y_k) = integral phi(t - delta_k) * Phi(t) dt

  corr(X1, X2) = corr(Y1, Y2) = rho is assumed common across case/control.

Two modes:
  A) FORWARD: given target AUCs (theta1, theta2), rho, prevalence gamma,
     alpha, power -> compute required N (paper's Section 3 formula).
  B) REVERSE: given a fixed N (e.g. your validation cohort size) -> compute
     the achieved power to detect a given AUC difference, or the minimum
     detectable difference at a target power.
"""

import numpy as np
from scipy import stats, integrate, optimize


# ---------------------------------------------------------------------------
# Convert between standardized effect size delta and AUC theta
#   theta = P(X > Y) where X ~ N(delta, 1), Y ~ N(0, 1)
#         = Phi(delta / sqrt(2))
# (This is the standard normal-normal AUC formula, consistent with Table 1
# of the paper -- delta=1.0 -> theta=0.760, delta=1.5 -> theta=0.856, etc.)
# ---------------------------------------------------------------------------
def delta_to_theta(delta):
    return stats.norm.cdf(delta / np.sqrt(2))


def theta_to_delta(theta):
    return np.sqrt(2) * stats.norm.ppf(theta)


# ---------------------------------------------------------------------------
# sigma_k^2(epsilon) = Var{G_k(X_k)} = integral phi(x - delta_k) * Phi(x)^2 dx - theta_k^2
# (Table 1 of the paper; computed here by numerical integration for any delta)
# ---------------------------------------------------------------------------
def sigma_k_sq(delta_k):
    theta_k = delta_to_theta(delta_k)

    def integrand(x):
        return stats.norm.pdf(x - delta_k) * (stats.norm.cdf(x) ** 2)

    val, _ = integrate.quad(integrand, -12, 12)
    return val - theta_k ** 2


# ---------------------------------------------------------------------------
# sigma_12(epsilon) = Cov{G1(X1), G2(X2)}
#   = double integral Phi(x1)*Phi(x2)*phi_rho(x1-delta1, x2-delta2) dx1 dx2 - theta1*theta2
# computed via 2D numerical integration of the bivariate normal density.
# ---------------------------------------------------------------------------
def sigma_12(delta1, delta2, rho):
    theta1 = delta_to_theta(delta1)
    theta2 = delta_to_theta(delta2)

    mean = [delta1, delta2]
    cov = [[1, rho], [rho, 1]]
    mvn = stats.multivariate_normal(mean=mean, cov=cov)

    def integrand(x2, x1):
        return stats.norm.cdf(x1) * stats.norm.cdf(x2) * mvn.pdf([x1, x2])

    val, _ = integrate.dblquad(
        integrand, -8, 8, lambda x1: -8, lambda x1: 8, epsabs=1e-8, epsrel=1e-8
    )
    return val - theta1 * theta2


# ---------------------------------------------------------------------------
# v = (gamma * (1-gamma))^{-1} * [sigma1^2 + sigma2^2 - 2*sigma12]   (Eq. 6)
# ---------------------------------------------------------------------------
def compute_v(delta1, delta2, rho, gamma):
    s1 = sigma_k_sq(delta1)
    s2 = sigma_k_sq(delta2)
    s12 = sigma_12(delta1, delta2, rho)
    v = (s1 + s2 - 2 * s12) / (gamma * (1 - gamma))
    return v, s1, s2, s12


# ---------------------------------------------------------------------------
# FORWARD: required sample size (Eq. 5)
#   N = v * (z_{1-alpha/2} + z_{1-beta})^2 / Delta^2
# ---------------------------------------------------------------------------
def required_sample_size(theta1, theta2, rho, gamma, alpha=0.05, power=0.8):
    delta1 = theta_to_delta(theta1)
    delta2 = theta_to_delta(theta2)
    v, s1, s2, s12 = compute_v(delta1, delta2, rho, gamma)

    diff = theta1 - theta2
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    N = v * (z_alpha + z_beta) ** 2 / (diff ** 2)
    N = int(np.ceil(N))
    m = int(np.ceil(gamma * N))   # cases
    n = N - m                      # controls

    print(f"\n{'='*70}")
    print(f"FORWARD: required sample size to detect AUC difference")
    print(f"{'='*70}")
    print(f"theta1 = {theta1} (delta1 = {delta1:.3f}), "
          f"theta2 = {theta2} (delta2 = {delta2:.3f})")
    print(f"Delta (theta1 - theta2) = {diff:.4f}")
    print(f"rho (correlation between the two models' scores) = {rho}")
    print(f"gamma (case/event prevalence) = {gamma}")
    print(f"alpha = {alpha} (2-sided), target power = {power}")
    print(f"sigma1^2(eps) = {s1:.5f}, sigma2^2(eps) = {s2:.5f}, "
          f"sigma12(eps) = {s12:.5f}")
    print(f"v = {v:.4f}")
    print(f"\nRequired total N = {N}  (cases m = {m}, controls n = {n})")
    return {"N": N, "m": m, "n": n, "v": v, "delta1": delta1, "delta2": delta2}


# ---------------------------------------------------------------------------
# REVERSE: achieved power at a fixed N (your actual cohort size)
# ---------------------------------------------------------------------------
def achieved_power(N, theta1, theta2, rho, gamma, alpha=0.05):
    delta1 = theta_to_delta(theta1)
    delta2 = theta_to_delta(theta2)
    v, s1, s2, s12 = compute_v(delta1, delta2, rho, gamma)

    diff = theta1 - theta2
    z_alpha = stats.norm.ppf(1 - alpha / 2)

    power = stats.norm.cdf(np.sqrt(N) * abs(diff) / np.sqrt(v) - z_alpha)

    print(f"\n{'='*70}")
    print(f"REVERSE: achieved power at fixed N = {N}")
    print(f"{'='*70}")
    print(f"theta1 = {theta1}, theta2 = {theta2}, Delta = {diff:.4f}")
    print(f"rho = {rho}, gamma = {gamma}, alpha = {alpha} (2-sided)")
    print(f"v = {v:.4f}")
    print(f"Achieved power = {power:.3f}")

    if power >= 0.8:
        verdict = "OK (>=0.80, conventionally adequate)"
    elif power >= 0.6:
        verdict = "BORDERLINE (0.60-0.80, likely underpowered)"
    else:
        verdict = "LOW (<0.60, substantially underpowered)"
    print(f"Verdict: {verdict}")
    return power


# ---------------------------------------------------------------------------
# REVERSE: minimum detectable AUC difference at a fixed N and target power
# ---------------------------------------------------------------------------
def min_detectable_difference(N, theta1_ref, rho, gamma, alpha=0.05, power=0.8,
                                search_range=(0.501, 0.999)):
    """
    Given a fixed N and a reference AUC theta1_ref (e.g. your baseline
    model's AUC), find the minimum theta2 (or theta1, whichever direction)
    such that the difference |theta1_ref - theta2| is detectable at the
    target power with this N. Searches theta2 > theta1_ref.
    """
    delta1 = theta_to_delta(theta1_ref)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    def f(theta2):
        delta2 = theta_to_delta(theta2)
        v, _, _, _ = compute_v(delta1, delta2, rho, gamma)
        diff = theta2 - theta1_ref
        if diff <= 0:
            return 1e6
        # required N for this diff, compare to actual N
        N_req = v * (z_alpha + z_beta) ** 2 / (diff ** 2)
        return N_req - N

    try:
        theta2_min = optimize.brentq(f, theta1_ref + 1e-4, search_range[1] - 1e-4)
    except Exception:
        theta2_min = np.nan

    print(f"\n{'='*70}")
    print(f"REVERSE: minimum detectable AUC at fixed N = {N}")
    print(f"{'='*70}")
    print(f"Reference AUC (theta1) = {theta1_ref}, rho = {rho}, gamma = {gamma}")
    print(f"Target power = {power}, alpha = {alpha} (2-sided)")
    if not np.isnan(theta2_min):
        print(f"Minimum detectable theta2 = {theta2_min:.4f}  "
              f"(minimum detectable Delta = {theta2_min - theta1_ref:.4f})")
    else:
        print("Could not find a solution in the search range.")
    return theta2_min


# ---------------------------------------------------------------------------
# INPUTS -- edit to match your actual comparison
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Placeholder assumptions -- REPLACE with your real values ---
    # THETA1: baseline/competitor model's AUC (e.g. clinical-only model,
    #         or a prior BioMIXER version)
    # THETA2: BC-BioMIXER's AUC (the model you expect to be better)
    # RHO:    correlation between the two models' predicted scores on the
    #         same patients (higher when both models share strong overlapping
    #         signal, e.g. both use similar clinical variables)
    # GAMMA:  pCR event proportion in the comparison cohort
    THETA1 = 0.75   # xxx: baseline model AUC
    THETA2 = 0.82   # xxx: BC-BioMIXER AUC
    RHO = 0.5       # xxx: correlation between the two models' scores
    GAMMA = 0.30    # xxx: pCR event proportion

    # --- A) Forward: how many patients would a prospective validation need? ---
    required_sample_size(THETA1, THETA2, RHO, GAMMA, alpha=0.05, power=0.8)

    # --- B) Reverse: with your ACTUAL cohort sizes, what power do you have? ---
    cohorts = {
        "Cohort 2": 262,
        "Cohort 3": 334,
        "Cohort 4": 162,
    }
    for label, n in cohorts.items():
        print(f"\n>>> {label} (N={n}) <<<")
        achieved_power(n, THETA1, THETA2, RHO, GAMMA, alpha=0.05)

    # --- C) Reverse: what's the smallest AUC gap you could detect with Cohort 3? ---
    min_detectable_difference(334, THETA1, RHO, GAMMA, alpha=0.05, power=0.8)
