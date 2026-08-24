"""
compare_auc.py -- Sample size for COMPARING TWO CORRELATED AUCs (paired design).

Jung SH. Sample size calculation for comparing two ROC curves.
Pharmaceutical Statistics. 2024;23(4):557-569.

Asymptotically equivalent to a DeLong test.
  X_k ~ N(delta_k, 1) in cases, Y_k ~ N(0, 1) in controls
  theta_k = Phi(delta_k / sqrt(2))
"""

import numpy as np
from scipy import stats, integrate, optimize


def delta_to_theta(delta):
    return stats.norm.cdf(delta / np.sqrt(2))


def theta_to_delta(theta):
    return np.sqrt(2) * stats.norm.ppf(theta)


def sigma_k_sq(delta_k):
    theta_k = delta_to_theta(delta_k)

    def integrand(x):
        return stats.norm.pdf(x - delta_k) * (stats.norm.cdf(x) ** 2)

    val, _ = integrate.quad(integrand, -12, 12)
    return val - theta_k ** 2


def sigma_12(delta1, delta2, rho):
    theta1 = delta_to_theta(delta1)
    theta2 = delta_to_theta(delta2)
    mvn = stats.multivariate_normal(mean=[delta1, delta2], cov=[[1, rho], [rho, 1]])

    def integrand(x2, x1):
        return stats.norm.cdf(x1) * stats.norm.cdf(x2) * mvn.pdf([x1, x2])

    val, _ = integrate.dblquad(integrand, -8, 8, lambda x1: -8, lambda x1: 8,
                               epsabs=1e-8, epsrel=1e-8)
    return val - theta1 * theta2


def compute_v(delta1, delta2, rho, gamma):
    s1 = sigma_k_sq(delta1)
    s2 = sigma_k_sq(delta2)
    s12 = sigma_12(delta1, delta2, rho)
    v = (s1 + s2 - 2 * s12) / (gamma * (1 - gamma))
    return v, s1, s2, s12


def required_sample_size(theta1, theta2, rho, gamma, alpha=0.05, power=0.8):
    delta1, delta2 = theta_to_delta(theta1), theta_to_delta(theta2)
    v, s1, s2, s12 = compute_v(delta1, delta2, rho, gamma)
    diff = theta1 - theta2
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    N = int(np.ceil(v * (z_alpha + z_beta) ** 2 / (diff ** 2)))
    m = int(np.ceil(gamma * N))
    return {"N": N, "m": m, "n": N - m, "v": v,
            "delta1": delta1, "delta2": delta2, "diff": diff,
            "sigma1_sq": s1, "sigma2_sq": s2, "sigma12": s12,
            "theta1": theta1, "theta2": theta2, "rho": rho, "gamma": gamma,
            "alpha": alpha, "power": power}


def achieved_power(N, theta1, theta2, rho, gamma, alpha=0.05):
    delta1, delta2 = theta_to_delta(theta1), theta_to_delta(theta2)
    v, _, _, _ = compute_v(delta1, delta2, rho, gamma)
    diff = theta1 - theta2
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    power = stats.norm.cdf(np.sqrt(N) * abs(diff) / np.sqrt(v) - z_alpha)
    if power >= 0.8:
        verdict = "OK (>=0.80)"
    elif power >= 0.6:
        verdict = "BORDERLINE (0.60-0.80)"
    else:
        verdict = "LOW (<0.60)"
    return {"N": N, "power": float(power), "v": v, "diff": diff,
            "theta1": theta1, "theta2": theta2, "rho": rho, "gamma": gamma,
            "alpha": alpha, "verdict": verdict}


def min_detectable_difference(N, theta1_ref, rho, gamma, alpha=0.05, power=0.8,
                              search_range=(0.501, 0.999)):
    delta1 = theta_to_delta(theta1_ref)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    def f(theta2):
        delta2 = theta_to_delta(theta2)
        v, _, _, _ = compute_v(delta1, delta2, rho, gamma)
        diff = theta2 - theta1_ref
        if diff <= 0:
            return 1e6
        return v * (z_alpha + z_beta) ** 2 / (diff ** 2) - N

    try:
        theta2_min = optimize.brentq(f, theta1_ref + 1e-4, search_range[1] - 1e-4)
    except Exception:
        theta2_min = np.nan
    return {"N": N, "theta1": theta1_ref, "theta2_min": theta2_min,
            "min_detectable_diff": (theta2_min - theta1_ref) if not np.isnan(theta2_min) else np.nan,
            "rho": rho, "gamma": gamma, "power": power, "alpha": alpha}
