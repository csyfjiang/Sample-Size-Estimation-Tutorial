"""
compare_auc.py -- Sample size for COMPARING TWO CORRELATED AUCs (paired design).

Jung SH. Sample size calculation for comparing two ROC curves.
Pharmaceutical Statistics. 2024;23(4):557-569.

Asymptotically equivalent to a DeLong test.
  X_k ~ N(delta_k, 1) in cases, Y_k ~ N(0, 1) in controls
  theta_k = Phi(delta_k / sqrt(2))
"""

import numpy as np
from scipy import stats, optimize

# Gauss-Hermite nodes/weights, reused for every quadrature call. The integrands
# are expectations of Phi(.) under (bivariate) normal densities, so Gauss-Hermite
# is exact-to-machine-precision with a modest node count and is ~100x faster than
# scipy.integrate.dblquad -- which matters because min_detectable_difference()
# root-finds over repeated sigma_12 evaluations.
_GH_N = 64
_GH_T, _GH_W = np.polynomial.hermite.hermgauss(_GH_N)
_SQRT2 = np.sqrt(2.0)
_SQRTPI = np.sqrt(np.pi)


def delta_to_theta(delta):
    return stats.norm.cdf(delta / np.sqrt(2))


def theta_to_delta(theta):
    return np.sqrt(2) * stats.norm.ppf(theta)


def sigma_k_sq(delta_k):
    """Var{Phi(X_k)} with X_k ~ N(delta_k, 1), via 1D Gauss-Hermite.
        E[Phi(u+delta)^2],  u ~ N(0,1),  u = sqrt(2) t
    """
    theta_k = delta_to_theta(delta_k)
    x = _SQRT2 * _GH_T + delta_k
    val = np.sum(_GH_W * stats.norm.cdf(x) ** 2) / _SQRTPI
    return val - theta_k ** 2


def sigma_12(delta1, delta2, rho):
    """Cov{Phi(X1), Phi(X2)} with (X1,X2) ~ N((delta1,delta2), [[1,rho],[rho,1]]),
    via 2D Gauss-Hermite over the Cholesky factor L=[[1,0],[rho,sqrt(1-rho^2)]].
    """
    theta1 = delta_to_theta(delta1)
    theta2 = delta_to_theta(delta2)
    z1 = _SQRT2 * _GH_T                      # standard-normal grid (1D)
    z2 = _SQRT2 * _GH_T
    Z1, Z2 = np.meshgrid(z1, z2, indexing="ij")
    W = np.outer(_GH_W, _GH_W) / np.pi
    x1 = delta1 + Z1
    x2 = delta2 + rho * Z1 + np.sqrt(max(1.0 - rho ** 2, 0.0)) * Z2
    val = np.sum(W * stats.norm.cdf(x1) * stats.norm.cdf(x2))
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
