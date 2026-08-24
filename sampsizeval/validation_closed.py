"""
validation_closed.py -- EXTERNAL VALIDATION precision via CLOSED-FORM formulas.

Riley RD, Snell KIE, Archer L, et al. Evaluation of clinical prediction models
(part 3): calculating the sample size required for an external validation
study. BMJ 2023;383:e074821.

Reverse mode: given N (your cohort size) -> achieved 95% CI width for
  1. O/E ratio            (closed form)
  2. calibration slope    (simulation approximation of the Fisher-information)
  3. c statistic          (Newcombe's formula, closed form)
  4. standardised net benefit (Marsh et al formula)
"""

import numpy as np
from scipy import stats


# --- Criterion 1: O/E ratio -----------------------------------------------
def oe_ci_width(n, phi):
    se_ln_oe = np.sqrt((1 - phi) / (n * phi))
    lower = np.exp(-1.96 * se_ln_oe)
    upper = np.exp(1.96 * se_ln_oe)
    return se_ln_oe, lower, upper, (upper - lower)


# --- Criterion 2: calibration slope (simulation) --------------------------
def _fit_calibration_slope(lp, y):
    X = np.column_stack([np.ones_like(lp), lp])
    beta = np.zeros(2)
    for _ in range(50):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-8, None)
        grad = X.T @ (y - p)
        H = (X * W[:, None]).T @ X
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta[1]


def calibration_slope_ci_width(n, phi, c_stat=None, beta_params=None,
                               n_sims=2000, seed=1):
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_sims):
        if beta_params is not None:
            a, b = beta_params
            p = np.clip(rng.beta(a, b, size=n), 1e-6, 1 - 1e-6)
            lp = np.log(p / (1 - p))
        elif c_stat is not None:
            delta = stats.norm.ppf(c_stat) * np.sqrt(2)
            mu0, mu1, s = -delta / 2, delta / 2, 1.0
            y = rng.binomial(1, phi, size=n)
            lp = np.where(y == 1, rng.normal(mu1, s, size=n), rng.normal(mu0, s, size=n))
        else:
            raise ValueError("Must supply either beta_params or c_stat")
        true_p = 1 / (1 + np.exp(-lp))
        y_sim = rng.binomial(1, true_p)
        try:
            slopes.append(_fit_calibration_slope(lp, y_sim))
        except Exception:
            continue
    slopes = np.array(slopes)
    se = np.std(slopes, ddof=1)
    return se, 2 * 1.96 * se, slopes.mean(), len(slopes)


# --- Criterion 3: c statistic (Newcombe) ----------------------------------
def c_stat_ci_width(n, c_stat, phi):
    C = c_stat
    term1 = C * (1 - C)
    term2 = 1 + (n / 2 - 1) * (1 - C) / (2 - C)
    term3 = (n / 2 - 1) * C / (1 + C)
    numerator = term1 * (term2 + term3)
    denominator = (n ** 2) * phi * (1 - phi)
    se_c = np.sqrt(numerator / denominator)
    return se_c, 2 * 1.96 * se_c


# --- Criterion 4: standardised net benefit --------------------------------
def net_benefit_ci_width(n, phi, sensitivity, specificity, p_t):
    w = ((1 - phi) / phi) * (p_t / (1 - p_t))
    var_component = (
        sensitivity * (1 - sensitivity) / phi
        + (w ** 2) * specificity * (1 - specificity) / (1 - phi)
        + (w ** 2) * (1 - specificity) ** 2 / (phi * (1 - phi))
    )
    se_snb = np.sqrt(var_component / n) / phi
    return se_snb, 2 * 1.96 * se_snb


def sens_spec_from_c_and_threshold(c_stat, p_t, phi):
    delta = stats.norm.ppf(c_stat) * np.sqrt(2)
    mu0, mu1, s = -delta / 2, delta / 2, 1.0
    lp_t = np.log(p_t / (1 - p_t))
    sensitivity = 1 - stats.norm.cdf(lp_t, loc=mu1, scale=s)
    specificity = stats.norm.cdf(lp_t, loc=mu0, scale=s)
    return sensitivity, specificity


def reverse_lookup_binary(n, phi, c_stat, beta_params=None,
                          thresholds=(0.1, 0.3), n_sims=2000, seed=1):
    """Reverse-lookup achieved precision on all four criteria. Returns a dict."""
    se_oe, lo, hi, w_oe = oe_ci_width(n, phi)
    se_slope, w_slope, mean_slope, n_ok = calibration_slope_ci_width(
        n, phi, c_stat=c_stat if beta_params is None else None,
        beta_params=beta_params, n_sims=n_sims, seed=seed)
    se_c, w_c = c_stat_ci_width(n, c_stat, phi)

    nb = {}
    for p_t in thresholds:
        sens, spec = sens_spec_from_c_and_threshold(c_stat, p_t, phi)
        se_snb, w_snb = net_benefit_ci_width(n, phi, sens, spec, p_t)
        nb[p_t] = {"sens": sens, "spec": spec, "se": se_snb, "ci_width": w_snb}

    return {
        "n": n, "phi": phi, "c_stat": c_stat,
        "oe_ci_width": w_oe, "oe_lower": lo, "oe_upper": hi,
        "oe_ok": w_oe <= 0.22,
        "calibration_slope_ci_width": w_slope, "calibration_slope_mean": mean_slope,
        "calibration_slope_ok": w_slope <= 0.3,
        "c_stat_ci_width": w_c, "c_stat_ok": w_c <= 0.1,
        "net_benefit": nb,
    }
