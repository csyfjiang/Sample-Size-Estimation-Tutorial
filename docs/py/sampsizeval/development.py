"""
development.py -- DEVELOPMENT-stage sample size (single binary-outcome model).

Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size required for
developing a clinical prediction model. BMJ 2020;368:m441.

Four criteria (Box 1):
  B1: precise estimate of the overall outcome proportion (intercept)
  B2: small mean absolute prediction error (MAPE), van Smeden et al (P <= 30)
  B3: shrinkage factor S >= 0.9 (usually the binding constraint)
  B4: small optimism (<= 0.05) in apparent R2_Nagelkerke

Every function is pure (returns numbers/dicts, no printing) so it can be
reused by the CLI and the browser (Pyodide) demo.
"""

import numpy as np
from scipy.optimize import brentq


def max_r2cs(phi):
    """Maximum possible Cox-Snell R^2 for a binary outcome with event
    proportion phi (Riley et al 2020, supplementary S5):
        max(R2cs) = 1 - {phi^phi * (1-phi)^(1-phi)}^2
    """
    L0_per_obs = (phi ** phi) * ((1 - phi) ** (1 - phi))
    return 1 - L0_per_obs ** 2


# --- Criterion B1: precision of overall outcome proportion -----------------
def b1_margin_of_error(n, phi):
    return 1.96 * np.sqrt(phi * (1 - phi) / n)


def b1_required_n(phi, target_margin=0.05):
    return (1.96 / target_margin) ** 2 * phi * (1 - phi)


# --- Criterion B2: mean absolute prediction error (MAPE) -------------------
def b2_achieved_mape(n, phi, P):
    phi_use = phi if phi <= 0.5 else 1 - phi
    ln_mape = -0.508 - 0.544 * np.log(n) + 0.259 * np.log(phi_use) + 0.504 * np.log(P)
    return np.exp(ln_mape)


def b2_required_n(phi, P, target_mape=0.05):
    phi_use = phi if phi <= 0.5 else 1 - phi
    ln_n = (-0.508 + 0.259 * np.log(phi_use) + 0.504 * np.log(P) - np.log(target_mape)) / 0.544
    return np.exp(ln_n)


# --- Criterion B3: shrinkage factor ---------------------------------------
def b3_required_n(P, r2cs, target_S=0.9):
    denom = (target_S - 1) * np.log(1 - r2cs / target_S)
    return P / denom


def b3_achieved_shrinkage(n, P, r2cs):
    """Reverse: solve n = P / [(S-1) ln(1 - R2cs/S)] for S in (R2cs, 1)."""
    def f(S):
        if S <= r2cs or S <= 0:
            return np.inf
        denom = (S - 1) * np.log(1 - r2cs / S)
        if denom == 0:
            return np.inf
        return P / denom - n

    lo, hi = max(r2cs + 1e-6, 1e-6), 0.999999
    try:
        f_lo, f_hi = f(lo), f(hi)
        if np.sign(f_lo) == np.sign(f_hi):
            return hi if f_hi > 0 else lo
        return brentq(f, lo, hi, xtol=1e-6)
    except Exception:
        return np.nan


# --- Criterion B4: optimism in apparent R2_Nagelkerke ---------------------
def b4_required_n(P, r2cs, max_r2cs_val, target_delta=0.05):
    S = r2cs / (r2cs + target_delta * max_r2cs_val)
    denom = (S - 1) * np.log(1 - r2cs / S)
    return P / denom, S


def b4_achieved_optimism(n, P, r2cs, max_r2cs_val):
    S_hat = b3_achieved_shrinkage(n, P, r2cs)
    if np.isnan(S_hat) or S_hat <= 0:
        return np.nan, S_hat
    delta = r2cs * (1 - S_hat) / (S_hat * max_r2cs_val)
    return delta, S_hat


def reverse_lookup_development(n, phi, P, r2cs,
                              target_margin=0.05, target_mape=0.05,
                              target_S=0.9, target_optimism=0.05):
    """Reverse-lookup all four criteria at a fixed N. Returns a dict."""
    max_r2 = max_r2cs(phi)

    margin = b1_margin_of_error(n, phi)
    n_b1 = b1_required_n(phi, target_margin)

    if P <= 30:
        mape = b2_achieved_mape(n, phi, P)
        n_b2 = b2_required_n(phi, P, target_mape)
    else:
        mape, n_b2 = np.nan, np.nan

    S_hat = b3_achieved_shrinkage(n, P, r2cs)
    n_b3 = b3_required_n(P, r2cs, target_S)

    delta_hat, _ = b4_achieved_optimism(n, P, r2cs, max_r2)
    n_b4, _ = b4_required_n(P, r2cs, max_r2, target_optimism)

    all_n = [n_b1, n_b3, n_b4] + ([n_b2] if not np.isnan(n_b2) else [])
    n_required = max(all_n)

    binding = {"B1": n_b1, "B2": n_b2, "B3": n_b3, "B4": n_b4}
    binding = {k: v for k, v in binding.items() if not np.isnan(v)}
    worst = max(binding, key=binding.get)

    return {
        "n": n, "phi": phi, "P": P, "r2cs": r2cs,
        "max_r2cs": max_r2, "r2_nagelkerke": r2cs / max_r2,
        "B1_margin": margin, "B1_required_n": n_b1, "B1_ok": margin <= target_margin,
        "B2_mape": mape, "B2_required_n": n_b2,
        "B2_ok": (not np.isnan(mape)) and mape <= target_mape,
        "B3_shrinkage": S_hat, "B3_required_n": n_b3, "B3_ok": S_hat >= target_S,
        "B4_optimism": delta_hat, "B4_required_n": n_b4,
        "B4_ok": (not np.isnan(delta_hat)) and delta_hat <= target_optimism,
        "overall_required_n": n_required,
        "sufficient": n >= n_required,
        "binding_criterion": worst,
        "shortfall": max(0.0, n_required - n),
    }
