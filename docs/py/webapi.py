"""
webapi.py -- thin JSON-friendly wrapper around sampsizeval for the browser
(Pyodide) demo. Every function returns a plain dict of native Python types.
"""

import math
import numpy as np
import sampsizeval as ssv


def _clean(obj):
    """Recursively convert numpy scalars to native types and NaN/inf to None,
    so the result survives JSON / pyodide's toJs conversion."""
    if isinstance(obj, dict):
        return {(_clean(k) if not isinstance(k, (int, float, str)) else k): _clean(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):        # numpy scalar (bool_, float64, int64, ...)
        obj = obj.item()
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj


def development(n, phi, P, r2cs=None):
    if r2cs is None:
        r2cs = 0.15 * ssv.development.max_r2cs(phi)
    r = ssv.development.reverse_lookup_development(int(n), float(phi), int(P), float(r2cs))
    return _clean(r)


def validation_closed(n, phi, c_stat, n_sims=1500):
    r = ssv.validation_closed.reverse_lookup_binary(
        int(n), float(phi), float(c_stat), n_sims=int(n_sims))
    return _clean(r)


def validation_sim(n, mu, sigma, gamma=0.0, S=1.0, n_sims=400):
    r = ssv.validation_sim.reverse_lookup_simulation(
        int(n), float(mu), float(sigma), float(gamma), float(S), n_sims=int(n_sims))
    return _clean(r)


def compare(n, theta1, theta2, rho, gamma, alpha=0.05, power=0.8):
    fwd = ssv.compare_auc.required_sample_size(
        float(theta1), float(theta2), float(rho), float(gamma), float(alpha), float(power))
    out = {"required": _clean(fwd)}
    if n:
        out["achieved"] = _clean(ssv.compare_auc.achieved_power(
            int(n), float(theta1), float(theta2), float(rho), float(gamma), float(alpha)))
        out["mdd"] = _clean(ssv.compare_auc.min_detectable_difference(
            int(n), float(theta1), float(rho), float(gamma), float(alpha), float(power)))
    return out


# --- CSV -> params (called with plain lists from JS) -----------------------
def params_from_predictions(prob, outcome):
    return _clean(ssv.data.params_from_predictions(
        {"predicted_prob": prob, "true_outcome": outcome}))


def params_from_two_models(score_a, score_b, outcome):
    return _clean(ssv.data.params_from_two_models(
        {"model_A_score": score_a, "model_B_score": score_b, "true_outcome": outcome}))
