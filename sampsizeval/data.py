"""
data.py -- estimate calculator inputs from patient-level data (CSV -> params).

Turns a real dataset into the scalar inputs the four calculators need, so a
user can drag in a CSV instead of guessing phi / c-statistic / rho.

Accepts either a pandas DataFrame or a plain dict of column arrays (the
Pyodide demo passes dicts to avoid a pandas dependency in the browser).
"""

import numpy as np
from scipy import stats


def _col(data, name):
    if hasattr(data, "columns"):          # pandas DataFrame
        return np.asarray(data[name], dtype=float)
    return np.asarray(data[name], dtype=float)  # dict of arrays


def _auc(scores, y):
    pos = scores[y == 1]
    neg = scores[y == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    U = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    return U / (n1 * n0)


def params_from_predictions(data, prob_col="predicted_prob", outcome_col="true_outcome"):
    """For validation_closed / validation_sim.

    Returns phi, c_stat, and mu/sigma of the linear predictor (logit of
    predicted probability), estimated from real predictions.
    """
    p = np.clip(_col(data, prob_col), 1e-6, 1 - 1e-6)
    y = _col(data, outcome_col).astype(int)
    lp = np.log(p / (1 - p))
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "phi": float(y.mean()),
        "c_stat": float(_auc(p, y)),
        "mu": float(lp.mean()),
        "sigma": float(lp.std(ddof=1)),
    }


def params_from_two_models(data, score_a_col="model_A_score",
                           score_b_col="model_B_score", outcome_col="true_outcome"):
    """For compare_auc.

    Returns theta1, theta2 (each model's AUC) and rho (correlation of the two
    scores, averaged across cases and controls, as recommended in the tutorial).
    """
    a = _col(data, score_a_col)
    b = _col(data, score_b_col)
    y = _col(data, outcome_col).astype(int)
    theta1, theta2 = _auc(a, y), _auc(b, y)
    r_case = stats.pearsonr(a[y == 1], b[y == 1])[0] if (y == 1).sum() > 2 else np.nan
    r_ctrl = stats.pearsonr(a[y == 0], b[y == 0])[0] if (y == 0).sum() > 2 else np.nan
    rho = float(np.nanmean([r_case, r_ctrl]))
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "gamma": float(y.mean()),
        "theta1": float(theta1),
        "theta2": float(theta2),
        "rho": rho,
        "rho_cases": float(r_case) if not np.isnan(r_case) else None,
        "rho_controls": float(r_ctrl) if not np.isnan(r_ctrl) else None,
    }


def params_from_outcome(data, outcome_col="true_outcome"):
    """For development-stage: just the event proportion phi."""
    y = _col(data, outcome_col).astype(int)
    return {"n": int(len(y)), "events": int(y.sum()), "phi": float(y.mean())}
