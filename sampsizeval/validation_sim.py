"""
validation_sim.py -- EXTERNAL VALIDATION precision via SIMULATION.

Snell KIE, Archer L, Ensor J, et al. External validation of clinical prediction
models: simulation-based sample size calculations were more reliable than
rules-of-thumb. J Clin Epidemiol 2021;135:79-89.

Assumes LP ~ Normal(mu, sigma^2) in the validation population, and a true
calibration model logit(p) = gamma + S * LP (gamma=0, S=1 => well calibrated).
"""

import numpy as np
from scipy import stats


def simulate_one_validation_dataset(n, mu, sigma, gamma, S, rng):
    LP = rng.normal(mu, sigma, size=n)
    true_logit_p = gamma + S * LP
    true_p = np.clip(1 / (1 + np.exp(-true_logit_p)), 1e-10, 1 - 1e-10)
    Y = rng.binomial(1, true_p)
    return LP, Y


def fit_calibration_model(LP, Y, max_iter=50, tol=1e-8):
    X = np.column_stack([np.ones_like(LP), LP])
    beta = np.zeros(2)
    for _ in range(max_iter):
        eta = X @ beta
        p = np.clip(1 / (1 + np.exp(-eta)), 1e-10, 1 - 1e-10)
        W = p * (1 - p)
        grad = X.T @ (Y - p)
        H = (X * W[:, None]).T @ X
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    eta = X @ beta
    p = np.clip(1 / (1 + np.exp(-eta)), 1e-10, 1 - 1e-10)
    W = p * (1 - p)
    H = (X * W[:, None]).T @ X
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    se = np.sqrt(np.diag(cov))
    return beta[0], beta[1], se[0], se[1]


def fit_citl(LP, Y, max_iter=50, tol=1e-8):
    gamma = 0.0
    for _ in range(max_iter):
        p = np.clip(1 / (1 + np.exp(-(gamma + LP))), 1e-10, 1 - 1e-10)
        W = p * (1 - p)
        H = np.sum(W)
        step = np.sum(Y - p) / H if H > 0 else 0.0
        gamma = gamma + step
        if abs(step) < tol:
            break
    p = np.clip(1 / (1 + np.exp(-(gamma + LP))), 1e-10, 1 - 1e-10)
    H = np.sum(p * (1 - p))
    se_gamma = np.sqrt(1 / H) if H > 0 else np.nan
    return gamma, se_gamma


def c_statistic_and_se(p_pred, Y):
    pos = p_pred[Y == 1]
    neg = p_pred[Y == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return np.nan, np.nan
    all_vals = np.concatenate([pos, neg])
    ranks = stats.rankdata(all_vals)
    U = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    c = U / (n1 * n0)
    q1 = c / (2 - c)
    q2 = 2 * c ** 2 / (1 + c)
    var_c = (c * (1 - c) + (n1 - 1) * (q1 - c ** 2) + (n0 - 1) * (q2 - c ** 2)) / (n1 * n0)
    return c, np.sqrt(max(var_c, 0))


def oe_ratio_and_ci(p_pred, Y):
    observed = np.sum(Y)
    expected = np.sum(p_pred)
    if expected <= 0:
        return np.nan, np.nan, np.nan, np.nan
    oe = observed / expected
    n = len(Y)
    phi_hat = observed / n
    if phi_hat <= 0 or phi_hat >= 1:
        return oe, np.nan, np.nan, np.nan
    se_ln_oe = np.sqrt((1 - phi_hat) / (n * phi_hat))
    return oe, se_ln_oe, oe * np.exp(-1.96 * se_ln_oe), oe * np.exp(1.96 * se_ln_oe)


def net_benefit(p_pred, Y, p_t):
    predicted_positive = p_pred >= p_t
    prevalence = np.mean(Y)
    tp = np.sum(predicted_positive & (Y == 1))
    fn = np.sum(~predicted_positive & (Y == 1))
    fp = np.sum(predicted_positive & (Y == 0))
    tn = np.sum(~predicted_positive & (Y == 0))
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    nb = sens * prevalence - (1 - spec) * (1 - prevalence) * (p_t / (1 - p_t))
    return nb, sens, spec


def reverse_lookup_simulation(n, mu, sigma, gamma=0.0, S=1.0,
                              nb_thresholds=(0.1, 0.3), n_sims=500, seed=1):
    """Simulate n_sims datasets at fixed N; report mean estimates + avg CI widths."""
    rng = np.random.default_rng(seed)
    c_stats, c_w = [], []
    slope_e, slope_w = [], []
    citl_e, citl_w = [], []
    oe_e, oe_w = [], []
    nb_res = {p_t: {"est": [], "w": []} for p_t in nb_thresholds}
    n_events = []

    for _ in range(n_sims):
        LP, Y = simulate_one_validation_dataset(n, mu, sigma, gamma, S, rng)
        p_pred = 1 / (1 + np.exp(-LP))
        n_events.append(Y.sum())

        c, se_c = c_statistic_and_se(p_pred, Y)
        if not np.isnan(c):
            c_stats.append(c)
            c_w.append(2 * 1.96 * se_c)
        try:
            _, S_hat, _, se_S = fit_calibration_model(LP, Y)
            slope_e.append(S_hat)
            slope_w.append(2 * 1.96 * se_S)
        except Exception:
            pass
        try:
            g_citl, se_citl = fit_citl(LP, Y)
            citl_e.append(g_citl)
            citl_w.append(2 * 1.96 * se_citl)
        except Exception:
            pass
        oe, _, lo, hi = oe_ratio_and_ci(p_pred, Y)
        if not np.isnan(oe):
            oe_e.append(oe)
            if not np.isnan(hi) and not np.isnan(lo):
                oe_w.append(hi - lo)
        for p_t in nb_thresholds:
            nb, sens, spec = net_benefit(p_pred, Y, p_t)
            n1, n0 = Y.sum(), n - Y.sum()
            if not np.isnan(nb) and n1 > 0 and n0 > 0 and not np.isnan(sens) and not np.isnan(spec):
                prev = n1 / n
                w = p_t / (1 - p_t)
                var_nb = (prev ** 2) * sens * (1 - sens) / n1 + ((w * (1 - prev)) ** 2) * spec * (1 - spec) / n0
                nb_res[p_t]["est"].append(nb)
                nb_res[p_t]["w"].append(2 * 1.96 * np.sqrt(var_nb))

    def m(x):
        return float(np.mean(x)) if x else np.nan

    return {
        "n": n, "mean_events": m(n_events),
        "c_statistic_mean": m(c_stats), "c_statistic_ci_width": m(c_w),
        "calibration_slope_mean": m(slope_e), "calibration_slope_ci_width": m(slope_w),
        "citl_mean": m(citl_e), "citl_ci_width": m(citl_w),
        "oe_mean": m(oe_e), "oe_ci_width": m(oe_w),
        "net_benefit": {p_t: {"mean": m(nb_res[p_t]["est"]), "ci_width": m(nb_res[p_t]["w"])}
                        for p_t in nb_thresholds},
        "gamma": gamma, "S": S,
    }


def forward_search_sample_size(mu, sigma, gamma=0.0, S=1.0,
                               target_c_width=0.1, target_slope_width=0.2,
                               target_oe_width=0.2, n_start=200, n_step=200,
                               n_max=6000, n_sims=300, seed=1):
    n = n_start
    while n <= n_max:
        r = reverse_lookup_simulation(n, mu, sigma, gamma, S, n_sims=n_sims, seed=seed)
        if (r["c_statistic_ci_width"] <= target_c_width
                and r["calibration_slope_ci_width"] <= target_slope_width
                and r["oe_ci_width"] <= target_oe_width):
            return n, r
        n += n_step
    return None, None
