"""
estimation_ss_2.py

Simulation-based precision / sample size calculator for external validation
of a CLINICAL PREDICTION MODEL WITH A BINARY OUTCOME (e.g. pCR).

Based on:
Snell KIE, Archer L, Ensor J, Bonnett LJ, Debray TPA, Phillips B, Collins GS,
Riley RD. External validation of clinical prediction models: simulation-based
sample size calculations were more reliable than rules-of-thumb.
J Clin Epidemiol 2021;135:79-89.

Difference from estimation_ss_1.py:
  - ss_1 uses CLOSED-FORM formulas (Fisher information matrix, Newcombe's
    formula) from Riley et al part 3 (BMJ 2023).
  - ss_2 (this script) uses SIMULATION, following Box 2 of Snell et al 2021.
    It assumes the model's linear predictor (LP) is Normal(mu, sigma^2) in
    the validation population, and lets you directly specify miscalibration
    via the calibration model:
        logit(p_i) = gamma + S * LP_i
    where gamma = calibration-in-the-large, S = calibration slope.
    Well-calibrated model: gamma = 0, S = 1.

Two modes:
  A) REVERSE LOOKUP (your main use case): given a fixed N (e.g. your cohort
     size), simulate many datasets and report the achieved average 95% CI
     width for each performance measure.
  B) FORWARD SEARCH: given a target CI width, iteratively increase N until
     the target is met (Box 2, steps 4-9).

Usage: edit the INPUTS block at the bottom and run.
"""

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Core simulation of ONE external validation dataset
# ---------------------------------------------------------------------------
def simulate_one_validation_dataset(n, mu, sigma, gamma, S, rng):
    """
    Simulate one external validation dataset of size n.

    LP_i ~ Normal(mu, sigma^2)               (model's linear predictor)
    logit(p_i) = gamma + S * LP_i            (true calibration model, Eq.2)
    Y_i ~ Bernoulli(p_i)                     (true outcome)

    Returns: LP (model's own predicted linear predictor, used for prediction),
             Y (simulated true outcome)
    """
    LP = rng.normal(mu, sigma, size=n)
    true_logit_p = gamma + S * LP
    true_p = 1 / (1 + np.exp(-true_logit_p))
    true_p = np.clip(true_p, 1e-10, 1 - 1e-10)
    Y = rng.binomial(1, true_p)
    return LP, Y


# ---------------------------------------------------------------------------
# Fit calibration model (logistic regression of Y on LP) -> gamma_hat, S_hat
# ---------------------------------------------------------------------------
def fit_calibration_model(LP, Y, max_iter=50, tol=1e-8):
    """
    Newton-Raphson logistic regression: logit(P(Y=1)) = gamma + S*LP
    Returns gamma_hat, S_hat, and their standard errors (from the
    observed Fisher information / inverse Hessian).
    """
    X = np.column_stack([np.ones_like(LP), LP])
    beta = np.zeros(2)
    for _ in range(max_iter):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        p = np.clip(p, 1e-10, 1 - 1e-10)
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

    # final Hessian for standard errors
    eta = X @ beta
    p = 1 / (1 + np.exp(-eta))
    p = np.clip(p, 1e-10, 1 - 1e-10)
    W = p * (1 - p)
    H = (X * W[:, None]).T @ X
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    se = np.sqrt(np.diag(cov))

    gamma_hat, S_hat = beta[0], beta[1]
    se_gamma, se_S = se[0], se[1]
    return gamma_hat, S_hat, se_gamma, se_S


# ---------------------------------------------------------------------------
# Calibration-in-the-large fixed at S=1 (offset model), per Box 1 definition
# ---------------------------------------------------------------------------
def fit_citl(LP, Y, max_iter=50, tol=1e-8):
    """
    Calibration-in-the-large: fit logit(p_i) = gamma + 1*LP_i (LP as offset).
    Returns gamma_hat (CITL) and its SE.
    """
    gamma = 0.0
    for _ in range(max_iter):
        eta = gamma + LP
        p = 1 / (1 + np.exp(-eta))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        W = p * (1 - p)
        grad = np.sum(Y - p)
        H = np.sum(W)
        step = grad / H if H > 0 else 0.0
        gamma = gamma + step
        if abs(step) < tol:
            break
    eta = gamma + LP
    p = 1 / (1 + np.exp(-eta))
    p = np.clip(p, 1e-10, 1 - 1e-10)
    W = p * (1 - p)
    H = np.sum(W)
    se_gamma = np.sqrt(1 / H) if H > 0 else np.nan
    return gamma, se_gamma


# ---------------------------------------------------------------------------
# C statistic (AUROC) via Mann-Whitney, with DeLong-style SE approximation
# ---------------------------------------------------------------------------
def c_statistic_and_se(p_pred, Y):
    """
    Compute the c statistic (AUROC) using predicted probabilities p_pred
    and true outcomes Y, plus its standard error via a Hanley-McNeil style
    approximation (adequate for simulation-based sample size purposes).
    """
    pos = p_pred[Y == 1]
    neg = p_pred[Y == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return np.nan, np.nan

    # Mann-Whitney U -> c statistic
    # rank-based computation for speed
    all_vals = np.concatenate([pos, neg])
    ranks = stats.rankdata(all_vals)
    rank_pos_sum = ranks[:n1].sum()
    U = rank_pos_sum - n1 * (n1 + 1) / 2
    c = U / (n1 * n0)

    # Hanley-McNeil SE approximation
    q1 = c / (2 - c)
    q2 = 2 * c ** 2 / (1 + c)
    var_c = (c * (1 - c) + (n1 - 1) * (q1 - c ** 2) + (n0 - 1) * (q2 - c ** 2)) / (n1 * n0)
    se_c = np.sqrt(max(var_c, 0))
    return c, se_c


# ---------------------------------------------------------------------------
# O/E ratio and its SE (on log scale, back-transformed)
# ---------------------------------------------------------------------------
def oe_ratio_and_ci(p_pred, Y):
    """
    Observed/Expected ratio, with 95% CI derived on the log scale then
    back-transformed (as done in Snell et al Section 2.2.2).
    """
    observed = np.sum(Y)
    expected = np.sum(p_pred)
    if expected <= 0:
        return np.nan, np.nan, np.nan, np.nan
    oe = observed / expected
    # variance of ln(O/E): approx var(O)/O^2 assuming O ~ Poisson-like
    # standard approach: SE(ln OE) = sqrt(1/O - 1/E) is one simple approx;
    # we use SE(ln OE) ~= sqrt((1-phi_hat)/(N*phi_hat)) as in part 3 Criterion 1
    n = len(Y)
    phi_hat = observed / n
    if phi_hat <= 0 or phi_hat >= 1:
        return oe, np.nan, np.nan, np.nan
    se_ln_oe = np.sqrt((1 - phi_hat) / (n * phi_hat))
    lower = oe * np.exp(-1.96 * se_ln_oe)
    upper = oe * np.exp(1.96 * se_ln_oe)
    return oe, se_ln_oe, lower, upper


# ---------------------------------------------------------------------------
# Net benefit at a chosen threshold
# ---------------------------------------------------------------------------
def net_benefit(p_pred, Y, p_t):
    """
    NB_pt = sensitivity*prevalence - (1-specificity)*(1-prevalence)*pt/(1-pt)
    (Box 1 formula)
    """
    predicted_positive = p_pred >= p_t
    n = len(Y)
    prevalence = np.mean(Y)
    tp = np.sum((predicted_positive) & (Y == 1))
    fn = np.sum((~predicted_positive) & (Y == 1))
    fp = np.sum((predicted_positive) & (Y == 0))
    tn = np.sum((~predicted_positive) & (Y == 0))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    nb = sensitivity * prevalence - (1 - specificity) * (1 - prevalence) * (p_t / (1 - p_t))
    return nb, sensitivity, specificity


# ---------------------------------------------------------------------------
# MAIN REVERSE LOOKUP: given N, simulate many datasets, report average CI widths
# ---------------------------------------------------------------------------
def reverse_lookup_simulation(n, mu, sigma, gamma=0.0, S=1.0,
                               nb_thresholds=(0.1, 0.3),
                               n_sims=500, seed=1, label=""):
    """
    Box 2, steps 5-8, applied at a FIXED sample size n (reverse-lookup mode).

    mu, sigma: assumed Normal(mu, sigma^2) distribution of the model's LP
               in the validation population.
    gamma, S:  TRUE calibration model parameters used to generate outcomes.
               gamma=0, S=1 => model is well calibrated in this population.
               S<1 (e.g. 0.9, 0.8) => model predictions are too extreme
               (overfitting-style miscalibration), matching Snell et al 3.2.2.

    Returns a dict of mean estimates and average 95% CI widths across n_sims
    simulated external validation datasets, mirroring Table 2 of Snell et al.
    """
    rng = np.random.default_rng(seed)

    c_stats, c_ci_widths = [], []
    slope_ests, slope_ci_widths = [], []
    citl_ests, citl_ci_widths = [], []
    oe_ests, oe_ci_widths = [], []
    nb_results = {p_t: {"est": [], "ci_width": []} for p_t in nb_thresholds}
    n_events_list = []

    for _ in range(n_sims):
        LP, Y = simulate_one_validation_dataset(n, mu, sigma, gamma, S, rng)
        p_pred = 1 / (1 + np.exp(-LP))  # model's OWN predicted probability (uses LP directly, S=1,gamma=0 assumed by model)
        n_events_list.append(Y.sum())

        # discrimination
        c, se_c = c_statistic_and_se(p_pred, Y)
        if not np.isnan(c):
            c_stats.append(c)
            c_ci_widths.append(2 * 1.96 * se_c)

        # calibration slope + CITL (from full calibration model fit)
        try:
            g_hat, S_hat, se_g, se_S = fit_calibration_model(LP, Y)
            slope_ests.append(S_hat)
            slope_ci_widths.append(2 * 1.96 * se_S)
        except Exception:
            pass

        # calibration-in-the-large (slope fixed at 1, offset model)
        try:
            g_citl, se_citl = fit_citl(LP, Y)
            citl_ests.append(g_citl)
            citl_ci_widths.append(2 * 1.96 * se_citl)
        except Exception:
            pass

        # O/E
        oe, se_ln_oe, lo, hi = oe_ratio_and_ci(p_pred, Y)
        if not np.isnan(oe):
            oe_ests.append(oe)
            if not np.isnan(hi) and not np.isnan(lo):
                oe_ci_widths.append(hi - lo)

        # net benefit at each threshold
        for p_t in nb_thresholds:
            nb, sens, spec = net_benefit(p_pred, Y, p_t)
            if not np.isnan(nb):
                # approximate SE via normal approx on sens/spec binomial variance
                n1 = Y.sum()
                n0 = n - n1
                if n1 > 0 and n0 > 0 and not np.isnan(sens) and not np.isnan(spec):
                    prevalence = n1 / n
                    var_sens = sens * (1 - sens) / n1
                    var_spec = spec * (1 - spec) / n0
                    w = (p_t / (1 - p_t))
                    var_nb = (prevalence ** 2) * var_sens + ((w * (1 - prevalence)) ** 2) * var_spec
                    se_nb = np.sqrt(var_nb)
                    nb_results[p_t]["est"].append(nb)
                    nb_results[p_t]["ci_width"].append(2 * 1.96 * se_nb)

    results = {
        "n": n,
        "mean_events": np.mean(n_events_list),
        "c_statistic_mean": np.mean(c_stats) if c_stats else np.nan,
        "c_statistic_ci_width": np.mean(c_ci_widths) if c_ci_widths else np.nan,
        "calibration_slope_mean": np.mean(slope_ests) if slope_ests else np.nan,
        "calibration_slope_ci_width": np.mean(slope_ci_widths) if slope_ci_widths else np.nan,
        "citl_mean": np.mean(citl_ests) if citl_ests else np.nan,
        "citl_ci_width": np.mean(citl_ci_widths) if citl_ci_widths else np.nan,
        "oe_mean": np.mean(oe_ests) if oe_ests else np.nan,
        "oe_ci_width": np.mean(oe_ci_widths) if oe_ci_widths else np.nan,
        "net_benefit": {
            p_t: {
                "mean": np.mean(nb_results[p_t]["est"]) if nb_results[p_t]["est"] else np.nan,
                "ci_width": np.mean(nb_results[p_t]["ci_width"]) if nb_results[p_t]["ci_width"] else np.nan,
            }
            for p_t in nb_thresholds
        },
    }

    _print_results(results, label, gamma, S)
    return results


def _print_results(r, label, gamma, S):
    calib_status = "well calibrated (gamma=0, S=1)" if (gamma == 0 and S == 1) \
        else f"MISCALIBRATED (true gamma={gamma:.3f}, true S={S})"
    print(f"\n{'='*72}")
    print(f"N = {r['n']}   {'(' + label + ')' if label else ''}")
    print(f"Assumed calibration in validation population: {calib_status}")
    print(f"Mean simulated events: {r['mean_events']:.1f}")
    print(f"{'='*72}")

    print(f"\n[C-statistic]        mean = {r['c_statistic_mean']:.3f}   "
          f"avg 95% CI width = {r['c_statistic_ci_width']:.3f}   (target <=0.1)")
    print(f"[Calibration slope]  mean = {r['calibration_slope_mean']:.3f}   "
          f"avg 95% CI width = {r['calibration_slope_ci_width']:.3f}   (target <=0.2-0.3)")
    print(f"[Calibration-in-large] mean = {r['citl_mean']:.3f}   "
          f"avg 95% CI width = {r['citl_ci_width']:.3f}")
    print(f"[O/E ratio]          mean = {r['oe_mean']:.3f}   "
          f"avg 95% CI width = {r['oe_ci_width']:.3f}   (target <=0.2 on ln scale)")
    for p_t, d in r["net_benefit"].items():
        print(f"[Net benefit @ {p_t}]  mean = {d['mean']:.4f}   "
              f"avg 95% CI width = {d['ci_width']:.4f}")

    print(f"\n{'-'*72}")
    verdicts = []
    if r["c_statistic_ci_width"] <= 0.1:
        verdicts.append("c statistic: OK")
    elif r["c_statistic_ci_width"] <= 0.15:
        verdicts.append("c statistic: BORDERLINE")
    else:
        verdicts.append("c statistic: IMPRECISE")

    if r["calibration_slope_ci_width"] <= 0.3:
        verdicts.append("Calibration slope: OK (<=0.3)")
    elif r["calibration_slope_ci_width"] <= 0.5:
        verdicts.append("Calibration slope: BORDERLINE")
    else:
        verdicts.append("Calibration slope: IMPRECISE (this is usually the binding constraint)")

    if r["oe_ci_width"] <= 0.2:
        verdicts.append("O/E: OK")
    elif r["oe_ci_width"] <= 0.35:
        verdicts.append("O/E: BORDERLINE")
    else:
        verdicts.append("O/E: IMPRECISE")

    for v in verdicts:
        print(f"  - {v}")


# ---------------------------------------------------------------------------
# FORWARD SEARCH: find minimum N to hit target CI widths (Box 2, full process)
# ---------------------------------------------------------------------------
def forward_search_sample_size(mu, sigma, gamma=0.0, S=1.0,
                                target_c_width=0.1,
                                target_slope_width=0.2,
                                target_oe_width=0.2,
                                n_start=200, n_step=200, n_max=6000,
                                n_sims=300, seed=1, verbose=True):
    """
    Box 2, steps 4-9: iteratively increase N until all target CI widths are met.
    Coarse step search (n_step) then can be refined manually if needed.
    """
    n = n_start
    while n <= n_max:
        r = reverse_lookup_simulation(n, mu, sigma, gamma, S, n_sims=n_sims,
                                       seed=seed, label="forward search")
        ok = (
            r["c_statistic_ci_width"] <= target_c_width
            and r["calibration_slope_ci_width"] <= target_slope_width
            and r["oe_ci_width"] <= target_oe_width
        )
        if ok:
            print(f"\n*** Minimum sample size found: N = {n} "
                  f"(~{r['mean_events']:.0f} events) meets all targets ***")
            return n, r
        n += n_step
    print(f"\nNo sample size up to {n_max} met all targets -- widen search or relax targets.")
    return None, None


# ---------------------------------------------------------------------------
# INPUTS -- edit to match your actual model / cohorts
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Assumed LP distribution in the validation population ---
    # Placeholder values -- REPLACE with your model's actual reported LP
    # distribution (mean, SD) from Cohort 1 development, or approximate it
    # from your reported c statistic / event proportion if unavailable.
    MU = -0.85      # mean of LP (roughly corresponds to base probability ~0.30)
    SIGMA = 1.0     # SD of LP (wider = better discrimination)

    cohorts = {
        "Cohort 2 (internal validation)": 262,
        "Cohort 3 (external validation)": 334,
        "Cohort 4 (external validation)": 162,
    }

    print("\n" + "#" * 72)
    print("# SCENARIO A: assume model is WELL CALIBRATED in each cohort")
    print("#" * 72)
    results_wellcal = {}
    for label, n in cohorts.items():
        results_wellcal[label] = reverse_lookup_simulation(
            n=n, mu=MU, sigma=SIGMA, gamma=0.0, S=1.0,
            nb_thresholds=(0.1, 0.3), n_sims=500, seed=42, label=label
        )

    print("\n" + "#" * 72)
    print("# SCENARIO B: assume mild MISCALIBRATION (true slope S=0.9)")
    print("#" * 72)
    results_miscal = {}
    for label, n in cohorts.items():
        results_miscal[label] = reverse_lookup_simulation(
            n=n, mu=MU, sigma=SIGMA, gamma=0.0, S=0.9,
            nb_thresholds=(0.1, 0.3), n_sims=500, seed=42, label=label
        )

    # --- Summary table ---
    print("\n" + "=" * 90)
    print("SUMMARY: average 95% CI widths, well-calibrated vs mildly miscalibrated (S=0.9)")
    print("=" * 90)
    header = f"{'Cohort':32s} {'c-stat(cal)':>11s} {'c-stat(mis)':>11s} " \
             f"{'slope(cal)':>11s} {'slope(mis)':>11s} {'O/E(cal)':>9s} {'O/E(mis)':>9s}"
    print(header)
    for label in cohorts:
        wc, mc = results_wellcal[label], results_miscal[label]
        print(f"{label:32s} "
              f"{wc['c_statistic_ci_width']:>11.3f} {mc['c_statistic_ci_width']:>11.3f} "
              f"{wc['calibration_slope_ci_width']:>11.3f} {mc['calibration_slope_ci_width']:>11.3f} "
              f"{wc['oe_ci_width']:>9.3f} {mc['oe_ci_width']:>9.3f}")

    # --- Optional: forward search example (commented out by default; slow) ---
    # print("\n" + "#" * 72)
    # print("# FORWARD SEARCH: minimum N for target CI widths (well-calibrated)")
    # print("#" * 72)
    # forward_search_sample_size(
    #     mu=MU, sigma=SIGMA, gamma=0.0, S=1.0,
    #     target_c_width=0.1, target_slope_width=0.2, target_oe_width=0.2,
    #     n_start=300, n_step=300, n_max=4000, n_sims=300, seed=1
    # )
