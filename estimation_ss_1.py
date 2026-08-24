"""
estimation_ss_1.py

Reverse-lookup precision calculator for external validation of a
CLINICAL PREDICTION MODEL WITH A BINARY OUTCOME (e.g. pCR).

Based on:
Riley RD, Snell KIE, Archer L, et al. Evaluation of clinical prediction
models (part 3): calculating the sample size required for an external
validation study. BMJ 2023;383:e074821.

This script does the REVERSE of pmvalsampsize's usual mode:
  - pmvalsampsize (forward mode):  target CI width  ->  required N
  - estimation_ss_1 (reverse mode): given N (your cohort size)  ->  achieved CI width

Four criteria implemented for binary outcomes:
  1. Observed/Expected (O/E) ratio precision
  2. Calibration slope precision (requires linear predictor distribution ->
     approximated here via simulation from an assumed beta distribution of
     predicted probabilities, converted to the logit / linear predictor scale)
  3. c statistic precision (Newcombe's formula, closed form)
  4. Standardised net benefit precision (at a chosen probability threshold)

Usage: edit the INPUTS block at the bottom and run.
"""

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Criterion 1: Observed/Expected (O/E) ratio precision
# ---------------------------------------------------------------------------
def oe_ci_width(n, phi):
    """
    Given a sample size n and assumed true outcome event proportion phi,
    return the achieved 95% CI width on the O/E ratio scale, assuming the
    model is well calibrated (true O/E = 1).

    SE(ln(O/E)) = sqrt((1 - phi) / (n * phi))
    CI on O/E scale approximated via exponentiating +/-1.96*SE around ln(1)=0.
    """
    se_ln_oe = np.sqrt((1 - phi) / (n * phi))
    lower = np.exp(-1.96 * se_ln_oe)
    upper = np.exp(1.96 * se_ln_oe)
    return se_ln_oe, lower, upper, (upper - lower)


# ---------------------------------------------------------------------------
# Criterion 2: Calibration slope precision (simulation-based approximation)
# ---------------------------------------------------------------------------
def calibration_slope_ci_width(n, phi, c_stat=None, beta_params=None,
                                n_sims=2000, seed=1):
    """
    Approximate the achieved SE / 95% CI width of the calibration slope
    for a given sample size n, via simulation, assuming the model is truly
    well calibrated (true intercept alpha=0, true slope beta=1).

    Requires an assumed distribution of the linear predictor (LP).
    Two ways to supply this:
      (a) beta_params = (a, b): assumed Beta(a, b) distribution of PREDICTED
          PROBABILITIES (matches the paper's covid-19 example, fig 4), which
          is converted internally to the logit (linear predictor) scale.
      (b) c_stat: if no distribution is known, approximate LP as normal with
          equal variance in outcome/non-outcome groups and mean difference
          set to reproduce the assumed c statistic (last-resort approach
          flagged in the paper -- Box 1).

    This is a simulation approximation of the closed-form Fisher-information
    approach used by pmvalsampsize; it is adequate for reverse-lookup /
    sanity-check purposes but you should confirm with the official
    pmvalsampsize package for anything going into a manuscript.
    """
    rng = np.random.default_rng(seed)
    slopes = []

    for _ in range(n_sims):
        if beta_params is not None:
            a, b = beta_params
            p = rng.beta(a, b, size=n)
            p = np.clip(p, 1e-6, 1 - 1e-6)
            lp = np.log(p / (1 - p))  # logit transform -> linear predictor
        elif c_stat is not None:
            # crude normal approximation: LP | event ~ N(mu1, s2), LP | no event ~ N(mu0, s2)
            # choose mu1 - mu0 to approximately reproduce c_stat under normal-normal AUC formula
            # AUC = Phi( (mu1 - mu0) / (s*sqrt(2)) )
            delta = stats.norm.ppf(c_stat) * np.sqrt(2)
            mu0, mu1, s = -delta / 2, delta / 2, 1.0
            y = rng.binomial(1, phi, size=n)
            lp = np.where(y == 1, rng.normal(mu1, s, size=n), rng.normal(mu0, s, size=n))
        else:
            raise ValueError("Must supply either beta_params or c_stat")

        # simulate outcomes from TRUE model (alpha=0, beta=1) applied to lp
        true_p = 1 / (1 + np.exp(-lp))
        y_sim = rng.binomial(1, true_p)

        # fit logistic regression: y_sim ~ alpha_hat + beta_hat * lp
        # (equivalent to Cox recalibration model in the paper)
        try:
            beta_hat = _fit_calibration_slope(lp, y_sim)
            slopes.append(beta_hat)
        except Exception:
            continue

    slopes = np.array(slopes)
    se = np.std(slopes, ddof=1)
    ci_width = 2 * 1.96 * se
    return se, ci_width, slopes.mean(), len(slopes)


def _fit_calibration_slope(lp, y):
    """Simple Newton-Raphson logistic regression of y on lp (1 predictor + intercept)."""
    X = np.column_stack([np.ones_like(lp), lp])
    beta = np.zeros(2)
    for _ in range(50):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        W = np.clip(W, 1e-8, None)
        grad = X.T @ (y - p)
        H = (X * W[:, None]).T @ X
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta[1]  # slope coefficient


# ---------------------------------------------------------------------------
# Criterion 3: c statistic precision (Newcombe's formula, closed form)
# ---------------------------------------------------------------------------
def c_stat_ci_width(n, c_stat, phi):
    """
    Newcombe's formula for the standard error of the c statistic.
    Returns SE(C) and the approximate 95% CI width.
    """
    C = c_stat
    term1 = C * (1 - C)
    term2 = 1 + (n / 2 - 1) * (1 - C) / (2 - C)
    term3 = (n / 2 - 1) * C / (1 + C)
    numerator = term1 * (term2 + term3)
    denominator = (n ** 2) * phi * (1 - phi)
    se_c = np.sqrt(numerator / denominator)
    ci_width = 2 * 1.96 * se_c
    return se_c, ci_width


# ---------------------------------------------------------------------------
# Criterion 4: Standardised net benefit precision (Marsh et al formula)
# ---------------------------------------------------------------------------
def net_benefit_ci_width(n, phi, sensitivity, specificity, p_t):
    """
    Achieved SE / 95% CI width of the standardised net benefit at
    probability threshold p_t, given assumed sensitivity/specificity at
    that threshold (inferred from the assumed calibration/discrimination).
    """
    w = ((1 - phi) / phi) * (p_t / (1 - p_t))
    var_component = (
        sensitivity * (1 - sensitivity) / phi
        + (w ** 2) * specificity * (1 - specificity) / (1 - phi)
        + (w ** 2) * (1 - specificity) ** 2 / (phi * (1 - phi))
    )
    se_snb = np.sqrt(var_component / n) / phi  # standardised: divide by phi as in NBpt/phi
    ci_width = 2 * 1.96 * se_snb
    return se_snb, ci_width


def sens_spec_from_c_and_threshold(c_stat, p_t, phi):
    """
    Rough approximation of sensitivity/specificity at threshold p_t,
    inferring an underlying normal linear-predictor separation from c_stat
    (same 'last resort' approximation flagged in Box 1 of the paper).
    Use only as an approximation; prefer real data if available.
    """
    delta = stats.norm.ppf(c_stat) * np.sqrt(2)
    mu0, mu1, s = -delta / 2, delta / 2, 1.0
    lp_t = np.log(p_t / (1 - p_t))
    sensitivity = 1 - stats.norm.cdf(lp_t, loc=mu1, scale=s)
    specificity = stats.norm.cdf(lp_t, loc=mu0, scale=s)
    return sensitivity, specificity


# ---------------------------------------------------------------------------
# Master reverse-lookup function: given N, report achieved precision on all 4 criteria
# ---------------------------------------------------------------------------
def reverse_lookup_binary(n, phi, c_stat, beta_params=None,
                           thresholds=(0.1, 0.3), n_sims=2000, seed=1,
                           label=""):
    print(f"\n{'='*70}")
    print(f"Reverse lookup for N = {n}   {'(' + label + ')' if label else ''}")
    print(f"Assumed: outcome proportion phi={phi}, c statistic={c_stat}")
    print(f"{'='*70}")

    # Criterion 1: O/E
    se_oe, lo, hi, w_oe = oe_ci_width(n, phi)
    print(f"\n[Criterion 1] O/E ratio")
    print(f"  SE(ln O/E) = {se_oe:.4f}")
    print(f"  95% CI on O/E scale (assuming true O/E=1): ({lo:.3f}, {hi:.3f}), width = {w_oe:.3f}")
    print(f"  Reference target in paper: width <= 0.22 (covid example)")

    # Criterion 2: calibration slope (simulation-based)
    se_slope, w_slope, mean_slope, n_ok = calibration_slope_ci_width(
        n, phi, c_stat=c_stat if beta_params is None else None,
        beta_params=beta_params, n_sims=n_sims, seed=seed
    )
    print(f"\n[Criterion 2] Calibration slope (simulation, {n_ok} valid sims)")
    print(f"  SE(slope) ~= {se_slope:.4f}")
    print(f"  95% CI width ~= {w_slope:.3f}  (mean simulated slope = {mean_slope:.3f}, true = 1.0)")
    print(f"  Reference targets in paper: width <= 0.3 (loose) or <= 0.2 (tight)")

    # Criterion 3: c statistic
    se_c, w_c = c_stat_ci_width(n, c_stat, phi)
    print(f"\n[Criterion 3] c statistic")
    print(f"  SE(C) = {se_c:.4f}")
    print(f"  95% CI width = {w_c:.3f}")
    print(f"  Reference target in paper: width <= 0.1 (SE <= 0.0255)")

    # Criterion 4: standardised net benefit at each threshold
    print(f"\n[Criterion 4] Standardised net benefit")
    for p_t in thresholds:
        sens, spec = sens_spec_from_c_and_threshold(c_stat, p_t, phi)
        se_snb, w_snb = net_benefit_ci_width(n, phi, sens, spec, p_t)
        print(f"  Threshold p_t={p_t}: sens~={sens:.3f}, spec~={spec:.3f}, "
              f"SE(sNB)~={se_snb:.4f}, 95% CI width~={w_snb:.3f}")

    print(f"\n{'-'*70}")
    print("Summary verdict:")
    verdicts = []
    if w_oe <= 0.22:
        verdicts.append("O/E: OK")
    else:
        verdicts.append("O/E: WIDE (imprecise)")
    if w_slope <= 0.3:
        verdicts.append("Calibration slope: OK (<=0.3)")
    elif w_slope <= 0.5:
        verdicts.append("Calibration slope: BORDERLINE (0.3-0.5)")
    else:
        verdicts.append("Calibration slope: WIDE (imprecise)")
    if w_c <= 0.1:
        verdicts.append("c statistic: OK (<=0.1)")
    elif w_c <= 0.15:
        verdicts.append("c statistic: BORDERLINE (0.1-0.15)")
    else:
        verdicts.append("c statistic: WIDE (imprecise)")
    for v in verdicts:
        print(f"  - {v}")

    return {
        "n": n, "phi": phi, "c_stat": c_stat,
        "oe_ci_width": w_oe,
        "calibration_slope_ci_width": w_slope,
        "c_stat_ci_width": w_c,
    }


# ---------------------------------------------------------------------------
# INPUTS -- edit these to match your actual cohorts
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Placeholder assumptions -- REPLACE with your real Cohort 1 development
    # results (optimism-adjusted c statistic, actual pCR proportion).
    ASSUMED_PHI = 0.30       # assumed / observed pCR event proportion
    ASSUMED_C = 0.75         # assumed c statistic (from development / internal validation)
    BETA_PARAMS = (1.5, 2.5) # assumed shape of predicted probability distribution
                              # (tune to match your model's actual predicted-risk histogram;
                              #  see fig 4 of part 3 for how Gupta et al approximated theirs)

    cohorts = {
        "Cohort 2 (internal validation)": 262,
        "Cohort 3 (external validation)": 334,
        "Cohort 4 (external validation)": 162,
    }

    results = {}
    for label, n in cohorts.items():
        results[label] = reverse_lookup_binary(
            n=n,
            phi=ASSUMED_PHI,
            c_stat=ASSUMED_C,
            beta_params=BETA_PARAMS,
            thresholds=(0.1, 0.3),
            n_sims=2000,
            seed=42,
            label=label,
        )

    print(f"\n{'='*70}")
    print("ALL COHORTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Cohort':35s} {'O/E width':>10s} {'Slope width':>12s} {'c-stat width':>13s}")
    for label, r in results.items():
        print(f"{label:35s} {r['oe_ci_width']:>10.3f} "
              f"{r['calibration_slope_ci_width']:>12.3f} {r['c_stat_ci_width']:>13.3f}")
