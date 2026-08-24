"""Smoke / sanity tests for sampsizeval. Run: pytest -q"""

import numpy as np
import pytest

import sampsizeval as ssv


def test_max_r2cs_reference_values():
    # Reference values reported in Riley et al 2020
    assert ssv.development.max_r2cs(0.5) == pytest.approx(0.75, abs=0.01)
    assert ssv.development.max_r2cs(0.3) == pytest.approx(0.71, abs=0.01)
    assert ssv.development.max_r2cs(0.1) == pytest.approx(0.48, abs=0.01)


def test_development_shrinkage_binding():
    r = ssv.development.reverse_lookup_development(
        n=717, phi=0.30, P=20, r2cs=0.15 * ssv.development.max_r2cs(0.30))
    assert r["binding_criterion"] == "B3"        # shrinkage usually binds
    assert 0 < r["B3_shrinkage"] < 1


def test_validation_closed_widths_positive():
    r = ssv.validation_closed.reverse_lookup_binary(n=334, phi=0.30, c_stat=0.75, n_sims=300)
    assert r["c_stat_ci_width"] > 0
    assert r["calibration_slope_ci_width"] > 0
    # bigger N -> narrower c-statistic CI
    r2 = ssv.validation_closed.reverse_lookup_binary(n=1000, phi=0.30, c_stat=0.75, n_sims=300)
    assert r2["c_stat_ci_width"] < r["c_stat_ci_width"]


def test_validation_sim_recovers_slope_one():
    r = ssv.validation_sim.reverse_lookup_simulation(n=500, mu=-0.85, sigma=1.0, n_sims=200)
    assert r["calibration_slope_mean"] == pytest.approx(1.0, abs=0.15)


def test_compare_auc_delta_to_theta_roundtrip():
    for theta in (0.70, 0.76, 0.85):
        d = ssv.compare_auc.theta_to_delta(theta)
        assert ssv.compare_auc.delta_to_theta(d) == pytest.approx(theta, abs=1e-6)


def test_compare_auc_power_monotone_in_n():
    p1 = ssv.compare_auc.achieved_power(200, 0.75, 0.82, 0.5, 0.30)["power"]
    p2 = ssv.compare_auc.achieved_power(600, 0.75, 0.82, 0.5, 0.30)["power"]
    assert p2 > p1


def test_data_params_from_predictions():
    rng = np.random.default_rng(0)
    n = 400
    lp = rng.normal(-0.85, 1.2, n)
    p = 1 / (1 + np.exp(-lp))
    y = rng.binomial(1, p)
    est = ssv.data.params_from_predictions({"predicted_prob": p, "true_outcome": y})
    assert 0.2 < est["phi"] < 0.4
    assert 0.6 < est["c_stat"] < 0.9
