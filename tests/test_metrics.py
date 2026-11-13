"""Tests for dynamical system identification metrics and trajectory validation routines."""

from __future__ import annotations

import sys
import os
import numpy as np

# Ensure src is on Python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sindy.metrics import (
    support_metrics,
    coef_error,
    derivative_r2,
    rollout_error,
    lyapunov_time_ratio,
    attractor_similarity,
)
from sindy.systems import lorenz


def test_support_and_coef_error_identical_systems() -> None:
    """Verify that identical coefficient matrices yield zero error and perfect support."""
    Xi_true = np.zeros((10, 3), dtype=np.float64)
    Xi_true[[1, 2], 0] = [-10.0, 10.0]
    Xi_true[[1, 2, 6], 1] = [28.0, -1.0, -1.0]
    Xi_true[[3, 5], 2] = [-8.0 / 3.0, 1.0]

    Xi_found = np.copy(Xi_true)

    # 1. Support metrics
    sup = support_metrics(Xi_found, Xi_true)
    assert sup["precision"] == 1.0
    assert sup["recall"] == 1.0
    assert sup["f1"] == 1.0
    assert sup["fp"] == 0
    assert sup["fn"] == 0
    assert sup["tp"] == 7

    # 2. Coefficient relative Frobenius error
    err = coef_error(Xi_found, Xi_true)
    assert err == 0.0


def test_support_one_wrong_coefficient_yields_recall_one_precision_less_one() -> None:
    """Verify that a model with one extra false positive term yields recall=1.0 and precision < 1.0."""
    Xi_true = np.zeros((10, 3), dtype=np.float64)
    Xi_true[[1, 2], 0] = [-10.0, 10.0]
    Xi_true[[1, 2, 6], 1] = [28.0, -1.0, -1.0]
    Xi_true[[3, 5], 2] = [-8.0 / 3.0, 1.0]

    # Add exactly one spurious/wrong coefficient
    Xi_found = np.copy(Xi_true)
    Xi_found[4, 0] = 0.5  # Spurious term x^2 in dx/dt

    sup = support_metrics(Xi_found, Xi_true)

    assert sup["recall"] == 1.0, f"Expected recall 1.0, got {sup['recall']}"
    assert sup["precision"] < 1.0, f"Expected precision < 1.0, got {sup['precision']}"
    assert np.isclose(sup["precision"], 7.0 / 8.0), f"Expected precision 7/8 = 0.875, got {sup['precision']}"
    assert sup["fp"] == 1
    assert sup["fn"] == 0
    assert sup["tp"] == 7

    # Coefficient error should now be strictly positive
    err = coef_error(Xi_found, Xi_true)
    assert err > 0.0


def test_derivative_r2() -> None:
    """Test coefficient of determination R^2 on derivatives."""
    rng = np.random.default_rng(42)
    m, p, n = 100, 6, 2
    Theta = rng.standard_normal((m, p))
    Xi = rng.standard_normal((p, n))
    dXdt_exact = Theta @ Xi

    # Exact match -> R^2 == 1.0
    r2_exact = derivative_r2(Theta, Xi, dXdt_exact)
    assert np.isclose(r2_exact, 1.0, atol=1e-12)

    # Noisy dXdt -> R^2 < 1.0
    dXdt_noisy = dXdt_exact + 0.1 * rng.standard_normal((m, n))
    r2_noisy = derivative_r2(Theta, Xi, dXdt_noisy)
    assert 0.95 < r2_noisy < 1.0


def test_rollout_error_and_divergence_time() -> None:
    """Test forward rollout error and divergence time detection."""
    x0 = np.array([-8.0, 8.0, 27.0])
    t_end = 5.0
    dt = 0.005

    # Identical vector fields -> error is 0.0 and divergence time is t_end
    t, err, div_t = rollout_error(lorenz, lorenz, x0, t_end=t_end, dt=dt)
    assert np.all(err == 0.0)
    assert div_t == t_end

    # Perturbed system
    def lorenz_perturbed(x: np.ndarray) -> np.ndarray:
        return lorenz(x, params={"sigma": 11.0, "rho": 28.0, "beta": 8.0 / 3.0})

    t_p, err_p, div_t_p = rollout_error(lorenz, lorenz_perturbed, x0, t_end=10.0, dt=dt, divergence_threshold=0.1)
    assert div_t_p < 10.0
    assert err_p[0] == 0.0
    assert err_p[-1] > 0.1


def test_lyapunov_time_ratio() -> None:
    """Test Lyapunov time ratio calculation."""
    div_time = 10.0
    lam_max = 0.906
    ratio = lyapunov_time_ratio(div_time, lambda_max=lam_max)
    assert np.isclose(ratio, 9.06)


def test_attractor_similarity() -> None:
    """Test invariant measure and statistical similarity on identical trajectories."""
    from sindy.simulate import simulate
    x0 = np.array([-8.0, 8.0, 27.0])
    t, X = simulate(lorenz, x0, t_end=10.0, dt=0.01)

    sim = attractor_similarity(X, X, bins=30)
    assert sim["wasserstein_z"] == 0.0
    assert sim["density_tv_xz"] == 0.0
    assert sim["density_js_xz"] == 0.0


if __name__ == "__main__":
    tests = [
        test_support_and_coef_error_identical_systems,
        test_support_one_wrong_coefficient_yields_recall_one_precision_less_one,
        test_derivative_r2,
        test_rollout_error_and_divergence_time,
        test_lyapunov_time_ratio,
        test_attractor_similarity,
    ]

    print(f"Running {len(tests)} metrics tests...")
    for test in tests:
        print(f"  -> {test.__name__} ...", end=" ", flush=True)
        test()
        print("PASSED")
    print("\nAll metrics tests PASSED successfully!")
