"""Tests for Sequentially Thresholded Least Squares (STLSQ) optimizer."""

from __future__ import annotations

import sys
import os
import numpy as np

# Ensure src is on Python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sindy.stlsq import STLSQ, pretty_print_equations
from sindy.systems import lorenz
from sindy.simulate import simulate
from sindy.library import FeatureLibrary


def test_synthetic_exact_recovery_zero_noise() -> None:
    """(a) Test exact coefficient and support recovery on clean synthetic data."""
    rng = np.random.default_rng(42)
    m, p, n = 300, 10, 3

    Theta = rng.standard_normal((m, p))
    Xi_true = np.zeros((p, n), dtype=np.float64)

    # Sparse true coefficients
    Xi_true[[1, 3], 0] = [-5.0, 3.5]
    Xi_true[[2, 5, 8], 1] = [12.0, -1.5, 0.8]
    Xi_true[[0, 4], 2] = [2.0, -7.2]

    dXdt = Theta @ Xi_true

    model = STLSQ(threshold=0.2, max_iter=20, normalize=True)
    model.fit(Theta, dXdt)

    assert model.coef_ is not None
    assert model.coef_.shape == (p, n)

    # Exact coefficient recovery
    max_error = np.max(np.abs(model.coef_ - Xi_true))
    assert max_error < 1e-10, f"Max coefficient error {max_error:.4e} exceeds 1e-10"

    # Exact support recovery
    support_recovered = np.array_equal((model.coef_ != 0.0), (Xi_true != 0.0))
    assert support_recovered, "Recovered support does not match true support"

    # Predictions match dXdt
    pred = model.predict(Theta)
    np.testing.assert_allclose(pred, dXdt, atol=1e-10)


def test_support_recovery_with_one_percent_noise() -> None:
    """(b) Test that active support is perfectly recovered under 1% measurement noise."""
    rng = np.random.default_rng(101)
    m, p, n = 500, 10, 2

    Theta = rng.standard_normal((m, p))
    Xi_true = np.zeros((p, n), dtype=np.float64)
    Xi_true[[1, 4], 0] = [6.0, -4.5]
    Xi_true[[2, 7], 1] = [-8.0, 3.2]

    dXdt = Theta @ Xi_true
    # 1% noise relative to std
    noise = 0.01 * np.std(dXdt, axis=0) * rng.standard_normal(dXdt.shape)
    dXdt_noisy = dXdt + noise

    model = STLSQ(threshold=0.5, max_iter=25, normalize=True)
    model.fit(Theta, dXdt_noisy)

    assert model.coef_ is not None
    recovered_support = (model.coef_ != 0.0)
    true_support = (Xi_true != 0.0)

    assert np.array_equal(recovered_support, true_support), (
        f"Support mismatch under 1% noise.\nRecovered:\n{recovered_support}\nTrue:\n{true_support}"
    )

    # Coefficients should remain close to truth
    rel_error = np.linalg.norm(model.coef_ - Xi_true) / np.linalg.norm(Xi_true)
    assert rel_error < 0.02, f"Relative error {rel_error:.4f} exceeds 2%"


def test_threshold_sensitivity_extremes() -> None:
    """(c) Test that too large lambda deletes real terms, and too small lambda keeps junk."""
    rng = np.random.default_rng(202)
    m, p, n = 250, 8, 2

    Theta = rng.standard_normal((m, p))
    Xi_true = np.zeros((p, n), dtype=np.float64)
    Xi_true[[1, 2], 0] = [3.0, -2.5]
    Xi_true[[0, 4, 6], 1] = [1.5, 4.0, -2.0]

    dXdt = Theta @ Xi_true
    # Add modest noise so spurious correlations exist
    noise = 0.05 * np.std(dXdt, axis=0) * rng.standard_normal(dXdt.shape)
    dXdt_noisy = dXdt + noise

    true_nonzero_count = np.count_nonzero(Xi_true != 0.0)

    # Extreme 1: excessively large threshold (e.g. 50.0) deletes genuine terms
    model_too_large = STLSQ(threshold=50.0, max_iter=20, normalize=True).fit(Theta, dXdt_noisy)
    count_too_large = np.count_nonzero(model_too_large.coef_ != 0.0)
    assert count_too_large < true_nonzero_count, (
        f"Expected large lambda to delete real terms: got {count_too_large} active, true has {true_nonzero_count}"
    )

    # Extreme 2: excessively small threshold (e.g. 1e-6) retains noisy junk terms
    model_too_small = STLSQ(threshold=1e-6, max_iter=20, normalize=True).fit(Theta, dXdt_noisy)
    count_too_small = np.count_nonzero(model_too_small.coef_ != 0.0)
    assert count_too_small > true_nonzero_count, (
        f"Expected small lambda to retain junk: got {count_too_small} active, true has {true_nonzero_count}"
    )


def test_monotonic_non_increasing_active_set() -> None:
    """(d) Test that the active set S_k is strictly monotonically non-increasing across iterations."""
    rng = np.random.default_rng(303)
    m, p, n = 350, 15, 3

    Theta = rng.standard_normal((m, p))
    Xi_true = np.zeros((p, n), dtype=np.float64)
    Xi_true[[0, 3, 7], 0] = [5.0, -3.0, 2.0]
    Xi_true[[2, 6], 1] = [-4.0, 7.0]
    Xi_true[[1, 5, 10, 12], 2] = [2.5, -6.0, 1.8, -3.5]

    dXdt = Theta @ Xi_true + 0.02 * rng.standard_normal((m, n))

    model = STLSQ(threshold=0.8, max_iter=20, normalize=True)
    model.fit(Theta, dXdt)

    history = model.active_history_
    assert len(history) >= 2, "Expected at least 2 iterations recorded in history"

    for it in range(len(history) - 1):
        mask_prev = history[it]
        mask_next = history[it + 1]

        # Monotonicity test: mask_next must be a subset of mask_prev
        # Once an entry is False in mask_prev, it must remain False in mask_next
        reactivated_entries = np.count_nonzero(mask_next & ~mask_prev)
        assert reactivated_entries == 0, (
            f"Active set violated monotonicity at iteration {it}: {reactivated_entries} terms reactivated"
        )

        # Active count must be non-increasing
        assert np.count_nonzero(mask_next) <= np.count_nonzero(mask_prev)


def test_entire_column_inactive_edge_case() -> None:
    """Verify safe handling when an entire output column is pruned to all zeros."""
    rng = np.random.default_rng(404)
    m, p = 100, 6
    Theta = rng.standard_normal((m, p))

    # Col 0 has real signal; Col 1 has pure small noise
    dXdt = np.empty((m, 2))
    dXdt[:, 0] = 5.0 * Theta[:, 1]
    dXdt[:, 1] = 0.001 * rng.standard_normal(m)

    model = STLSQ(threshold=0.1, max_iter=10, normalize=True).fit(Theta, dXdt)

    assert model.coef_ is not None
    # Col 1 should be completely zeroed out without crashing
    assert np.all(model.coef_[:, 1] == 0.0)
    # Col 0 should keep its true active term
    assert np.count_nonzero(model.coef_[:, 0] != 0.0) == 1
    assert np.isclose(model.coef_[1, 0], 5.0, atol=1e-10)


def test_pretty_print_equations_formatting() -> None:
    """Test pretty_print_equations on canonical Lorenz-63 coefficients."""
    feats = ["1", "x", "y", "z", "x^2", "x y", "x z", "y^2", "y z", "z^2"]
    p = len(feats)
    Xi = np.zeros((p, 3), dtype=np.float64)

    # Lorenz parameters: sigma=10, rho=28, beta=8/3
    Xi[1, 0] = -10.0
    Xi[2, 0] = 10.0

    Xi[1, 1] = 28.0
    Xi[2, 1] = -1.0
    Xi[6, 1] = -1.0

    Xi[3, 2] = -8.0 / 3.0  # -2.667
    Xi[5, 2] = 1.0

    # Add small sub-cutoff noise to ensure it is omitted
    Xi[0, 0] = 1e-15
    Xi[4, 1] = -5e-14

    eqs = pretty_print_equations(Xi, feats, var_names=["x", "y", "z"], precision=3)

    assert len(eqs) == 3
    assert eqs[0] == "dx/dt = -10.000 x + 10.000 y"
    assert eqs[1] == "dy/dt = 28.000 x - 1.000 y - 1.000 x z"
    assert eqs[2] == "dz/dt = -2.667 z + 1.000 x y"

    # Test all-zero column produces "= 0"
    zero_eq = pretty_print_equations(np.zeros((p, 1)), feats, var_names=["w"])
    assert zero_eq[0] == "dw/dt = 0"


def test_lorenz_discovery_end_to_end() -> None:
    """End-to-end integration test: discover exact Lorenz ODEs from trajectory data."""
    x0 = np.array([-8.0, 8.0, 27.0])
    dt = 0.01
    t, X = simulate(lorenz, x0, t_end=20.0, dt=dt)
    dX_true = lorenz(X)

    lib = FeatureLibrary(poly_degree=2, include_bias=True, var_names=["x", "y", "z"])
    Theta = lib.fit_transform(X)

    model = STLSQ(threshold=0.1, max_iter=20, normalize=True)
    model.fit(Theta, dX_true)

    eqs = pretty_print_equations(model.coef_, lib.get_feature_names(), ["x", "y", "z"])

    print("\n" + "=" * 60)
    print("STLSQ DISCOVERED LORENZ EQUATIONS")
    print("=" * 60)
    for eq in eqs:
        print("  ", eq)
    print(f"Iterations to converge: {model.n_iter_}")
    print("=" * 60)

    # Verify exact parameter recovery
    # dx/dt: -10 x + 10 y
    np.testing.assert_allclose(model.coef_[[1, 2], 0], [-10.0, 10.0], rtol=1e-5)
    assert np.count_nonzero(model.coef_[:, 0] != 0.0) == 2

    # dy/dt: 28 x - y - x z
    np.testing.assert_allclose(model.coef_[[1, 2, 6], 1], [28.0, -1.0, -1.0], rtol=1e-5)
    assert np.count_nonzero(model.coef_[:, 1] != 0.0) == 3

    # dz/dt: -2.667 z + x y
    np.testing.assert_allclose(model.coef_[[3, 5], 2], [-8.0 / 3.0, 1.0], rtol=1e-5)
    assert np.count_nonzero(model.coef_[:, 2] != 0.0) == 2


if __name__ == "__main__":
    tests = [
        test_synthetic_exact_recovery_zero_noise,
        test_support_recovery_with_one_percent_noise,
        test_threshold_sensitivity_extremes,
        test_monotonic_non_increasing_active_set,
        test_entire_column_inactive_edge_case,
        test_pretty_print_equations_formatting,
        test_lorenz_discovery_end_to_end,
    ]

    print(f"Running {len(tests)} STLSQ tests...")
    for test in tests:
        print(f"  -> {test.__name__} ...", end=" ", flush=True)
        test()
        print("PASSED")
    print("\nAll STLSQ tests PASSED successfully!")

