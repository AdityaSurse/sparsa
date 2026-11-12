"""Tests for numerical simulation, RK4 integration, dynamical systems, and noise injection."""

from __future__ import annotations

import sys
from typing import Callable, List, Tuple
import numpy as np

# Ensure src is on path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sindy.systems import (
    SYSTEMS,
    lorenz,
    lotka_volterra,
    van_der_pol,
    damped_pendulum,
    LORENZ_TRUE_COEFFICIENTS,
    LOTKA_VOLTERRA_TRUE_COEFFICIENTS,
    VAN_DER_POL_TRUE_COEFFICIENTS,
    DAMPED_PENDULUM_TRUE_COEFFICIENTS,
)
from sindy.simulate import (
    rk4_step,
    simulate,
    generate_dataset,
    add_noise,
)


def test_rk4_exponential_decay() -> None:
    """(a) Test that RK4 on dx/dt = -x matches exp(-t) with error < 1e-8 at dt=0.001."""
    def linear_decay(x: np.ndarray) -> np.ndarray:
        return -x

    x0 = np.array([1.0], dtype=np.float64)
    t_end = 2.0
    dt = 0.001

    t, X = simulate(linear_decay, x0, t_end=t_end, dt=dt)
    x_exact = np.exp(-t)

    # Validate shapes
    assert t.shape == (2001,)
    assert X.shape == (2001, 1)

    max_error = np.max(np.abs(X[:, 0] - x_exact))
    assert max_error < 1e-8, f"Max error {max_error:.4e} exceeds 1e-8 threshold"


def test_rk4_fourth_order_convergence() -> None:
    """(b) Test that halving dt reduces global error by approximately 2^4 = 16 (4th-order)."""
    def linear_decay(x: np.ndarray) -> np.ndarray:
        return -x

    x0 = np.array([1.0], dtype=np.float64)
    t_end = 1.0

    dt_coarse = 0.05
    dt_fine = dt_coarse / 2.0  # 0.025

    t_coarse, X_coarse = simulate(linear_decay, x0, t_end=t_end, dt=dt_coarse)
    t_fine, X_fine = simulate(linear_decay, x0, t_end=t_end, dt=dt_fine)

    err_coarse = np.max(np.abs(X_coarse[:, 0] - np.exp(-t_coarse)))
    err_fine = np.max(np.abs(X_fine[:, 0] - np.exp(-t_fine)))

    error_ratio = err_coarse / err_fine
    estimated_order = np.log2(error_ratio)

    # 4th-order RK4 should yield error ratio ~ 16, order ~ 4.0
    assert 14.5 < error_ratio < 17.5, (
        f"Expected error ratio ~ 16.0, got {error_ratio:.4f} (order: {estimated_order:.2f})"
    )
    assert 3.8 < estimated_order < 4.2, f"Estimated order {estimated_order:.2f} is not ~ 4.0"


def test_lorenz_bounded_and_butterfly_shape() -> None:
    """(c) Test that Lorenz stays bounded and has the butterfly attractor shape (check z range)."""
    x0 = np.array([-8.0, 8.0, 27.0], dtype=np.float64)
    t_end = 40.0
    dt = 0.005

    t, X = simulate(lorenz, x0, t_end=t_end, dt=dt)

    assert np.all(np.isfinite(X)), "Lorenz trajectory contains non-finite values (NaN/Inf)"

    x_vals = X[:, 0]
    y_vals = X[:, 1]
    z_vals = X[:, 2]

    # Verify z stays strictly positive and within the standard Lorenz-63 bound [0, 55]
    z_min = np.min(z_vals)
    z_max = np.max(z_vals)
    assert z_min > 0.0, f"Lorenz z minimum should be > 0, got {z_min:.2f}"
    assert z_max < 55.0, f"Lorenz z maximum should be < 55, got {z_max:.2f}"

    # Verify butterfly shape: orbits alternate across both positive and negative x/y lobes
    assert np.min(x_vals) < -12.0 and np.max(x_vals) > 12.0, (
        f"x range [{np.min(x_vals):.2f}, {np.max(x_vals):.2f}] does not span both lobes"
    )
    assert np.min(y_vals) < -15.0 and np.max(y_vals) > 15.0, (
        f"y range [{np.min(y_vals):.2f}, {np.max(y_vals):.2f}] does not span both lobes"
    )


def test_add_noise_std_level() -> None:
    """(d) Test that added noise std matches the requested noise_level * std(X)."""
    rng = np.random.default_rng(12345)
    m, n = 80000, 3

    # Synthetic multi-variable trajectory with distinct variances
    X = rng.normal(loc=[10.0, -5.0, 25.0], scale=[2.0, 7.0, 15.0], size=(m, n))
    noise_level = 0.05  # 5%

    X_noisy = add_noise(X, noise_level=noise_level, seed=42)
    assert X_noisy.shape == X.shape

    noise = X_noisy - X
    empirical_std = np.std(noise, axis=0, ddof=0)
    expected_std = noise_level * np.std(X, axis=0, ddof=0)

    relative_discrepancy = np.abs(empirical_std - expected_std) / expected_std

    # With 80,000 samples, empirical std should match expected std within 1.5%
    assert np.all(relative_discrepancy < 0.015), (
        f"Noise std discrepancy {relative_discrepancy} exceeds 1.5% tolerance"
    )

    # Test noise_level = 0 returns an exact copy
    X_clean_copy = add_noise(X, noise_level=0.0)
    np.testing.assert_array_equal(X, X_clean_copy)


def test_all_systems_vectorized_and_1d() -> None:
    """Verify that all system vector fields support both 1D and 2D batch inputs."""
    systems_to_test = [
        (lorenz, 3),
        (lotka_volterra, 2),
        (van_der_pol, 2),
        (damped_pendulum, 2),
    ]

    for f_sys, dim in systems_to_test:
        x_1d = np.ones(dim, dtype=np.float64)
        dx_1d = f_sys(x_1d)
        assert dx_1d.shape == (dim,), f"{f_sys.__name__} failed 1D shape check"

        x_2d = np.ones((10, dim), dtype=np.float64)
        dx_2d = f_sys(x_2d)
        assert dx_2d.shape == (10, dim), f"{f_sys.__name__} failed 2D batch shape check"

        # Check consistency between 1D and 2D
        np.testing.assert_allclose(dx_1d, dx_2d[0])


def test_dataset_generation_and_oracle_derivatives() -> None:
    """Test multi-trajectory generation and oracle derivative computation."""
    n_traj = 3
    t_end = 2.0
    dt = 0.01

    dataset = generate_dataset("lorenz", n_traj=n_traj, t_end=t_end, dt=dt, seed=99)
    assert len(dataset) == n_traj

    for t, X, dXdt_true in dataset:
        assert t.shape == (201,)
        assert X.shape == (201, 3)
        assert dXdt_true.shape == (201, 3)

        # Confirm oracle derivative matches direct system evaluation on every row
        direct_eval = lorenz(X)
        np.testing.assert_allclose(dXdt_true, direct_eval, rtol=1e-12, atol=1e-12)


def test_coefficient_dictionaries() -> None:
    """Verify coefficient dictionary structures for all dynamical systems."""
    assert "x" in LORENZ_TRUE_COEFFICIENTS
    assert LORENZ_TRUE_COEFFICIENTS["x"]["x"] == -10.0
    assert LORENZ_TRUE_COEFFICIENTS["x"]["y"] == 10.0
    assert LORENZ_TRUE_COEFFICIENTS["y"]["x z"] == -1.0
    assert LORENZ_TRUE_COEFFICIENTS["z"]["x y"] == 1.0

    assert LOTKA_VOLTERRA_TRUE_COEFFICIENTS["u"]["u"] == 1.0
    assert LOTKA_VOLTERRA_TRUE_COEFFICIENTS["u"]["u v"] == -0.1

    assert VAN_DER_POL_TRUE_COEFFICIENTS["y"]["mu" if "mu" in VAN_DER_POL_TRUE_COEFFICIENTS["y"] else "y"] == 2.0
    assert VAN_DER_POL_TRUE_COEFFICIENTS["y"]["x^2 y"] == -2.0

    assert DAMPED_PENDULUM_TRUE_COEFFICIENTS["omega"]["omega"] == -0.25
    assert DAMPED_PENDULUM_TRUE_COEFFICIENTS["omega"]["sin(theta)"] == -5.0


if __name__ == "__main__":
    test_functions = [
        test_rk4_exponential_decay,
        test_rk4_fourth_order_convergence,
        test_lorenz_bounded_and_butterfly_shape,
        test_add_noise_std_level,
        test_all_systems_vectorized_and_1d,
        test_dataset_generation_and_oracle_derivatives,
        test_coefficient_dictionaries,
    ]

    print(f"Running {len(test_functions)} test suites...")
    for fn in test_functions:
        print(f"  -> {fn.__name__} ...", end=" ", flush=True)
        fn()
        print("PASSED")
    print("\nAll tests PASSED successfully!")

