"""Tests for numerical differentiation routines: central differences, Savitzky-Golay, and TV regularization."""

from __future__ import annotations

import sys
import os
import numpy as np

# Ensure src is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sindy.derivatives import central_diff, savgol_diff, tv_diff
from sindy.systems import lorenz
from sindy.simulate import simulate, add_noise


def test_central_diff_exact_on_quadratic() -> None:
    """Test that central_diff is exact to machine precision (< 1e-10) on x(t) = t^2."""
    dt = 0.01
    t = np.arange(0.0, 5.0, dt)
    # Both 1D and 2D checks
    x_2d = (t**2).reshape(-1, 1)
    dx_exact_2d = (2.0 * t).reshape(-1, 1)

    dx_cd_2d = central_diff(x_2d, dt)
    max_error_2d = np.max(np.abs(dx_cd_2d - dx_exact_2d))
    assert max_error_2d < 1e-10, f"Central diff max error on t^2 was {max_error_2d:.4e} (expected < 1e-10)"

    # Check 1D input
    dx_cd_1d = central_diff(t**2, dt)
    assert dx_cd_1d.shape == t.shape
    max_error_1d = np.max(np.abs(dx_cd_1d - 2.0 * t))
    assert max_error_1d < 1e-10


def test_estimator_output_shapes() -> None:
    """Verify that all estimators preserve input dimensionality for (m, n) and (m,)."""
    m, n = 150, 4
    dt = 0.05
    rng = np.random.default_rng(42)
    X_2d = rng.standard_normal((m, n))
    X_1d = rng.standard_normal(m)

    for method in [central_diff, savgol_diff, tv_diff]:
        res_2d = method(X_2d, dt)
        assert res_2d.shape == (m, n), f"{method.__name__} failed 2D shape check: got {res_2d.shape}"

        res_1d = method(X_1d, dt)
        assert res_1d.shape == (m,), f"{method.__name__} failed 1D shape check: got {res_1d.shape}"


def test_savgol_beats_central_diff_at_noise_001() -> None:
    """Verify that savgol_diff outperforms central_diff on noisy Lorenz data at noise=0.01."""
    dt = 0.01
    x0 = np.array([-8.0, 8.0, 27.0])
    t, X_clean = simulate(lorenz, x0, t_end=15.0, dt=dt)
    dX_true = lorenz(X_clean)

    # Corrupt with 1% noise
    X_noisy = add_noise(X_clean, noise_level=0.01, seed=123)

    dX_cd = central_diff(X_noisy, dt)
    dX_sg = savgol_diff(X_noisy, dt, window=17, polyorder=3)

    rmse_cd = np.sqrt(np.mean((dX_cd - dX_true) ** 2))
    rmse_sg = np.sqrt(np.mean((dX_sg - dX_true) ** 2))

    assert rmse_sg < rmse_cd, (
        f"Savitzky-Golay (RMSE={rmse_sg:.3f}) should beat Central Diff (RMSE={rmse_cd:.3f})"
    )
    # Expect at least a 3x error reduction on 1% noise at dt=0.01
    improvement_ratio = rmse_cd / rmse_sg
    assert improvement_ratio > 3.0, (
        f"Expected > 3.0x improvement, got {improvement_ratio:.2f}x"
    )


def test_tv_diff_noise_suppression() -> None:
    """Verify that tv_diff significantly suppresses heavy noise."""
    dt = 0.01
    t = np.linspace(0.0, 5.0, 501)
    x_clean = np.sin(2.0 * t).reshape(-1, 1)
    dx_true = (2.0 * np.cos(2.0 * t)).reshape(-1, 1)

    rng = np.random.default_rng(99)
    x_noisy = x_clean + 0.05 * rng.standard_normal(x_clean.shape)

    dx_cd = central_diff(x_noisy, dt)
    dx_tv = tv_diff(x_noisy, dt, alpha=0.01, max_iter=40)

    rmse_cd = np.sqrt(np.mean((dx_cd - dx_true) ** 2))
    rmse_tv = np.sqrt(np.mean((dx_tv - dx_true) ** 2))

    assert rmse_tv < rmse_cd * 0.35, (
        f"TV regularized diff ({rmse_tv:.3f}) should drastically beat Central Diff ({rmse_cd:.3f})"
    )


if __name__ == "__main__":
    tests = [
        test_central_diff_exact_on_quadratic,
        test_estimator_output_shapes,
        test_savgol_beats_central_diff_at_noise_001,
        test_tv_diff_noise_suppression,
    ]

    print(f"Running {len(tests)} differentiation tests...")
    for test in tests:
        print(f"  -> {test.__name__} ...", end=" ", flush=True)
        test()
        print("PASSED")
    print("\nAll differentiation tests PASSED successfully!")

