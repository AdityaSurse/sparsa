"""Tests for FeatureLibrary: combinatorial monomial expansion, shapes, names, and conditioning."""

from __future__ import annotations

import sys
import os
import math
import numpy as np

# Ensure src is on Python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sindy.library import FeatureLibrary, print_library
from sindy.systems import lorenz
from sindy.simulate import simulate


def test_combinatorial_feature_count() -> None:
    """Verify that p matches theoretical combinatorial counts (n+d)! / (n! d!)."""
    test_cases = [
        # (n, degree, include_bias, trig, expected_p)
        (3, 2, True, False, 10),    # binom(3+2, 2) = 10
        (3, 3, True, False, 20),    # binom(3+3, 3) = 20
        (2, 2, True, False, 6),     # binom(2+2, 2) = 6
        (4, 2, True, False, 15),    # binom(4+2, 2) = 15
        (3, 1, True, False, 4),     # 1 + 3 = 4
        (3, 2, False, False, 9),    # 10 - 1 = 9
        (2, 2, True, True, 10),     # 6 + 2*sin + 2*cos = 10
        (1, 4, True, False, 5),     # 1, x, x^2, x^3, x^4 = 5
    ]

    for n, deg, bias, trig, expected_p in test_cases:
        lib = FeatureLibrary(poly_degree=deg, include_bias=bias, trig=trig)
        dummy_X = np.ones((5, n))
        lib.fit(dummy_X)

        assert lib.n_features == expected_p, (
            f"Failed count for n={n}, deg={deg}, bias={bias}, trig={trig}: "
            f"expected {expected_p}, got {lib.n_features}"
        )
        assert len(lib.get_feature_names()) == expected_p


def test_manual_computation_3_row_example() -> None:
    """Verify column values exactly match manual analytical calculation on a 3-row matrix."""
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [0.0, -1.0],
    ])

    lib = FeatureLibrary(poly_degree=2, include_bias=True, trig=False, var_names=["x", "y"])
    Theta = lib.fit_transform(X)

    expected_names = ["1", "x", "y", "x^2", "x y", "y^2"]
    assert lib.get_feature_names() == expected_names

    expected_Theta = np.array([
        [1.0, 1.0,  2.0, 1.0,  2.0,  4.0],
        [1.0, 3.0,  4.0, 9.0, 12.0, 16.0],
        [1.0, 0.0, -1.0, 0.0,  0.0,  1.0],
    ])

    np.testing.assert_allclose(Theta, expected_Theta, rtol=1e-14, atol=1e-14)


def test_names_align_with_columns() -> None:
    """Verify that each column in Theta matches the exact analytical function of its name."""
    rng = np.random.default_rng(42)
    m, n = 50, 3
    X = rng.uniform(low=-2.0, high=2.0, size=(m, n))
    var_names = ["x", "y", "z"]

    lib = FeatureLibrary(poly_degree=2, include_bias=True, trig=True, var_names=var_names)
    Theta = lib.fit_transform(X)
    names = lib.get_feature_names()

    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    lookup = {
        "1": np.ones(m),
        "x": x,
        "y": y,
        "z": z,
        "x^2": x**2,
        "x y": x * y,
        "x z": x * z,
        "y^2": y**2,
        "y z": y * z,
        "z^2": z**2,
        "sin(x)": np.sin(x),
        "cos(x)": np.cos(x),
        "sin(y)": np.sin(y),
        "cos(y)": np.cos(y),
        "sin(z)": np.sin(z),
        "cos(z)": np.cos(z),
    }

    assert len(names) == len(lookup)

    for col_idx, name in enumerate(names):
        assert name in lookup, f"Unexpected feature name: {name}"
        expected_col = lookup[name]
        actual_col = Theta[:, col_idx]
        np.testing.assert_allclose(
            actual_col, expected_col, rtol=1e-14, atol=1e-14,
            err_msg=f"Column mismatch for feature '{name}'"
        )


def test_no_nans_or_infs() -> None:
    """Verify that Theta matrix never contains NaNs or Infs."""
    rng = np.random.default_rng(101)
    X = rng.normal(size=(100, 4))
    lib = FeatureLibrary(poly_degree=3, include_bias=True, trig=True)
    Theta = lib.fit_transform(X)

    assert not np.isnan(Theta).any()
    assert not np.isinf(Theta).any()


def test_normalization_and_unnormalize_xi() -> None:
    """Test column normalization and exact unnormalization of regression coefficients."""
    rng = np.random.default_rng(202)
    X = rng.uniform(1.0, 5.0, size=(60, 2))

    lib = FeatureLibrary(poly_degree=2, include_bias=True, normalize=True)
    Theta_norm = lib.fit_transform(X)

    # Check unit L2 norm
    column_norms = np.linalg.norm(Theta_norm, axis=0)
    np.testing.assert_allclose(column_norms, np.ones(lib.n_features), rtol=1e-12)

    # Synthetic coefficient matrix Xi
    p = lib.n_features
    n = 2
    Xi_norm = rng.standard_normal((p, n))

    # Evaluate Theta_norm @ Xi_norm
    pred_from_norm = Theta_norm @ Xi_norm

    # Unnormalize Xi and compare with unnormalized Theta
    Xi_orig = lib.unnormalize_xi(Xi_norm)
    lib_unnorm = FeatureLibrary(poly_degree=2, include_bias=True, normalize=False)
    Theta_orig = lib_unnorm.fit_transform(X)
    pred_from_orig = Theta_orig @ Xi_orig

    np.testing.assert_allclose(pred_from_norm, pred_from_orig, rtol=1e-12)


def test_lorenz_condition_numbers() -> None:
    """Report and assert condition numbers for Lorenz-63 library at degree 2 and degree 3."""
    x0 = np.array([-8.0, 8.0, 27.0])
    t, X = simulate(lorenz, x0, t_end=20.0, dt=0.01)

    lib2 = FeatureLibrary(poly_degree=2, include_bias=True)
    Theta2 = lib2.fit_transform(X)
    cond2 = float(np.linalg.cond(Theta2))

    lib3 = FeatureLibrary(poly_degree=3, include_bias=True)
    Theta3 = lib3.fit_transform(X)
    cond3 = float(np.linalg.cond(Theta3))

    print("\n" + "=" * 60)
    print("LORENZ TRAJECTORY CANDIDATE LIBRARY CONDITION NUMBERS")
    print("=" * 60)
    print(f"Degree 2 (p={lib2.n_features}): cond(Theta) = {cond2:12.2f}")
    print(f"Degree 3 (p={lib3.n_features}): cond(Theta) = {cond3:12.2f}")
    print(f"Condition number ratio (deg3 / deg2) = {cond3 / cond2:8.2f}x")
    print("=" * 60)

    # Sanity bounds on conditioning
    assert 1e3 < cond2 < 2e4, f"Degree 2 cond(Theta) was {cond2:.2e}"
    assert 1e5 < cond3 < 2e6, f"Degree 3 cond(Theta) was {cond3:.2e}"
    # Verify higher degree increases ill-conditioning by ~ 2 orders of magnitude
    assert cond3 > 50.0 * cond2


def test_print_library() -> None:
    """Test print_library helper executes without error."""
    lib = FeatureLibrary(poly_degree=2, include_bias=True, trig=True, var_names=["x", "y"])
    dummy_X = np.ones((4, 2))
    lib.fit(dummy_X)
    print("\n")
    print_library(lib)


if __name__ == "__main__":
    tests = [
        test_combinatorial_feature_count,
        test_manual_computation_3_row_example,
        test_names_align_with_columns,
        test_no_nans_or_infs,
        test_normalization_and_unnormalize_xi,
        test_lorenz_condition_numbers,
        test_print_library,
    ]

    print(f"Running {len(tests)} FeatureLibrary tests...")
    for test in tests:
        print(f"  -> {test.__name__} ...", end=" ", flush=True)
        test()
        print("PASSED")
    print("\nAll FeatureLibrary tests PASSED successfully!")

