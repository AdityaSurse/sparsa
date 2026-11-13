"""Unit tests for Weak-Form SINDy variational projection and test function kernels."""

from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sindy.weak_form import (
    weak_test_function_kernels,
    weak_form_projection,
)
from sindy.library import FeatureLibrary
from sindy.stlsq import STLSQ
from sindy.metrics import support_metrics


def test_kernel_quadrature_properties():
    """Verify normalization, symmetry, and boundary vanishing of test function kernels."""
    w = 21
    dt = 0.01
    psi_feat, psi_deriv, C_phi = weak_test_function_kernels(w, dt, p_poly=4)

    assert len(psi_feat) == w
    assert len(psi_deriv) == w
    assert C_phi > 0.0

    # Normalization: sum(psi_feat) must equal 1.0
    np.testing.assert_allclose(np.sum(psi_feat), 1.0, atol=1e-12)

    # Vanishing boundary / zero-mean: sum(psi_deriv) must equal 0.0
    np.testing.assert_allclose(np.sum(psi_deriv), 0.0, atol=1e-12)

    # Symmetry: phi is symmetric around midpoint, phi' is anti-symmetric
    np.testing.assert_allclose(psi_feat, psi_feat[::-1], atol=1e-12)
    np.testing.assert_allclose(psi_deriv, -psi_deriv[::-1], atol=1e-12)


def test_integration_by_parts_exact_identity():
    """Verify that weak-form integration by parts matches analytical derivative on clean data."""
    dt = 0.005
    t = np.arange(0.0, 5.0, dt)
    # Sine and cosine test signal
    X = np.column_stack([np.sin(2.0 * np.pi * t), np.cos(2.0 * np.pi * t)])
    dX_analytic = np.column_stack([2.0 * np.pi * np.cos(2.0 * np.pi * t), -2.0 * np.pi * np.sin(2.0 * np.pi * t)])

    # Use identity library Theta(X) = X
    lib = FeatureLibrary(poly_degree=1, include_bias=False, var_names=["s", "c"])
    Theta = lib.fit_transform(X)

    Theta_weak, G_weak = weak_form_projection(X, Theta, dt=dt, window_sec=0.2, stride_ratio=0.25)

    assert Theta_weak.ndim == 2
    assert G_weak.ndim == 2
    assert Theta_weak.shape[0] == G_weak.shape[0]

    # For dx/dt = [0, 2*pi; -2*pi, 0] * [s; c]
    # G_weak should match Theta_weak @ Xi_true
    Xi_true = np.array([
        [0.0, -2.0 * np.pi],
        [2.0 * np.pi, 0.0],
    ])
    G_expected = Theta_weak @ Xi_true
    rel_error = np.linalg.norm(G_weak - G_expected) / np.linalg.norm(G_expected)
    assert rel_error < 1e-4, f"Integration by parts error {rel_error} exceeds tolerance"


def test_weak_sindy_recovers_support_at_high_noise():
    """Verify that weak-form SINDy maintains support recovery at 5% noise."""
    from sindy.systems import lorenz
    from sindy.simulate import simulate

    dt = 0.005
    # Simulate 3 trajectories
    rng = np.random.default_rng(123)
    trajs = []
    for _ in range(3):
        x0 = rng.uniform(low=[-15.0, -15.0, 10.0], high=[15.0, 15.0, 40.0])
        _, X = simulate(lorenz, x0, t_end=6.0, dt=dt)
        X_noisy = X + 0.05 * np.std(X, axis=0) * rng.standard_normal(X.shape)
        trajs.append(X_noisy)

    lib = FeatureLibrary(poly_degree=3, include_bias=True, var_names=["x", "y", "z"])
    lib.fit(trajs[0])
    names = lib.get_feature_names()
    p = len(names)

    Xi_true = np.zeros((p, 3))
    Xi_true[names.index("x"), 0] = -10.0
    Xi_true[names.index("y"), 0] = 10.0
    Xi_true[names.index("x"), 1] = 28.0
    Xi_true[names.index("y"), 1] = -1.0
    Xi_true[names.index("x z"), 1] = -1.0
    Xi_true[names.index("z"), 2] = -8.0 / 3.0
    Xi_true[names.index("x y"), 2] = 1.0

    theta_weak_list = []
    g_weak_list = []
    for tr in trajs:
        th = lib.transform(tr)
        th_w, g_w = weak_form_projection(tr, th, dt=dt, window_sec=0.25, stride_ratio=0.25)
        theta_weak_list.append(th_w)
        g_weak_list.append(g_w)

    Theta_w = np.vstack(theta_weak_list)
    G_w = np.vstack(g_weak_list)

    opt = STLSQ(threshold=0.15, max_iter=25, normalize=True)
    opt.fit(Theta_w, G_w)

    supp = support_metrics(opt.coef_, Xi_true)
    assert supp["f1"] >= 0.90, f"Weak SINDy F1={supp['f1']} is below expected 0.90 at 5% noise"


if __name__ == "__main__":
    print("Running weak-form unit tests...")
    test_kernel_quadrature_properties()
    print("  -> test_kernel_quadrature_properties ... PASSED")
    test_integration_by_parts_exact_identity()
    print("  -> test_integration_by_parts_exact_identity ... PASSED")
    test_weak_sindy_recovers_support_at_high_noise()
    print("  -> test_weak_sindy_recovers_support_at_high_noise ... PASSED")
    print("\nAll weak-form tests PASSED successfully!")

