"""Weak-Form SINDy formulation via test-function integration by parts.

Implements the variational / weak formulation of Sparse Identification of Nonlinear Dynamics:
Instead of estimating pointwise derivatives dX/dt from noisy trajectory data, both sides of
    dx/dt = Theta(x) * Xi
are integrated against smooth, compactly supported test functions phi_k(t):
    integral_{a_k}^{b_k} phi_k(t) * (dx/dt) dt = integral_{a_k}^{b_k} phi_k(t) * Theta(x(t)) dt * Xi

By integration by parts:
    integral_{a_k}^{b_k} phi_k(t) * (dx/dt) dt = [phi_k(t) * x(t)]_{a_k}^{b_k} - integral_{a_k}^{b_k} phi'_k(t) * x(t) dt

Choosing test functions with vanishing boundary values phi_k(a_k) = phi_k(b_k) = 0 eliminates
the boundary term entirely:
    - integral_{a_k}^{b_k} phi'_k(t) * x(t) dt = ( integral_{a_k}^{b_k} phi_k(t) * Theta(x(t)) dt ) * Xi

This transfers differentiation to the analytical, infinitely smooth test function phi_k(t),
completely bypassing numerical differentiation of noisy data and attenuating high-frequency noise
via temporal averaging.
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np


def weak_test_function_kernels(
    w: int,
    dt: float,
    p_poly: int = 4,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Construct discrete quadrature convolution kernels for weak-form integration.

    Uses smooth polynomial bump test functions on normalized coordinate tau in [-1, 1]:
        phi(tau) = (1 - tau^2)^p_poly
    with analytical derivative:
        dphi/dtau = -2 * p_poly * tau * (1 - tau^2)^(p_poly - 1)
        phi'(t) = (2 / T_w) * (dphi/dtau)
    where T_w = (w - 1) * dt is the temporal window width.

    Parameters
    ----------
    w : int
        Number of discrete grid points in the test window (must be odd and >= 5).
    dt : float
        Uniform sampling time step.
    p_poly : int, default=4
        Polynomial exponent ensuring C^{p_poly-1} smoothness and vanishing boundary derivatives.

    Returns
    -------
    psi_feat : np.ndarray
        Feature averaging kernel of shape (w,), normalized such that sum(psi_feat) = 1.
    psi_deriv : np.ndarray
        Effective derivative kernel of shape (w,), satisfying sum(psi_deriv) = 0.
    C_phi : float
        Normalization integral C_phi = integral_{a}^{b} phi(t) dt.
    """
    if w < 5:
        raise ValueError(f"Window length w must be at least 5, got {w}")
    if w % 2 == 0:
        w += 1
    if dt <= 0.0:
        raise ValueError(f"Time step dt must be positive, got {dt}")

    tau = np.linspace(-1.0, 1.0, w)
    phi = (1.0 - tau**2)**p_poly
    dphi_dtau = -2.0 * p_poly * tau * (1.0 - tau**2)**(p_poly - 1)

    t_w = (w - 1) * dt
    phi_prime = (2.0 / t_w) * dphi_dtau

    # Trapezoidal quadrature weights
    w_quad = np.full(w, dt, dtype=np.float64)
    w_quad[0] *= 0.5
    w_quad[-1] *= 0.5

    C_phi = float(np.sum(w_quad * phi))
    if C_phi <= 0.0:
        raise ValueError("Degenerate test function integral C_phi <= 0")

    # Convolution kernels normalized by C_phi so that physical units match dx/dt and Theta(X)
    psi_feat = (w_quad * phi) / C_phi
    psi_deriv = -(w_quad * phi_prime) / C_phi

    return psi_feat, psi_deriv, C_phi


def weak_form_projection(
    X: np.ndarray,
    Theta: np.ndarray,
    dt: float,
    window_sec: float = 0.25,
    stride_ratio: float = 0.25,
    p_poly: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project trajectory data and candidate feature matrix onto compactly supported test functions.

    Parameters
    ----------
    X : np.ndarray
        State trajectory matrix of shape (m, n).
    Theta : np.ndarray
        Candidate feature matrix of shape (m, p) evaluated on X.
    dt : float
        Sampling time step.
    window_sec : float, default=0.25
        Physical duration of test function support in seconds.
    stride_ratio : float, default=0.25
        Stride between consecutive test windows as a fraction of window width.
    p_poly : int, default=4
        Polynomial order for test function bump.

    Returns
    -------
    Theta_weak : np.ndarray
        Weak-form integrated candidate feature matrix of shape (K, p).
    G_weak : np.ndarray
        Weak-form integrated derivative target matrix of shape (K, n).
    """
    x_arr = np.asarray(X, dtype=np.float64)
    theta_arr = np.asarray(Theta, dtype=np.float64)

    m, n = x_arr.shape
    m_theta, p = theta_arr.shape
    if m != m_theta:
        raise ValueError(f"Snapshot mismatch: X has {m} rows, Theta has {m_theta} rows")

    # Determine window length w in samples
    w = max(9, int(np.round(window_sec / dt)))
    if w % 2 == 0:
        w += 1
    if w > m:
        w = m if (m % 2 != 0) else (m - 1)
        if w < 5:
            raise ValueError(f"Trajectory length m={m} is too short for weak-form integration")

    psi_feat, psi_deriv, _ = weak_test_function_kernels(w, dt, p_poly=p_poly)

    stride = max(1, int(np.round(w * stride_ratio)))

    # Efficient vectorization using sliding window views
    X_win = np.lib.stride_tricks.sliding_window_view(x_arr, w, axis=0)[::stride]  # (K, n, w)
    Theta_win = np.lib.stride_tricks.sliding_window_view(theta_arr, w, axis=0)[::stride]  # (K, p, w)

    G_weak = np.einsum("knw, w -> kn", X_win, psi_deriv)
    Theta_weak = np.einsum("kpw, w -> kp", Theta_win, psi_feat)

    return Theta_weak, G_weak

