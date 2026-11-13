"""Evaluation metrics for data-driven dynamical system identification and trajectory forecasting.

Provides:
1. `support_metrics`: Precision, recall, and F1 score over the sparse coefficient pattern.
2. `coef_error`: Relative Frobenius norm error of discovered coefficients vs truth.
3. `derivative_r2`: Coefficient of determination R^2 on reconstructed time derivatives.
4. `rollout_error`: Relative trajectory tracking error curve and divergence time.
5. `lyapunov_time_ratio`: Divergence time normalized by characteristic Lyapunov times.
6. `attractor_similarity`: Statistical divergence of coordinate distributions and phase portraits.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple
import numpy as np

from sindy.simulate import simulate


def support_metrics(
    Xi_found: np.ndarray,
    Xi_true: np.ndarray,
    tol: float = 1e-5,
) -> Dict[str, float]:
    """Compute precision, recall, and F1 score over the active sparsity support pattern.

    Parameters
    ----------
    Xi_found : np.ndarray
        Estimated coefficient matrix of shape (p, n).
    Xi_true : np.ndarray
        Ground-truth coefficient matrix of shape (p, n).
    tol : float, default=1e-5
        Absolute cutoff threshold for classifying a coefficient as non-zero.

    Returns
    -------
    Dict[str, float]
        Dictionary containing precision, recall, f1, tp, fp, fn, and tn.
    """
    found_arr = np.asarray(Xi_found, dtype=np.float64)
    true_arr = np.asarray(Xi_true, dtype=np.float64)

    if found_arr.shape != true_arr.shape:
        raise ValueError(f"Shape mismatch: Xi_found {found_arr.shape} vs Xi_true {true_arr.shape}")

    found_mask = np.abs(found_arr) > tol
    true_mask = np.abs(true_arr) > tol

    tp = int(np.count_nonzero(found_mask & true_mask))
    fp = int(np.count_nonzero(found_mask & ~true_mask))
    fn = int(np.count_nonzero(~found_mask & true_mask))
    tn = int(np.count_nonzero(~found_mask & ~true_mask))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def coef_error(
    Xi_found: np.ndarray,
    Xi_true: np.ndarray,
) -> float:
    """Compute the relative Frobenius norm error between discovered and true coefficients.

    Formula:
        ||Xi_found - Xi_true||_F / ||Xi_true||_F

    Parameters
    ----------
    Xi_found : np.ndarray
        Estimated coefficient matrix of shape (p, n).
    Xi_true : np.ndarray
        Ground-truth coefficient matrix of shape (p, n).

    Returns
    -------
    float
        Relative Frobenius norm error.
    """
    found_arr = np.asarray(Xi_found, dtype=np.float64)
    true_arr = np.asarray(Xi_true, dtype=np.float64)

    norm_true = float(np.linalg.norm(true_arr))
    if norm_true == 0.0:
        return float(np.linalg.norm(found_arr))

    return float(np.linalg.norm(found_arr - true_arr) / norm_true)


def derivative_r2(
    Theta: np.ndarray,
    Xi: np.ndarray,
    dXdt: np.ndarray,
) -> float:
    """Compute the coefficient of determination R^2 between predicted and observed derivatives.

    Formula:
        R^2 = 1 - ||dXdt - Theta * Xi||_F^2 / ||dXdt - mean(dXdt)||_F^2

    Parameters
    ----------
    Theta : np.ndarray
        Candidate feature matrix of shape (m, p).
    Xi : np.ndarray
        Coefficient matrix of shape (p, n).
    dXdt : np.ndarray
        Target time-derivative matrix of shape (m, n).

    Returns
    -------
    float
        Total variance explained R^2 in [-inf, 1.0].
    """
    theta_arr = np.asarray(Theta, dtype=np.float64)
    xi_arr = np.asarray(Xi, dtype=np.float64)
    dxdt_arr = np.asarray(dXdt, dtype=np.float64)

    pred = theta_arr @ xi_arr
    ss_res = float(np.sum((dxdt_arr - pred) ** 2))
    ss_tot = float(np.sum((dxdt_arr - np.mean(dxdt_arr, axis=0, keepdims=True)) ** 2))

    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0

    return float(1.0 - (ss_res / ss_tot))


def rollout_error(
    f_true: Callable[[np.ndarray], np.ndarray],
    f_found: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    t_end: float,
    dt: float,
    divergence_threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute forward trajectory simulation error curve and divergence time.

    Integrates both models with 4th-order Runge-Kutta from the identical initial condition x0.
    Evaluates the relative error at each discrete time step:
        e(t) = ||x_true(t) - x_found(t)||_2 / ||x_true(t)||_2

    Divergence time is defined as the first timestamp where e(t) >= divergence_threshold.

    Parameters
    ----------
    f_true : Callable[[np.ndarray], np.ndarray]
        Ground-truth vector field function dx/dt = f_true(x).
    f_found : Callable[[np.ndarray], np.ndarray]
        Discovered SINDy vector field function dx/dt = f_found(x).
    x0 : np.ndarray
        Initial state vector of shape (n,).
    t_end : float
        Total rollout duration.
    dt : float
        Numerical integration time step.
    divergence_threshold : float, default=0.1
        Relative error cutoff for declaring trajectory divergence (e.g. 10% error).

    Returns
    -------
    t : np.ndarray
        Time array of shape (m,).
    rel_error : np.ndarray
        Relative error vector of shape (m,) where each element is e(t_i).
    divergence_time : float
        Timestamp when rel_error first reaches divergence_threshold, or t_end if never.
    """
    t, X_true = simulate(f_true, x0, t_end=t_end, dt=dt)
    _, X_found = simulate(f_found, x0, t_end=t_end, dt=dt)

    norm_diff = np.linalg.norm(X_true - X_found, axis=1)
    norm_true = np.linalg.norm(X_true, axis=1)
    norm_true_safe = np.where(norm_true == 0.0, 1e-12, norm_true)
    rel_error = norm_diff / norm_true_safe

    # Identify first instant exceeding divergence threshold or encountering numerical instability
    diverged = (rel_error >= divergence_threshold) | np.isnan(rel_error) | np.isinf(rel_error)
    exceed_indices = np.where(diverged)[0]
    if len(exceed_indices) > 0:
        divergence_time = float(t[exceed_indices[0]])
    else:
        divergence_time = float(t_end)

    return t, rel_error, divergence_time


def lyapunov_time_ratio(
    divergence_time: float,
    lambda_max: float = 0.906,
) -> float:
    """Compute trajectory divergence time in units of characteristic Lyapunov times.

    Formula:
        tau = divergence_time * lambda_max

    Theoretical Significance in Chaotic Dynamical Systems:
    ------------------------------------------------------
    Deterministic chaotic systems exhibit sensitive dependence on initial conditions
    governed by the maximal Lyapunov exponent lambda_max:
        ||delta x(t)|| approx ||delta x_0|| * exp(lambda_max * t)

    Even with an analytically exact vector field, machine roundoff (eps_mach approx 1e-16)
    acts as an unavoidable initial perturbation delta x_0. Under forward integration,
    this microscopic discrepancy is amplified exponentially by exp(lambda_max * t),
    eventually causing any two simulated trajectories to diverge by O(1) across the attractor
    within a finite predictability horizon:
        t_horizon approx (1 / lambda_max) * ln(1 / eps_mach)

    For the Lorenz-63 system (sigma=10, rho=28, beta=8/3), the maximal Lyapunov exponent
    is lambda_max approx 0.906, meaning 1 Lyapunov time is 1 / lambda_max approx 1.104 seconds.
    Normalizing divergence times by lambda_max provides an invariant measure of predictive
    capacity across dynamical regimes. Because point-wise trajectory forecasting is fundamentally
    bounded by chaos, model quality over long horizons must be evaluated via geometric and
    statistical attractor invariants rather than point-by-point differences.

    Parameters
    ----------
    divergence_time : float
        Observed time duration before trajectory error exceeds threshold.
    lambda_max : float, default=0.906
        Maximal Lyapunov exponent of the continuous dynamical system.

    Returns
    -------
    float
        Divergence duration expressed in dimensionless Lyapunov times.
    """
    if divergence_time < 0.0:
        raise ValueError(f"divergence_time must be non-negative, got {divergence_time}")
    if lambda_max <= 0.0:
        raise ValueError(f"lambda_max must be positive, got {lambda_max}")

    return float(divergence_time * lambda_max)


def attractor_similarity(
    X_true: np.ndarray,
    X_found: np.ndarray,
    bins: int = 50,
) -> Dict[str, float]:
    """Measure statistical similarity of chaotic attractors via invariant density metrics.

    Computes:
    1. 1D Wasserstein distance (Earth Mover's Distance) on the coordinate distributions.
    2. Total Variation distance between 2D binned probability densities on the (x, z) plane.
    3. Jensen-Shannon divergence between 2D (x, z) probability histograms.

    Parameters
    ----------
    X_true : np.ndarray
        Trajectory generated by true system of shape (m, n).
    X_found : np.ndarray
        Trajectory generated by discovered model of shape (m, n).
    bins : int, default=50
        Number of histogram bins per coordinate axis.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        - 'wasserstein_z': 1D optimal transport distance on z coordinate.
        - 'density_tv_xz': Total variation distance between (x, z) density maps in [0, 1].
        - 'density_js_xz': Jensen-Shannon divergence between (x, z) density maps in [0, ln 2].
    """
    x_t = np.asarray(X_true, dtype=np.float64)
    x_f = np.asarray(X_found, dtype=np.float64)

    m = min(len(x_t), len(x_f))
    # 1. Exact 1D Wasserstein Distance on z coordinate via sorted quantile differences
    z_true_sorted = np.sort(x_t[:m, 2])
    z_found_sorted = np.sort(x_f[:m, 2])
    w1_z = float(np.mean(np.abs(z_true_sorted - z_found_sorted)))

    # 2. 2D Probability Density Histogram on (x, z) plane
    x_min = min(float(np.min(x_t[:, 0])), float(np.min(x_f[:, 0])))
    x_max = max(float(np.max(x_t[:, 0])), float(np.max(x_f[:, 0])))
    z_min = min(float(np.min(x_t[:, 2])), float(np.min(x_f[:, 2])))
    z_max = max(float(np.max(x_t[:, 2])), float(np.max(x_f[:, 2])))

    range_2d = [[x_min, x_max], [z_min, z_max]]

    hist_t, _, _ = np.histogram2d(x_t[:, 0], x_t[:, 2], bins=bins, range=range_2d)
    hist_f, _, _ = np.histogram2d(x_f[:, 0], x_f[:, 2], bins=bins, range=range_2d)

    # Normalize to probability mass functions
    P = hist_t / max(np.sum(hist_t), 1e-12)
    Q = hist_f / max(np.sum(hist_f), 1e-12)

    # Total Variation distance: 0.5 * sum(|P - Q|)
    tv_xz = float(0.5 * np.sum(np.abs(P - Q)))

    # Jensen-Shannon Divergence: 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    M = 0.5 * (P + Q)
    eps = 1e-15

    kl_pm = np.sum(np.where(P > eps, P * np.log((P + eps) / (M + eps)), 0.0))
    kl_qm = np.sum(np.where(Q > eps, Q * np.log((Q + eps) / (M + eps)), 0.0))
    js_xz = float(0.5 * kl_pm + 0.5 * kl_qm)

    return {
        "wasserstein_z": w1_z,
        "density_tv_xz": tv_xz,
        "density_js_xz": js_xz,
    }
