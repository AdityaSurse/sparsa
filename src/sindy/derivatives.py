"""Numerical differentiation methods for trajectory state observations.

Provides:
1. `central_diff`: 2nd-order interior central differences with 2nd-order one-sided boundaries.
2. `savgol_diff`: Savitzky-Golay polynomial differentiation filter.
3. `tv_diff`: Total-variation regularized differentiation (ADMM solver).

All functions accept trajectory array X of shape (m, n) and time step dt,
returning time derivative dXdt of shape (m, n).
"""

from __future__ import annotations

from typing import Optional
import numpy as np

try:
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def central_diff(X: np.ndarray, dt: float) -> np.ndarray:
    """Compute 2nd-order finite difference derivatives.

    Uses central differences for interior points:
        dX/dt[i] = (X[i+1] - X[i-1]) / (2 * dt)
    and 2nd-order one-sided stencils at boundaries:
        dX/dt[0]   = (-3 * X[0] + 4 * X[1] - X[2]) / (2 * dt)
        dX/dt[m-1] = ( 3 * X[m-1] - 4 * X[m-2] + X[m-3]) / (2 * dt)

    This stencil is algebraically exact on quadratic polynomials x(t) = a*t^2 + b*t + c.

    Parameters
    ----------
    X : np.ndarray
        Trajectory data matrix of shape (m, n) or 1D array of shape (m,).
    dt : float
        Uniform sampling interval (time step).

    Returns
    -------
    np.ndarray
        Estimated derivative matrix of shape (m, n) or (m,).
    """
    if dt <= 0.0:
        raise ValueError(f"Time step dt must be positive, got {dt}")

    x_arr = np.asarray(X, dtype=np.float64)
    is_1d = (x_arr.ndim == 1)
    x_2d = np.atleast_2d(x_arr).T if is_1d else x_arr

    m, n = x_2d.shape
    if m < 3:
        raise ValueError(f"At least 3 temporal points required for 2nd-order differences, got {m}")

    dXdt = np.empty_like(x_2d)

    # 2nd-order central differences in the interior
    dXdt[1:-1, :] = (x_2d[2:, :] - x_2d[:-2, :]) / (2.0 * dt)

    # 2nd-order one-sided forward stencil at t_0
    dXdt[0, :] = (-3.0 * x_2d[0, :] + 4.0 * x_2d[1, :] - x_2d[2, :]) / (2.0 * dt)

    # 2nd-order one-sided backward stencil at t_{m-1}
    dXdt[-1, :] = (3.0 * x_2d[-1, :] - 4.0 * x_2d[-2, :] + x_2d[-3, :]) / (2.0 * dt)

    return dXdt[:, 0] if is_1d else dXdt


def savgol_diff(
    X: np.ndarray,
    dt: float,
    window: int = 15,
    polyorder: int = 3,
) -> np.ndarray:
    """Compute numerical derivatives using a Savitzky-Golay polynomial filter.

    Fits local polynomials to sliding windows and evaluates the analytical
    first derivative at the center point. Boundary regions are handled via
    consistent polynomial extrapolation.

    Parameters
    ----------
    X : np.ndarray
        Trajectory array of shape (m, n) or (m,).
    dt : float
        Uniform sampling time step.
    window : int, default=15
        Window length in number of samples. Must be a positive odd integer.
    polyorder : int, default=3
        Degree of the polynomial to fit. Must be less than `window`.

    Returns
    -------
    np.ndarray
        Smoothed derivative matrix of shape (m, n) or (m,).
    """
    if dt <= 0.0:
        raise ValueError(f"Time step dt must be positive, got {dt}")
    if window % 2 == 0:
        raise ValueError(f"Window length must be odd, got {window}")
    if polyorder >= window:
        raise ValueError(f"polyorder ({polyorder}) must be strictly less than window ({window})")

    x_arr = np.asarray(X, dtype=np.float64)
    is_1d = (x_arr.ndim == 1)
    x_2d = np.atleast_2d(x_arr).T if is_1d else x_arr

    m, n = x_2d.shape
    if m < window:
        raise ValueError(f"Trajectory length m={m} must be >= window length {window}")

    if _HAS_SCIPY:
        dXdt = savgol_filter(
            x_2d,
            window_length=window,
            polyorder=polyorder,
            deriv=1,
            delta=dt,
            axis=0,
            mode="interp",
        )
    else:
        # High-precision pure NumPy implementation identical to scipy's 'interp' mode
        half = window // 2
        z = np.arange(-half, half + 1)
        A = np.vander(z, polyorder + 1, increasing=True)
        pinv = np.linalg.pinv(A)
        kernel = pinv[1] / dt

        dXdt = np.empty_like(x_2d)

        # Interior convolution
        for col in range(n):
            dXdt[half:-half, col] = np.convolve(x_2d[:, col], kernel[::-1], mode="valid")

        # Boundary polynomial extrapolation
        t_edge = np.arange(window)
        A_edge = np.vander(t_edge, polyorder + 1, increasing=True)
        pinv_edge = np.linalg.pinv(A_edge)
        deriv_weights = np.arange(1, polyorder + 1)

        for col in range(n):
            # Left edge
            c_left = pinv_edge @ x_2d[:window, col]
            for i in range(half):
                p_powers = i ** np.arange(polyorder)
                dXdt[i, col] = np.sum(deriv_weights * c_left[1:] * p_powers) / dt

            # Right edge
            c_right = pinv_edge @ x_2d[-window:, col]
            for i in range(half):
                idx = window - half + i
                p_powers = idx ** np.arange(polyorder)
                dXdt[m - half + i, col] = np.sum(deriv_weights * c_right[1:] * p_powers) / dt

    return dXdt[:, 0] if is_1d else dXdt


def tv_diff(
    X: np.ndarray,
    dt: float,
    alpha: float = 0.05,
    max_iter: int = 40,
    rho: Optional[float] = None,
) -> np.ndarray:
    """Compute Total-Variation (TV) regularized numerical derivatives via ADMM.

    Solves the convex variational problem:
        min_u  0.5 * || A u - (X - X[0]) ||_2^2 + alpha * || D u ||_1
    where A is the continuous integration operator and D is the 1st-difference
    operator, penalizing rapid oscillations while preserving true gradient edges.

    Parameters
    ----------
    X : np.ndarray
        Observed noisy trajectory of shape (m, n) or (m,).
    dt : float
        Uniform sampling time step.
    alpha : float, default=0.05
        Total variation regularization weight. Larger values promote smoother derivatives.
    max_iter : int, default=40
        Maximum ADMM iterations.
    rho : Optional[float], default=None
        Augmented Lagrangian penalty parameter. Defaults to max(1.0, 10.0 * alpha).

    Returns
    -------
    np.ndarray
        Regularized derivative matrix of shape (m, n) or (m,).
    """
    if dt <= 0.0:
        raise ValueError(f"Time step dt must be positive, got {dt}")
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")

    x_arr = np.asarray(X, dtype=np.float64)
    is_1d = (x_arr.ndim == 1)
    x_2d = np.atleast_2d(x_arr).T if is_1d else x_arr

    m, n = x_2d.shape
    if m < 3:
        raise ValueError(f"Trajectory length m={m} must be >= 3")

    if alpha == 0.0:
        return central_diff(X, dt)

    if rho is None:
        rho = max(1.0, 10.0 * alpha)

    # Integrated target Y = X - X[0]
    Y = x_2d - x_2d[0, :]

    # Trapezoidal integration operator matrix A of shape (m, m)
    A = np.tril(np.ones((m, m), dtype=np.float64))
    A[:, 0] -= 0.5
    A[np.arange(m), np.arange(m)] -= 0.5
    A *= dt
    A[0, :] = 0.0

    # 1st-difference operator matrix D of shape (m-1, m)
    # D @ u = u[1:] - u[:-1]
    ATA = A.T @ A
    DTD = np.zeros((m, m), dtype=np.float64)
    np.fill_diagonal(DTD, 2.0)
    DTD[0, 0] = 1.0
    DTD[-1, -1] = 1.0
    DTD[np.arange(m - 1), np.arange(1, m)] = -1.0
    DTD[np.arange(1, m), np.arange(m - 1)] = -1.0

    # Pre-factor regularized linear system M = ATA + rho * DTD
    M = ATA + rho * DTD
    M += 1e-8 * np.eye(m)  # Numerical stability ridge
    L = np.linalg.cholesky(M)

    ATY = A.T @ Y
    U = np.zeros((m, n), dtype=np.float64)
    V = np.zeros((m - 1, n), dtype=np.float64)
    W = np.zeros((m - 1, n), dtype=np.float64)

    tau = alpha / rho

    # ADMM Iteration loop
    for _ in range(max_iter):
        # 1. Update U: (ATA + rho * DTD) U = ATY + rho * D^T (V - W)
        diff_term = V - W
        DT_diff = np.zeros((m, n), dtype=np.float64)
        DT_diff[0, :] = -diff_term[0, :]
        DT_diff[1:-1, :] = diff_term[:-1, :] - diff_term[1:, :]
        DT_diff[-1, :] = diff_term[-1, :]

        RHS = ATY + rho * DT_diff
        Z = np.linalg.solve(L, RHS)
        U = np.linalg.solve(L.T, Z)

        # 2. Update V via soft-thresholding on D @ U + W
        DU = U[1:, :] - U[:-1, :]
        target = DU + W
        V = np.sign(target) * np.maximum(np.abs(target) - tau, 0.0)

        # 3. Dual update W
        W = W + DU - V

    return U[:, 0] if is_1d else U

