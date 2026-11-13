"""Sequentially Thresholded Least Squares (STLSQ) optimizer for sparse regression.

Theoretical Background and Objective Derivation:
------------------------------------------------
In data-driven dynamical system discovery (SINDy), we seek a parsimonious
representation of the vector field x_dot = Theta(x) * Xi, balancing least-squares
fidelity against structural sparsity:

    min_{Xi}  0.5 * || dXdt - Theta * Xi ||_F^2 + lambda * || Xi ||_0

where ||Xi||_0 = sum_{j, k} I(Xi_{j, k} != 0) is the L0 pseudo-norm counting
active basis terms.

Direct L0 minimization is an NP-hard combinatorial problem over 2^p subset
patterns. A standard convex surrogate is the L1 relaxation (LASSO):

    min_{Xi}  0.5 * || dXdt - Theta * Xi ||_F^2 + lambda * || Xi ||_1

Why Refitting Eliminates the Shrinkage Bias of LASSO:
------------------------------------------------------
Under an orthogonal design, the closed-form minimizer of the LASSO objective
is given by soft-thresholding the ordinary least-squares (OLS) estimate:

    xi_LASSO = sign(xi_OLS) * max(|xi_OLS| - lambda, 0)

Every non-zero coefficient is systematically attenuated toward zero by the
penalty margin lambda. In nonlinear dynamics, this "shrinkage bias" alters
critical eigenvalues, shifts bifurcation points, and degrades long-term
trajectory forecasting.

STLSQ eliminates shrinkage bias through a two-step alternating scheme:
1. Support Selection: Coefficients below threshold lambda are pruned to zero.
2. Unconstrained Refitting: On the active support set S = {j : |xi_{j, k}| >= lambda},
   the algorithm solves an unconstrained ordinary least-squares problem:

       xi_S = argmin_c || dXdt_k - Theta_S * c ||_2^2
            = (Theta_S^T Theta_S)^(-1) Theta_S^T dXdt_k

Because the active coefficients are re-estimated without any L1 or L2 penalty,
the refitted parameters are asymptotically unbiased and match the oracle OLS
estimates on the true governing support.
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import numpy as np


class STLSQ:
    """Sequentially Thresholded Least Squares (STLSQ) sparse regression optimizer.

    Alternates between unconstrained least-squares regression via QR/SVD decomposition
    and hard thresholding until the active coefficient support stabilizes.

    Parameters
    ----------
    threshold : float, default=0.1
        Sparsity cutoff threshold (lambda). Active coefficients with absolute value
        below this threshold are iteratively pruned to zero.
    max_iter : int, default=20
        Maximum number of threshold-and-refit iterations.
    normalize : bool, default=True
        Whether to normalize the columns of Theta internally to unit L2 norm during
        optimization, and rescale the resulting coefficient matrix back to the original
        physical units upon convergence.

    Attributes
    ----------
    coef_ : np.ndarray
        Learned sparse coefficient matrix of shape (p, n).
    n_iter_ : int
        Number of threshold-refit iterations executed until support convergence.
    active_history_ : List[np.ndarray]
        Sequence of boolean support masks of shape (p, n) recorded at each iteration.
    norms_ : Optional[np.ndarray]
        Column L2 norms of Theta used for internal scaling.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        max_iter: int = 20,
        normalize: bool = True,
    ) -> None:
        if threshold < 0.0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")

        self.threshold = float(threshold)
        self.max_iter = int(max_iter)
        self.normalize = bool(normalize)

        self.coef_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self.active_history_: List[np.ndarray] = []
        self.norms_: Optional[np.ndarray] = None

    def fit(self, Theta: np.ndarray, dXdt: np.ndarray) -> STLSQ:
        """Fit the sparse coefficient matrix Xi to satisfy dXdt approx Theta @ Xi.

        Parameters
        ----------
        Theta : np.ndarray
            Candidate feature matrix of shape (m, p).
        dXdt : np.ndarray
            Target time-derivative matrix of shape (m, n) or vector of shape (m,).

        Returns
        -------
        STLSQ
            Fitted optimizer instance.
        """
        theta_arr = np.asarray(Theta, dtype=np.float64)
        dxdt_arr = np.asarray(dXdt, dtype=np.float64)

        if theta_arr.ndim != 2:
            raise ValueError(f"Theta must be 2D of shape (m, p), got {theta_arr.shape}")

        is_1d = (dxdt_arr.ndim == 1)
        dxdt_2d = dxdt_arr[:, np.newaxis] if is_1d else dxdt_arr

        m, p = theta_arr.shape
        m_y, n = dxdt_2d.shape

        if m != m_y:
            raise ValueError(f"Sample size mismatch: Theta has {m} rows, dXdt has {m_y}")

        # 1. Internal column normalization of Theta
        if self.normalize:
            col_norms = np.linalg.norm(theta_arr, axis=0)
            col_norms = np.where(col_norms == 0.0, 1.0, col_norms)
            Theta_work = theta_arr / col_norms
            self.norms_ = col_norms
        else:
            col_norms = np.ones(p, dtype=np.float64)
            Theta_work = theta_arr
            self.norms_ = col_norms

        # 2. Initial unconstrained least-squares solve
        Xi, _, _, _ = np.linalg.lstsq(Theta_work, dxdt_2d, rcond=None)

        # Initial active mask: all features initially active
        active_mask = np.ones((p, n), dtype=bool)
        self.active_history_ = [np.copy(active_mask)]

        # 3. Iterative threshold-and-refit loop
        for _ in range(self.max_iter):
            # Identify small coefficients below threshold lambda
            small = np.abs(Xi) < self.threshold
            new_active_mask = ~small

            # Termination condition: support mask has stabilized
            if np.array_equal(new_active_mask, self.active_history_[-1]):
                break

            active_mask = new_active_mask
            self.active_history_.append(np.copy(active_mask))

            # Zero out pruned entries
            Xi[small] = 0.0

            # Refit active coefficients column-by-column
            for k in range(n):
                active_k = active_mask[:, k]
                n_active = np.count_nonzero(active_k)

                if n_active > 0:
                    # Unconstrained refit on the active sub-dictionary
                    sub_theta = Theta_work[:, active_k]
                    sub_y = dxdt_2d[:, k]
                    coef_k, _, _, _ = np.linalg.lstsq(sub_theta, sub_y, rcond=None)
                    Xi[active_k, k] = coef_k
                    Xi[~active_k, k] = 0.0
                else:
                    # Handle edge case where entire column becomes inactive
                    Xi[:, k] = 0.0

        # Rescale coefficients back to original physical coordinates
        self.coef_ = Xi / col_norms[:, np.newaxis]
        self.n_iter_ = len(self.active_history_) - 1

        if is_1d:
            self.coef_ = self.coef_[:, 0]

        return self

    def predict(self, Theta: np.ndarray) -> np.ndarray:
        """Predict derivative values using learned coefficient matrix.

        Parameters
        ----------
        Theta : np.ndarray
            Feature matrix of shape (m, p).

        Returns
        -------
        np.ndarray
            Reconstructed derivative matrix of shape (m, n).
        """
        if self.coef_ is None:
            raise ValueError("STLSQ model is not fitted. Call fit() first.")
        theta_arr = np.asarray(Theta, dtype=np.float64)
        return theta_arr @ self.coef_


def pretty_print_equations(
    coef: np.ndarray,
    feature_names: Sequence[str],
    var_names: Optional[Sequence[str]] = None,
    precision: int = 3,
    cutoff: float = 1e-12,
) -> List[str]:
    """Format learned sparse equations into readable symbolic mathematical strings.

    Example output:
        "dx/dt = -10.000 x + 10.000 y"
        "dy/dt = 28.000 x - 1.000 y - 1.000 x z"

    Parameters
    ----------
    coef : np.ndarray
        Learned coefficient matrix of shape (p, n) or vector of shape (p,).
    feature_names : Sequence[str]
        List of p candidate basis term names.
    var_names : Optional[Sequence[str]], default=None
        List of n state variable names. If None, defaults to ['x', 'y', 'z']
        for n <= 3, or ['x0', 'x1', ...] for general n.
    precision : int, default=3
        Number of decimal digits for floating-point coefficients.
    cutoff : float, default=1e-12
        Threshold below which coefficients are omitted from the equation string.

    Returns
    -------
    List[str]
        List of formatted equation strings, one per state coordinate.
    """
    coef_arr = np.asarray(coef, dtype=np.float64)
    if coef_arr.ndim == 1:
        coef_arr = coef_arr[:, np.newaxis]

    p, n = coef_arr.shape
    if len(feature_names) != p:
        raise ValueError(f"feature_names length ({len(feature_names)}) must match coef rows ({p})")

    if var_names is None:
        if n == 1:
            resolved_vars = ["x"]
        elif n == 2:
            resolved_vars = ["x", "y"]
        elif n == 3:
            resolved_vars = ["x", "y", "z"]
        else:
            resolved_vars = [f"x{i}" for i in range(n)]
    else:
        if len(var_names) != n:
            raise ValueError(f"var_names length ({len(var_names)}) must match coef columns ({n})")
        resolved_vars = list(var_names)

    equations: List[str] = []

    for k in range(n):
        var = resolved_vars[k]
        lhs = f"d{var}/dt ="
        terms = []

        for j in range(p):
            c = coef_arr[j, k]
            if np.abs(c) < cutoff:
                continue

            feat = feature_names[j]
            mag_str = f"{np.abs(c):.{precision}f}"
            term_str = mag_str if feat == "1" else f"{mag_str} {feat}"
            sign = "-" if c < 0.0 else "+"
            terms.append((sign, term_str))

        if not terms:
            eq = f"{lhs} 0"
        else:
            parts = []
            for idx, (sign, term_str) in enumerate(terms):
                if idx == 0:
                    parts.append(f"-{term_str}" if sign == "-" else term_str)
                else:
                    parts.append(f"{sign} {term_str}")
            eq = f"{lhs} " + " ".join(parts)

        equations.append(eq)

    return equations

