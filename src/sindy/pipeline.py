"""End-to-end SINDy equation discovery pipeline.

Encapsulates numerical differentiation, candidate library generation,
sparse regression via STLSQ, and forward model trajectory forecasting.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from sindy.derivatives import central_diff, savgol_diff, tv_diff
from sindy.library import FeatureLibrary
from sindy.simulate import simulate
from sindy.stlsq import STLSQ, pretty_print_equations


class SINDy:
    """Sparse Identification of Nonlinear Dynamics (SINDy) pipeline.

    Discovers parsimonious differential equations from state trajectories:
        dx/dt = Theta(x) * Xi

    Parameters
    ----------
    poly_degree : int, default=2
        Maximum polynomial degree for candidate basis monomials.
    include_bias : bool, default=True
        Whether to include the constant term '1' in Theta.
    trig : bool, default=False
        Whether to include trigonometric basis functions sin(x), cos(x).
    var_names : Optional[Sequence[str]], default=None
        Custom symbolic variable names (e.g. ['x', 'y', 'z']).
    diff_method : str, default="central"
        Numerical differentiation scheme: 'central', 'savgol', or 'tv'.
    diff_kwargs : Optional[Dict[str, Any]], default=None
        Optional keyword arguments passed to the differentiation routine.
    threshold : float, default=0.1
        Sparsity cutoff parameter lambda for STLSQ.
    max_iter : int, default=20
        Maximum threshold-refit iterations.
    normalize : bool, default=True
        Whether to internally normalize columns of Theta during STLSQ regression.

    Attributes
    ----------
    library_ : FeatureLibrary
        Fitted candidate feature library.
    optimizer_ : STLSQ
        Fitted STLSQ sparse optimizer.
    coef_ : np.ndarray
        Learned sparse coefficient matrix Xi of shape (p, n).
    feature_names_ : List[str]
        List of p candidate basis feature names.
    var_names_ : List[str]
        List of n state variable names.
    dt_ : Optional[float]
        Time step used during fitting.
    """

    def __init__(
        self,
        poly_degree: int = 2,
        include_bias: bool = True,
        trig: bool = False,
        var_names: Optional[Sequence[str]] = None,
        diff_method: str = "central",
        diff_kwargs: Optional[Dict[str, Any]] = None,
        threshold: float = 0.1,
        max_iter: int = 20,
        normalize: bool = True,
    ) -> None:
        self.poly_degree = poly_degree
        self.include_bias = include_bias
        self.trig = trig
        self.var_names = list(var_names) if var_names is not None else None
        self.diff_method = diff_method.lower()
        self.diff_kwargs = diff_kwargs if diff_kwargs is not None else {}
        self.threshold = threshold
        self.max_iter = max_iter
        self.normalize = normalize

        self.library_: Optional[FeatureLibrary] = None
        self.optimizer_: Optional[STLSQ] = None
        self.coef_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []
        self.var_names_: List[str] = []
        self.dt_: Optional[float] = None

    def fit(
        self,
        X_list: Union[np.ndarray, Sequence[np.ndarray]],
        dt: float,
        dXdt_list: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    ) -> SINDy:
        """Fit the governing dynamical equations from trajectory data.

        Stacks multiple trajectories vertically before feature expansion and regression.

        Parameters
        ----------
        X_list : Union[np.ndarray, Sequence[np.ndarray]]
            Single trajectory matrix of shape (m, n) or list of trajectory matrices
            [(m_1, n), (m_2, n), ...].
        dt : float
            Sampling time interval.
        dXdt_list : Optional[Union[np.ndarray, Sequence[np.ndarray]]], default=None
            Precomputed or oracle time derivatives. If None, derivatives are computed
            using the specified diff_method.

        Returns
        -------
        SINDy
            Fitted estimator instance.
        """
        if dt <= 0.0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
        self.dt_ = dt

        # Standardize input trajectories to a list of 2D arrays
        if isinstance(X_list, np.ndarray):
            trajectories = [X_list]
        else:
            trajectories = list(X_list)

        # Compute or collect derivatives
        if dXdt_list is not None:
            if isinstance(dXdt_list, np.ndarray):
                derivatives = [dXdt_list]
            else:
                derivatives = list(dXdt_list)
        else:
            derivatives = [self._compute_derivative(traj, dt) for traj in trajectories]

        if len(trajectories) != len(derivatives):
            raise ValueError(
                f"Count mismatch: {len(trajectories)} trajectories vs {len(derivatives)} derivative arrays"
            )

        # Stack multiple trajectories vertically
        X_stacked = np.vstack(trajectories)
        dXdt_stacked = np.vstack(derivatives)

        # 1. Build and transform candidate library Theta(X)
        self.library_ = FeatureLibrary(
            poly_degree=self.poly_degree,
            include_bias=self.include_bias,
            trig=self.trig,
            var_names=self.var_names,
            normalize=False,
        )
        Theta = self.library_.fit_transform(X_stacked)
        self.feature_names_ = self.library_.get_feature_names()
        self.var_names_ = self.library_.resolved_var_names_

        # 2. Fit sparse coefficient matrix via STLSQ
        self.optimizer_ = STLSQ(
            threshold=self.threshold,
            max_iter=self.max_iter,
            normalize=self.normalize,
        )
        self.optimizer_.fit(Theta, dXdt_stacked)
        self.coef_ = self.optimizer_.coef_

        return self

    def vector_field(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the discovered continuous vector field dx/dt = Theta(x) * Xi.

        Parameters
        ----------
        x : np.ndarray
            State vector of shape (n,) or batch of shape (m, n).

        Returns
        -------
        np.ndarray
            Computed time derivative dx/dt of identical shape.
        """
        if self.library_ is None or self.coef_ is None:
            raise ValueError("SINDy model is not fitted. Call fit() first.")

        x_arr = np.asarray(x, dtype=np.float64)
        is_1d = (x_arr.ndim == 1)
        x_2d = np.atleast_2d(x_arr)

        Theta_x = self.library_.transform(x_2d)
        dxdt = Theta_x @ self.coef_

        return dxdt[0] if is_1d else dxdt

    def predict(
        self,
        x0: np.ndarray,
        t_end: float,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward-simulate the discovered ODE model using classic 4th-order Runge-Kutta.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state vector of shape (n,).
        t_end : float
            Simulation duration.
        dt : Optional[float], default=None
            Integration time step. Defaults to fitting time step dt_.

        Returns
        -------
        t : np.ndarray
            Time array of shape (m,).
        X_sim : np.ndarray
            Integrated trajectory matrix of shape (m, n).
        """
        if self.coef_ is None:
            raise ValueError("SINDy model is not fitted. Call fit() first.")

        step = self.dt_ if dt is None else dt
        if step is None or step <= 0.0:
            raise ValueError("A positive integration time step dt must be provided.")

        return simulate(self.vector_field, x0, t_end=t_end, dt=step)

    def print_equations(
        self,
        precision: int = 3,
        cutoff: float = 1e-12,
    ) -> List[str]:
        """Print and return discovered mathematical equations.

        Parameters
        ----------
        precision : int, default=3
            Decimal digits for coefficients.
        cutoff : float, default=1e-12
            Threshold below which coefficients are omitted.

        Returns
        -------
        List[str]
            Discovered equation strings.
        """
        if self.coef_ is None:
            raise ValueError("SINDy model is not fitted. Call fit() first.")

        eqs = pretty_print_equations(
            self.coef_,
            self.feature_names_,
            self.var_names_,
            precision=precision,
            cutoff=cutoff,
        )
        for eq in eqs:
            print(eq)
        return eqs

    def _compute_derivative(self, X: np.ndarray, dt: float) -> np.ndarray:
        """Dispatch numerical differentiation based on diff_method."""
        if self.diff_method == "central":
            return central_diff(X, dt)
        elif self.diff_method == "savgol":
            return savgol_diff(X, dt, **self.diff_kwargs)
        elif self.diff_method == "tv":
            return tv_diff(X, dt, **self.diff_kwargs)
        else:
            raise ValueError(
                f"Unknown diff_method '{self.diff_method}'. Expected 'central', 'savgol', or 'tv'."
            )

