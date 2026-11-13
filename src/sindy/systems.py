"""Dynamical systems definitions, vector fields, and true sparse coefficient layouts.

Each system is implemented as a vectorized vector field function:
    f(x, params) -> dx/dt
where `x` has shape (n,) or (m, n), and returns an array of identical shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple
import numpy as np


# ============================================================================
# True Sparse Coefficient Dictionaries (Layout matching Xi later)
# ============================================================================

LORENZ_TRUE_COEFFICIENTS: Dict[str, Dict[str, float]] = {
    "x": {"x": -10.0, "y": 10.0},
    "y": {"x": 28.0, "y": -1.0, "x z": -1.0},
    "z": {"z": -8.0 / 3.0, "x y": 1.0},
}

LOTKA_VOLTERRA_TRUE_COEFFICIENTS: Dict[str, Dict[str, float]] = {
    "u": {"u": 1.0, "u v": -0.1},
    "v": {"v": -1.5, "u v": 0.075},
}

VAN_DER_POL_TRUE_COEFFICIENTS: Dict[str, Dict[str, float]] = {
    "x": {"y": 1.0},
    "y": {"x": -1.0, "y": 2.0, "x^2 y": -2.0},
}

DAMPED_PENDULUM_TRUE_COEFFICIENTS: Dict[str, Dict[str, float]] = {
    "theta": {"omega": 1.0},
    "omega": {"omega": -0.25, "sin(theta)": -5.0},
}

ROSSLER_TRUE_COEFFICIENTS: Dict[str, Dict[str, float]] = {
    "x": {"y": -1.0, "z": -1.0},
    "y": {"x": 1.0, "y": 0.2},
    "z": {"1": 0.2, "z": -5.7, "x z": 1.0},
}


# ============================================================================
# Vector Field Implementations
# ============================================================================

def lorenz(
    x: np.ndarray,
    params: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Compute the time derivative for the Lorenz-63 chaotic system.

    Governing equations:
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

    Parameters
    ----------
    x : np.ndarray
        State array of shape (3,) for a single state or (m, 3) for m points.
    params : Optional[Mapping[str, float]], default=None
        System parameters. Defaults to {'sigma': 10.0, 'rho': 28.0, 'beta': 8/3}.

    Returns
    -------
    np.ndarray
        Time derivative dx/dt of shape (3,) or (m, 3).
    """
    p = {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}
    if params is not None:
        p.update(params)

    sigma, rho, beta = p["sigma"], p["rho"], p["beta"]

    x_arr = np.asarray(x, dtype=np.float64)
    is_1d = (x_arr.ndim == 1)
    x_2d = np.atleast_2d(x_arr)

    if x_2d.shape[1] != 3:
        raise ValueError(f"Lorenz system requires state dimension 3, got shape {x_arr.shape}")

    x_val = x_2d[:, 0]
    y_val = x_2d[:, 1]
    z_val = x_2d[:, 2]

    dxdt = np.empty_like(x_2d)
    dxdt[:, 0] = sigma * (y_val - x_val)
    dxdt[:, 1] = x_val * (rho - z_val) - y_val
    dxdt[:, 2] = x_val * y_val - beta * z_val

    return dxdt[0] if is_1d else dxdt


def lotka_volterra(
    x: np.ndarray,
    params: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Compute the time derivative for the Lotka-Volterra predator-prey system.

    Governing equations:
        du/dt = alpha * u - beta * u * v
        dv/dt = delta * u * v - gamma * v

    Parameters
    ----------
    x : np.ndarray
        State array of shape (2,) for a single state or (m, 2) for m points.
        x[:, 0] represents prey u, x[:, 1] represents predator v.
    params : Optional[Mapping[str, float]], default=None
        System parameters. Defaults to:
        {'alpha': 1.0, 'beta': 0.1, 'delta': 0.075, 'gamma': 1.5}.

    Returns
    -------
    np.ndarray
        Time derivative dx/dt of shape (2,) or (m, 2).
    """
    p = {"alpha": 1.0, "beta": 0.1, "delta": 0.075, "gamma": 1.5}
    if params is not None:
        p.update(params)

    alpha, beta, delta, gamma = p["alpha"], p["beta"], p["delta"], p["gamma"]

    x_arr = np.asarray(x, dtype=np.float64)
    is_1d = (x_arr.ndim == 1)
    x_2d = np.atleast_2d(x_arr)

    if x_2d.shape[1] != 2:
        raise ValueError(f"Lotka-Volterra requires state dimension 2, got shape {x_arr.shape}")

    u_val = x_2d[:, 0]
    v_val = x_2d[:, 1]

    dxdt = np.empty_like(x_2d)
    dxdt[:, 0] = alpha * u_val - beta * u_val * v_val
    dxdt[:, 1] = delta * u_val * v_val - gamma * v_val

    return dxdt[0] if is_1d else dxdt


def van_der_pol(
    x: np.ndarray,
    params: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Compute the time derivative for the Van der Pol oscillator (mu=2).

    Governing equations:
        dx/dt = y
        dy/dt = mu * (1 - x^2) * y - x = -x + mu * y - mu * x^2 * y

    Parameters
    ----------
    x : np.ndarray
        State array of shape (2,) for a single state or (m, 2) for m points.
        x[:, 0] is displacement x, x[:, 1] is velocity y = dx/dt.
    params : Optional[Mapping[str, float]], default=None
        System parameters. Defaults to {'mu': 2.0}.

    Returns
    -------
    np.ndarray
        Time derivative dx/dt of shape (2,) or (m, 2).
    """
    p = {"mu": 2.0}
    if params is not None:
        p.update(params)

    mu = p["mu"]

    x_arr = np.asarray(x, dtype=np.float64)
    is_1d = (x_arr.ndim == 1)
    x_2d = np.atleast_2d(x_arr)

    if x_2d.shape[1] != 2:
        raise ValueError(f"Van der Pol requires state dimension 2, got shape {x_arr.shape}")

    x_val = x_2d[:, 0]
    y_val = x_2d[:, 1]

    dxdt = np.empty_like(x_2d)
    dxdt[:, 0] = y_val
    dxdt[:, 1] = mu * (1.0 - x_val**2) * y_val - x_val

    return dxdt[0] if is_1d else dxdt


def damped_pendulum(
    x: np.ndarray,
    params: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Compute the time derivative for the nonlinear damped pendulum.

    Governing equations:
        dtheta/dt = omega
        domega/dt = -b * omega - c * sin(theta)

    Parameters
    ----------
    x : np.ndarray
        State array of shape (2,) for a single state or (m, 2) for m points.
        x[:, 0] is angle theta (rad), x[:, 1] is angular velocity omega (rad/s).
    params : Optional[Mapping[str, float]], default=None
        System parameters. Defaults to {'b': 0.25, 'c': 5.0}.

    Returns
    -------
    np.ndarray
        Time derivative dx/dt of shape (2,) or (m, 2).
    """
    p = {"b": 0.25, "c": 5.0}
    if params is not None:
        p.update(params)

    b, c = p["b"], p["c"]

    x_arr = np.asarray(x, dtype=np.float64)
    is_1d = (x_arr.ndim == 1)
    x_2d = np.atleast_2d(x_arr)

    if x_2d.shape[1] != 2:
        raise ValueError(f"Damped pendulum requires state dimension 2, got shape {x_arr.shape}")

    theta = x_2d[:, 0]
    omega = x_2d[:, 1]

    dxdt = np.empty_like(x_2d)
    dxdt[:, 0] = omega
    dxdt[:, 1] = -b * omega - c * np.sin(theta)

    return dxdt[0] if is_1d else dxdt


def rossler(
    x: np.ndarray,
    params: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Compute the time derivative for the Rössler chaotic attractor.

    Governing equations:
        dx/dt = -y - z
        dy/dt = x + a * y
        dz/dt = b + z * (x - c) = b - c * z + x * z

    Parameters
    ----------
    x : np.ndarray
        State array of shape (3,) or (m, 3).
    params : Optional[Mapping[str, float]], default=None
        System parameters. Defaults to {'a': 0.2, 'b': 0.2, 'c': 5.7}.

    Returns
    -------
    np.ndarray
        Time derivative dx/dt of shape (3,) or (m, 3).
    """
    p = {"a": 0.2, "b": 0.2, "c": 5.7}
    if params is not None:
        p.update(params)

    a, b, c = p["a"], p["b"], p["c"]

    x_arr = np.asarray(x, dtype=np.float64)
    is_1d = (x_arr.ndim == 1)
    x_2d = np.atleast_2d(x_arr)

    if x_2d.shape[1] != 3:
        raise ValueError(f"Rössler system requires state dimension 3, got shape {x_arr.shape}")

    x_val = x_2d[:, 0]
    y_val = x_2d[:, 1]
    z_val = x_2d[:, 2]

    dxdt = np.empty_like(x_2d)
    dxdt[:, 0] = -y_val - z_val
    dxdt[:, 1] = x_val + a * y_val
    dxdt[:, 2] = b + z_val * (x_val - c)

    return dxdt[0] if is_1d else dxdt


# ============================================================================
# System Specification Dataclass & Registry
# ============================================================================

@dataclass(frozen=True)
class DynamicalSystem:
    """Container encapsulating a dynamical system and its metadata."""

    name: str
    f: Callable[..., np.ndarray]
    state_names: List[str]
    default_params: Dict[str, float]
    true_coefficients: Dict[str, Dict[str, float]]
    default_ic_bounds: Tuple[Tuple[float, float], ...] = field(default_factory=tuple)

    @property
    def dimension(self) -> int:
        """Return state dimension n."""
        return len(self.state_names)

    def __call__(
        self,
        x: np.ndarray,
        params: Optional[Mapping[str, float]] = None,
    ) -> np.ndarray:
        """Evaluate the system vector field."""
        return self.f(x, params)


SYSTEMS: Dict[str, DynamicalSystem] = {
    "lorenz": DynamicalSystem(
        name="Lorenz-63",
        f=lorenz,
        state_names=["x", "y", "z"],
        default_params={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        true_coefficients=LORENZ_TRUE_COEFFICIENTS,
        default_ic_bounds=((-15.0, 15.0), (-15.0, 15.0), (10.0, 40.0)),
    ),
    "lotka_volterra": DynamicalSystem(
        name="Lotka-Volterra",
        f=lotka_volterra,
        state_names=["u", "v"],
        default_params={"alpha": 1.0, "beta": 0.1, "delta": 0.075, "gamma": 1.5},
        true_coefficients=LOTKA_VOLTERRA_TRUE_COEFFICIENTS,
        default_ic_bounds=((5.0, 25.0), (2.0, 15.0)),
    ),
    "van_der_pol": DynamicalSystem(
        name="Van der Pol",
        f=van_der_pol,
        state_names=["x", "y"],
        default_params={"mu": 2.0},
        true_coefficients=VAN_DER_POL_TRUE_COEFFICIENTS,
        default_ic_bounds=((-2.5, 2.5), (-2.5, 2.5)),
    ),
    "damped_pendulum": DynamicalSystem(
        name="Damped Pendulum",
        f=damped_pendulum,
        state_names=["theta", "omega"],
        default_params={"b": 0.25, "c": 5.0},
        true_coefficients=DAMPED_PENDULUM_TRUE_COEFFICIENTS,
        default_ic_bounds=((-2.5, 2.5), (-2.0, 2.0)),
    ),
    "rossler": DynamicalSystem(
        name="Rossler",
        f=rossler,
        state_names=["x", "y", "z"],
        default_params={"a": 0.2, "b": 0.2, "c": 5.7},
        true_coefficients=ROSSLER_TRUE_COEFFICIENTS,
        default_ic_bounds=((-10.0, 10.0), (-10.0, 10.0), (0.0, 20.0)),
    ),
}

