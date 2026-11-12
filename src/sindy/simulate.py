"""Numerical simulation, Runge-Kutta 4th-order integration, dataset generation, and noise corruption.

All integration operations are pure NumPy, stateless, and vectorized.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy as np

from sindy.systems import DynamicalSystem, SYSTEMS


def rk4_step(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Perform a single step of the classic 4th-order Runge-Kutta scheme.

    Formula:
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x_{n+1} = x_n + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    Parameters
    ----------
    f : Callable[[np.ndarray], np.ndarray]
        Vector field function returning dx/dt for state x.
    x : np.ndarray
        Current state array of shape (n,) or (m, n).
    dt : float
        Integration time step.

    Returns
    -------
    np.ndarray
        Integrated state at next time step, same shape as x: (n,) or (m, n).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    k1 = f(x_arr)
    k2 = f(x_arr + 0.5 * dt * k1)
    k3 = f(x_arr + 0.5 * dt * k2)
    k4 = f(x_arr + dt * k3)
    return x_arr + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(
    f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    t_end: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate an ODE system forward in time using fixed-step RK4.

    Parameters
    ----------
    f : Callable[[np.ndarray], np.ndarray]
        Vector field function mapping state x -> dx/dt.
    x0 : np.ndarray
        Initial state vector of shape (n,).
    t_end : float
        Total integration duration (t in [0, t_end]).
    dt : float
        Integration step size.

    Returns
    -------
    t : np.ndarray
        Monotonically increasing time grid of shape (m,).
    X : np.ndarray
        Integrated trajectory matrix of shape (m, n) where row i is x(t_i).
    """
    if dt <= 0.0:
        raise ValueError(f"Integration step dt must be positive, got {dt}")
    if t_end < 0.0:
        raise ValueError(f"Integration duration t_end must be non-negative, got {t_end}")

    x0_arr = np.asarray(x0, dtype=np.float64)
    if x0_arr.ndim != 1:
        raise ValueError(f"Initial condition x0 must be a 1D vector of shape (n,), got {x0_arr.shape}")

    n_steps = int(np.round(t_end / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    n_vars = x0_arr.size

    X = np.empty((n_steps + 1, n_vars), dtype=np.float64)
    X[0] = x0_arr

    for i in range(n_steps):
        X[i + 1] = rk4_step(f, X[i], dt)

    return t, X


def generate_dataset(
    system: Union[str, DynamicalSystem, Callable[[np.ndarray], np.ndarray]],
    n_traj: int,
    t_end: float,
    dt: float,
    seed: Optional[int] = None,
    ic_bounds: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate a multi-trajectory dataset with exact oracle derivatives.

    Parameters
    ----------
    system : Union[str, DynamicalSystem, Callable[[np.ndarray], np.ndarray]]
        Dynamical system instance, registered name key (e.g. 'lorenz'),
        or arbitrary vector field callable f(x).
    n_traj : int
        Number of distinct trajectories to simulate from randomized initial conditions.
    t_end : float
        End time for each trajectory.
    dt : float
        Time step size.
    seed : Optional[int], default=None
        Random seed ensuring reproducibility.
    ic_bounds : Optional[Sequence[Tuple[float, float]]], default=None
        Per-variable (min, max) sampling intervals for initial conditions.
        Defaults to system bounds if available, or [-2.0, 2.0] otherwise.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        List of tuples: (t, X, dXdt_true), each containing:
        - t: time grid of shape (m,)
        - X: state trajectory of shape (m, n)
        - dXdt_true: exact oracle derivatives of shape (m, n) evaluated via f(X).
    """
    if n_traj < 1:
        raise ValueError(f"n_traj must be >= 1, got {n_traj}")

    rng = np.random.default_rng(seed)

    # Resolve vector field function and sampling bounds
    if isinstance(system, str):
        if system not in SYSTEMS:
            raise KeyError(f"System '{system}' not found in registry: {list(SYSTEMS.keys())}")
        sys_obj = SYSTEMS[system]
        f_eval: Callable[[np.ndarray], np.ndarray] = lambda x: sys_obj.f(x)
        bounds = ic_bounds if ic_bounds is not None else sys_obj.default_ic_bounds
        dim = sys_obj.dimension
    elif isinstance(system, DynamicalSystem):
        f_eval = lambda x: system.f(x)
        bounds = ic_bounds if ic_bounds is not None else system.default_ic_bounds
        dim = system.dimension
    elif callable(system):
        f_eval = system
        if ic_bounds is not None:
            bounds = tuple(ic_bounds)
            dim = len(bounds)
        else:
            # Probe dimension by calling f on a test point
            test_x = np.array([1.0, 1.0], dtype=np.float64)
            try:
                out = system(test_x)
                dim = out.size
            except Exception:
                dim = 3
            bounds = tuple((-2.0, 2.0) for _ in range(dim))
    else:
        raise TypeError(f"Unsupported system type: {type(system)}")

    lows = np.array([b[0] for b in bounds], dtype=np.float64)
    highs = np.array([b[1] for b in bounds], dtype=np.float64)

    dataset: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for _ in range(n_traj):
        # Sample randomized initial state
        x0 = rng.uniform(lows, highs)

        # Forward integration
        t, X = simulate(f_eval, x0, t_end=t_end, dt=dt)

        # Compute oracle derivatives: vectorized across m points, fallback to loop if needed
        try:
            dXdt_true = f_eval(X)
            if dXdt_true.shape != X.shape:
                raise ValueError
        except Exception:
            dXdt_true = np.empty_like(X)
            for j in range(len(X)):
                dXdt_true[j] = f_eval(X[j])

        dataset.append((t, X, dXdt_true))

    return dataset


def add_noise(
    X: np.ndarray,
    noise_level: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add Gaussian measurement noise scaled to each variable's standard deviation.

    For each column j:
        std_j = std(X[:, j])
        noise[:, j] ~ Normal(0, (noise_level * std_j)^2)
        X_noisy[:, j] = X[:, j] + noise[:, j]

    Parameters
    ----------
    X : np.ndarray
        Clean trajectory matrix of shape (m, n).
    noise_level : float
        Noise magnitude fraction (e.g., 0.05 corresponds to 5% noise).
    seed : Optional[int], default=None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy trajectory matrix of shape (m, n).
    """
    if noise_level < 0.0:
        raise ValueError(f"noise_level must be non-negative, got {noise_level}")
    if noise_level == 0.0:
        return np.copy(X)

    X_arr = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)

    std_per_var = np.std(X_arr, axis=0, ddof=0)
    # If a variable is completely constant, avoid zero scale or scale to 1.0
    scale = np.where(std_per_var == 0.0, 1.0, std_per_var) * noise_level

    noise = rng.normal(loc=0.0, scale=scale, size=X_arr.shape)
    return X_arr + noise

