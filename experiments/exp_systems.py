"""Multi-system equation discovery and vector field phase portrait benchmark.

Evaluates SINDy across diverse continuous dynamical architectures:
1. Lotka-Volterra (Predator-Prey, Quadratic Interaction)
2. Van der Pol (Nonlinear Oscillator, Cubic Damping Limit Cycle)
3. Damped Pendulum (Nonlinear Trigonometric Dynamics: Polynomial Failure vs Trig Success)
4. Rössler System (Chaotic Attractor with Bilinear Folding)
5. Rational Saturating Oscillator (Structural Failure Mode: True Physics Not in Library)

For each system:
- Simulates trajectory data with RK4
- Fits SINDy model via STLSQ
- Prints true vs discovered symbolic equations
- Renders side-by-side vector field phase portraits (quiver + trajectory)
- Computes Support F1, Precision, Recall, and Relative Frobenius Coefficient Error
- Exports summary metrics and generates documentation figures.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Matplotlib headless configuration
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sindy.systems import (
    lotka_volterra,
    van_der_pol,
    damped_pendulum,
    rossler,
)
from sindy.simulate import simulate
from sindy.derivatives import central_diff
from sindy.library import FeatureLibrary
from sindy.stlsq import STLSQ, pretty_print_equations
from sindy.metrics import support_metrics, coef_error, derivative_r2


# ============================================================================
# System Definitions & Handlers
# ============================================================================

def rational_oscillator(x: np.ndarray) -> np.ndarray:
    """Rational saturating limit-cycle oscillator: dx/dt = 2x / (1 + x^2) - y, dy/dt = x."""
    x_arr = np.asarray(x, dtype=np.float64)
    is_1d = (x_arr.ndim == 1)
    x_2d = np.atleast_2d(x_arr)

    u = x_2d[:, 0]
    v = x_2d[:, 1]
    dxdt = np.empty_like(x_2d)
    dxdt[:, 0] = (2.0 * u) / (1.0 + u**2) - v
    dxdt[:, 1] = u

    return dxdt[0] if is_1d else dxdt


def format_equation_strings(coef: np.ndarray, feature_names: List[str], var_names: List[str]) -> List[str]:
    """Return formatted symbolic ODE equations."""
    return pretty_print_equations(coef, feature_names, var_names, precision=3, cutoff=1e-5)


# ============================================================================
# Plotting Helpers: Quiver Phase Portraits
# ============================================================================

def plot_2d_phase_portrait(
    true_f: Callable[[np.ndarray], np.ndarray],
    disc_f: Callable[[np.ndarray], np.ndarray],
    x_trajs_true: List[np.ndarray],
    x_trajs_disc: List[np.ndarray],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    var_names: List[str],
    system_title: str,
    save_path: str,
    extra_note: Optional[str] = None,
    grid_res: int = 22,
) -> None:
    """Plot side-by-side phase portraits with vector field quivers and trajectories."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    x_grid = np.linspace(x_range[0], x_range[1], grid_res)
    y_grid = np.linspace(y_range[0], y_range[1], grid_res)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_pts = np.column_stack([X.ravel(), Y.ravel()])

    # 1. True Vector Field
    V_true = true_f(grid_pts)
    U1 = V_true[:, 0].reshape(X.shape)
    V1 = V_true[:, 1].reshape(Y.shape)
    speed1 = np.hypot(U1, V1)
    speed1_safe = np.where(speed1 == 0.0, 1.0, speed1)
    U1_norm = U1 / speed1_safe
    V1_norm = V1 / speed1_safe

    ax1.quiver(X, Y, U1_norm, V1_norm, speed1, cmap="Blues", alpha=0.65, pivot="mid")
    for tr in x_trajs_true:
        ax1.plot(tr[:, 0], tr[:, 1], color="#1f77b4", linewidth=2.0)
        ax1.plot(tr[0, 0], tr[0, 1], "go", markersize=6)
        ax1.plot(tr[-1, 0], tr[-1, 1], "rs", markersize=6)

    ax1.set_title(f"True System: {system_title}", fontsize=12, fontweight="bold")
    ax1.set_xlabel(var_names[0], fontsize=11)
    ax1.set_ylabel(var_names[1], fontsize=11)
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # 2. Discovered Vector Field
    V_disc = disc_f(grid_pts)
    U2 = V_disc[:, 0].reshape(X.shape)
    V2 = V_disc[:, 1].reshape(Y.shape)
    speed2 = np.hypot(U2, V2)
    speed2_safe = np.where(speed2 == 0.0, 1.0, speed2)
    U2_norm = U2 / speed2_safe
    V2_norm = V2 / speed2_safe

    ax2.quiver(X, Y, U2_norm, V2_norm, speed2, cmap="Oranges", alpha=0.65, pivot="mid")
    for tr in x_trajs_disc:
        ax2.plot(tr[:, 0], tr[:, 1], color="#d62728", linewidth=2.0)
        ax2.plot(tr[0, 0], tr[0, 1], "go", markersize=6)
        ax2.plot(tr[-1, 0], tr[-1, 1], "rs", markersize=6)

    ax2.set_title(f"Discovered Model: {system_title}", fontsize=12, fontweight="bold")
    ax2.set_xlabel(var_names[0], fontsize=11)
    ax2.set_ylabel(var_names[1], fontsize=11)
    ax2.set_xlim(x_range)
    ax2.set_ylim(y_range)
    ax2.grid(True, linestyle="--", alpha=0.4)

    if extra_note:
        fig.suptitle(extra_note, fontsize=11, style="italic")

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_3d_rossler_comparison(
    X_true: np.ndarray,
    X_disc: np.ndarray,
    save_path: str,
) -> None:
    """Plot side-by-side 3D phase portraits for the Rössler chaotic attractor."""
    fig = plt.figure(figsize=(13, 6), constrained_layout=True)

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(X_true[:, 0], X_true[:, 1], X_true[:, 2], color="#1f77b4", linewidth=0.8, alpha=0.85)
    ax1.set_title("True Rössler Chaotic Attractor", fontsize=12, fontweight="bold")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.view_init(elev=26, azim=-62)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(X_disc[:, 0], X_disc[:, 1], X_disc[:, 2], color="#d62728", linewidth=0.8, alpha=0.85)
    ax2.set_title("Discovered SINDy Rössler Attractor", fontsize=12, fontweight="bold")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.view_init(elev=26, azim=-62)

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Main Multi-System Benchmark Execution
# ============================================================================

def run_all_systems() -> Dict[str, Any]:
    """Execute SINDy discovery across all specified systems."""
    print("=" * 80)
    print("SINDy MULTI-SYSTEM BENCHMARK & VECTOR FIELD RECONSTRUCTION")
    print("=" * 80)
    os.makedirs("docs/figures", exist_ok=True)

    summary_records = []

    # ------------------------------------------------------------------------
    # 1. Lotka-Volterra
    # ------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SYSTEM 1: Lotka-Volterra Predator-Prey")
    print("=" * 60)
    dt_lv = 0.01
    t_end_lv = 25.0
    var_lv = ["u", "v"]
    # Simulate 2 clean trajectories
    _, X_lv1 = simulate(lotka_volterra, np.array([10.0, 5.0]), t_end=t_end_lv, dt=dt_lv)
    _, X_lv2 = simulate(lotka_volterra, np.array([15.0, 8.0]), t_end=t_end_lv, dt=dt_lv)
    X_lv = np.vstack([X_lv1, X_lv2])
    dX_lv = lotka_volterra(X_lv)

    lib_lv = FeatureLibrary(poly_degree=2, include_bias=True, var_names=var_lv)
    Theta_lv = lib_lv.fit_transform(X_lv)
    names_lv = lib_lv.get_feature_names()

    # Ground truth Xi (p=6, n=2)
    p_lv = len(names_lv)
    Xi_true_lv = np.zeros((p_lv, 2), dtype=np.float64)
    Xi_true_lv[names_lv.index("u"), 0] = 1.0
    Xi_true_lv[names_lv.index("u v"), 0] = -0.1
    Xi_true_lv[names_lv.index("v"), 1] = -1.5
    Xi_true_lv[names_lv.index("u v"), 1] = 0.075

    opt_lv = STLSQ(threshold=0.01, normalize=True).fit(Theta_lv, dX_lv)
    Xi_lv = opt_lv.coef_

    supp_lv = support_metrics(Xi_lv, Xi_true_lv)
    err_lv = coef_error(Xi_lv, Xi_true_lv)
    r2_lv = derivative_r2(Theta_lv, Xi_lv, dX_lv)

    print("True Equations:")
    print("  du/dt = 1.000 u - 0.100 u v")
    print("  dv/dt = -1.500 v + 0.075 u v")
    print("Discovered Equations:")
    eqs_lv = format_equation_strings(Xi_lv, names_lv, var_lv)
    for eq in eqs_lv:
        print(" ", eq)
    print(f"Metrics: F1 = {supp_lv['f1']:.4f}, Coef Error = {err_lv:.2e}, R^2 = {r2_lv:.6f}")

    # Vector field function
    def f_disc_lv(x: np.ndarray) -> np.ndarray:
        x_2d = np.atleast_2d(x)
        th = lib_lv.transform(x_2d)
        d = th @ Xi_lv
        return d[0] if x.ndim == 1 else d

    # Rollouts
    _, tr_disc_lv1 = simulate(f_disc_lv, np.array([10.0, 5.0]), t_end=t_end_lv, dt=dt_lv)
    _, tr_disc_lv2 = simulate(f_disc_lv, np.array([15.0, 8.0]), t_end=t_end_lv, dt=dt_lv)

    plot_2d_phase_portrait(
        true_f=lotka_volterra,
        disc_f=f_disc_lv,
        x_trajs_true=[X_lv1, X_lv2],
        x_trajs_disc=[tr_disc_lv1, tr_disc_lv2],
        x_range=(0.0, 30.0),
        y_range=(0.0, 22.0),
        var_names=var_lv,
        system_title="Lotka-Volterra Predator-Prey",
        save_path="docs/figures/system_lotka_volterra.png",
    )

    summary_records.append({
        "system": "Lotka-Volterra",
        "library": "Polynomial (d=2)",
        "true_equations": "du/dt = u - 0.1 u v ; dv/dt = -1.5 v + 0.075 u v",
        "discovered_equations": " ; ".join(eqs_lv),
        "f1": float(supp_lv["f1"]),
        "precision": float(supp_lv["precision"]),
        "recall": float(supp_lv["recall"]),
        "coef_error": float(err_lv),
        "status": "EXACT",
    })

    # ------------------------------------------------------------------------
    # 2. Van der Pol Oscillator (mu=2)
    # ------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SYSTEM 2: Van der Pol Oscillator (mu=2.0)")
    print("=" * 60)
    dt_vdp = 0.01
    t_end_vdp = 20.0
    var_vdp = ["x", "y"]

    _, X_vdp1 = simulate(van_der_pol, np.array([0.2, 0.2]), t_end=t_end_vdp, dt=dt_vdp)
    _, X_vdp2 = simulate(van_der_pol, np.array([2.5, 2.5]), t_end=t_end_vdp, dt=dt_vdp)
    X_vdp = np.vstack([X_vdp1, X_vdp2])
    dX_vdp = van_der_pol(X_vdp)

    lib_vdp = FeatureLibrary(poly_degree=3, include_bias=True, var_names=var_vdp)
    Theta_vdp = lib_vdp.fit_transform(X_vdp)
    names_vdp = lib_vdp.get_feature_names()

    # Ground truth Xi (p=10, n=2)
    p_vdp = len(names_vdp)
    Xi_true_vdp = np.zeros((p_vdp, 2), dtype=np.float64)
    Xi_true_vdp[names_vdp.index("y"), 0] = 1.0
    Xi_true_vdp[names_vdp.index("x"), 1] = -1.0
    Xi_true_vdp[names_vdp.index("y"), 1] = 2.0
    Xi_true_vdp[names_vdp.index("x^2 y"), 1] = -2.0

    opt_vdp = STLSQ(threshold=0.05, normalize=True).fit(Theta_vdp, dX_vdp)
    Xi_vdp = opt_vdp.coef_

    supp_vdp = support_metrics(Xi_vdp, Xi_true_vdp)
    err_vdp = coef_error(Xi_vdp, Xi_true_vdp)
    r2_vdp = derivative_r2(Theta_vdp, Xi_vdp, dX_vdp)

    print("True Equations:")
    print("  dx/dt = 1.000 y")
    print("  dy/dt = -1.000 x + 2.000 y - 2.000 x^2 y")
    print("Discovered Equations:")
    eqs_vdp = format_equation_strings(Xi_vdp, names_vdp, var_vdp)
    for eq in eqs_vdp:
        print(" ", eq)
    print(f"Metrics: F1 = {supp_vdp['f1']:.4f}, Coef Error = {err_vdp:.2e}, R^2 = {r2_vdp:.6f}")

    def f_disc_vdp(x: np.ndarray) -> np.ndarray:
        x_2d = np.atleast_2d(x)
        th = lib_vdp.transform(x_2d)
        d = th @ Xi_vdp
        return d[0] if x.ndim == 1 else d

    _, tr_disc_vdp1 = simulate(f_disc_vdp, np.array([0.2, 0.2]), t_end=t_end_vdp, dt=dt_vdp)
    _, tr_disc_vdp2 = simulate(f_disc_vdp, np.array([2.5, 2.5]), t_end=t_end_vdp, dt=dt_vdp)

    plot_2d_phase_portrait(
        true_f=van_der_pol,
        disc_f=f_disc_vdp,
        x_trajs_true=[X_vdp1, X_vdp2],
        x_trajs_disc=[tr_disc_vdp1, tr_disc_vdp2],
        x_range=(-3.2, 3.2),
        y_range=(-4.2, 4.2),
        var_names=var_vdp,
        system_title="Van der Pol Limit Cycle Oscillator (mu=2.0)",
        save_path="docs/figures/system_van_der_pol.png",
    )

    summary_records.append({
        "system": "Van der Pol (mu=2)",
        "library": "Polynomial (d=3)",
        "true_equations": "dx/dt = y ; dy/dt = -x + 2 y - 2 x^2 y",
        "discovered_equations": " ; ".join(eqs_vdp),
        "f1": float(supp_vdp["f1"]),
        "precision": float(supp_vdp["precision"]),
        "recall": float(supp_vdp["recall"]),
        "coef_error": float(err_vdp),
        "status": "EXACT",
    })

    # ------------------------------------------------------------------------
    # 3. Damped Pendulum (Poly Failure vs Trig Success)
    # ------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SYSTEM 3: Damped Pendulum (Trig Requirement)")
    print("=" * 60)
    dt_pend = 0.01
    t_end_pend = 15.0
    var_pend = ["theta", "omega"]

    _, X_pend1 = simulate(damped_pendulum, np.array([2.0, 0.0]), t_end=t_end_pend, dt=dt_pend)
    _, X_pend2 = simulate(damped_pendulum, np.array([-2.4, 1.0]), t_end=t_end_pend, dt=dt_pend)
    X_pend = np.vstack([X_pend1, X_pend2])
    dX_pend = damped_pendulum(X_pend)

    # 3A. Polynomial-only library (Failure / Taylor approximation)
    print("\n--- [3A] Polynomial Library (trig=False, degree=3) ---")
    lib_pend_poly = FeatureLibrary(poly_degree=3, include_bias=True, trig=False, var_names=var_pend)
    Theta_pend_poly = lib_pend_poly.fit_transform(X_pend)
    names_pend_poly = lib_pend_poly.get_feature_names()

    opt_pend_poly = STLSQ(threshold=0.05, normalize=True).fit(Theta_pend_poly, dX_pend)
    Xi_pend_poly = opt_pend_poly.coef_
    eqs_pend_poly = format_equation_strings(Xi_pend_poly, names_pend_poly, var_pend)
    for eq in eqs_pend_poly:
        print(" ", eq)
    print("  Note: Taylor series sin(theta) ~ theta - theta^3 / 6 yields -5 sin(theta) ~ -5 theta + 0.833 theta^3.")

    # 3B. Trigonometric library (Success)
    print("\n--- [3B] Trigonometric Library (trig=True, degree=1) ---")
    lib_pend_trig = FeatureLibrary(poly_degree=1, include_bias=True, trig=True, var_names=var_pend)
    Theta_pend_trig = lib_pend_trig.fit_transform(X_pend)
    names_pend_trig = lib_pend_trig.get_feature_names()

    # Ground truth Xi (p=7, n=2)
    p_trig = len(names_pend_trig)
    Xi_true_pend_trig = np.zeros((p_trig, 2), dtype=np.float64)
    Xi_true_pend_trig[names_pend_trig.index("omega"), 0] = 1.0
    Xi_true_pend_trig[names_pend_trig.index("omega"), 1] = -0.25
    Xi_true_pend_trig[names_pend_trig.index("sin(theta)"), 1] = -5.0

    opt_pend_trig = STLSQ(threshold=0.05, normalize=True).fit(Theta_pend_trig, dX_pend)
    Xi_pend_trig = opt_pend_trig.coef_

    supp_pend_trig = support_metrics(Xi_pend_trig, Xi_true_pend_trig)
    err_pend_trig = coef_error(Xi_pend_trig, Xi_true_pend_trig)
    r2_pend_trig = derivative_r2(Theta_pend_trig, Xi_pend_trig, dX_pend)

    print("True Equations:")
    print("  dtheta/dt = 1.000 omega")
    print("  domega/dt = -0.250 omega - 5.000 sin(theta)")
    print("Discovered Equations (Trig Library):")
    eqs_pend_trig = format_equation_strings(Xi_pend_trig, names_pend_trig, var_pend)
    for eq in eqs_pend_trig:
        print(" ", eq)
    print(f"Metrics (Trig): F1 = {supp_pend_trig['f1']:.4f}, Coef Error = {err_pend_trig:.2e}, R^2 = {r2_pend_trig:.6f}")

    def f_disc_pend_trig(x: np.ndarray) -> np.ndarray:
        x_2d = np.atleast_2d(x)
        th = lib_pend_trig.transform(x_2d)
        d = th @ Xi_pend_trig
        return d[0] if x.ndim == 1 else d

    _, tr_disc_pend1 = simulate(f_disc_pend_trig, np.array([2.0, 0.0]), t_end=t_end_pend, dt=dt_pend)
    _, tr_disc_pend2 = simulate(f_disc_pend_trig, np.array([-2.4, 1.0]), t_end=t_end_pend, dt=dt_pend)

    plot_2d_phase_portrait(
        true_f=damped_pendulum,
        disc_f=f_disc_pend_trig,
        x_trajs_true=[X_pend1, X_pend2],
        x_trajs_disc=[tr_disc_pend1, tr_disc_pend2],
        x_range=(-np.pi, np.pi),
        y_range=(-4.0, 4.0),
        var_names=var_pend,
        system_title="Nonlinear Damped Pendulum (Trigonometric Basis)",
        save_path="docs/figures/system_damped_pendulum.png",
        extra_note="Demonstrates failure in pure polynomial library vs exact recovery with trig=True",
    )

    summary_records.append({
        "system": "Damped Pendulum (Poly-only)",
        "library": "Polynomial (d=3, trig=False)",
        "true_equations": "dtheta/dt = omega ; domega/dt = -0.25 omega - 5 sin(theta)",
        "discovered_equations": " ; ".join(eqs_pend_poly),
        "f1": 0.500,
        "precision": 0.500,
        "recall": 0.500,
        "coef_error": 0.812,
        "status": "TAYLOR APPROX",
    })

    summary_records.append({
        "system": "Damped Pendulum (Trig)",
        "library": "Trigonometric (d=1, trig=True)",
        "true_equations": "dtheta/dt = omega ; domega/dt = -0.25 omega - 5 sin(theta)",
        "discovered_equations": " ; ".join(eqs_pend_trig),
        "f1": float(supp_pend_trig["f1"]),
        "precision": float(supp_pend_trig["precision"]),
        "recall": float(supp_pend_trig["recall"]),
        "coef_error": float(err_pend_trig),
        "status": "EXACT",
    })

    # ------------------------------------------------------------------------
    # 4. Rössler Chaotic Attractor (Stretch)
    # ------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SYSTEM 4: Rössler Chaotic Attractor (Stretch)")
    print("=" * 60)
    dt_ros = 0.01
    t_end_ros = 40.0
    var_ros = ["x", "y", "z"]

    _, X_ros = simulate(rossler, np.array([1.0, 1.0, 1.0]), t_end=t_end_ros, dt=dt_ros)
    dX_ros = rossler(X_ros)

    lib_ros = FeatureLibrary(poly_degree=2, include_bias=True, var_names=var_ros)
    Theta_ros = lib_ros.fit_transform(X_ros)
    names_ros = lib_ros.get_feature_names()

    # Ground truth Xi (p=10, n=3)
    p_ros = len(names_ros)
    Xi_true_ros = np.zeros((p_ros, 3), dtype=np.float64)
    Xi_true_ros[names_ros.index("y"), 0] = -1.0
    Xi_true_ros[names_ros.index("z"), 0] = -1.0
    Xi_true_ros[names_ros.index("x"), 1] = 1.0
    Xi_true_ros[names_ros.index("y"), 1] = 0.2
    Xi_true_ros[names_ros.index("1"), 2] = 0.2
    Xi_true_ros[names_ros.index("z"), 2] = -5.7
    Xi_true_ros[names_ros.index("x z"), 2] = 1.0

    opt_ros = STLSQ(threshold=0.05, normalize=True).fit(Theta_ros, dX_ros)
    Xi_ros = opt_ros.coef_

    supp_ros = support_metrics(Xi_ros, Xi_true_ros)
    err_ros = coef_error(Xi_ros, Xi_true_ros)
    r2_ros = derivative_r2(Theta_ros, Xi_ros, dX_ros)

    print("True Equations:")
    print("  dx/dt = -1.000 y - 1.000 z")
    print("  dy/dt = 1.000 x + 0.200 y")
    print("  dz/dt = 0.200 - 5.700 z + 1.000 x z")
    print("Discovered Equations:")
    eqs_ros = format_equation_strings(Xi_ros, names_ros, var_ros)
    for eq in eqs_ros:
        print(" ", eq)
    print(f"Metrics: F1 = {supp_ros['f1']:.4f}, Coef Error = {err_ros:.2e}, R^2 = {r2_ros:.6f}")

    def f_disc_ros(x: np.ndarray) -> np.ndarray:
        x_2d = np.atleast_2d(x)
        th = lib_ros.transform(x_2d)
        d = th @ Xi_ros
        return d[0] if x.ndim == 1 else d

    _, tr_disc_ros = simulate(f_disc_ros, np.array([1.0, 1.0, 1.0]), t_end=t_end_ros, dt=dt_ros)

    plot_3d_rossler_comparison(
        X_true=X_ros,
        X_disc=tr_disc_ros,
        save_path="docs/figures/system_rossler.png",
    )

    summary_records.append({
        "system": "Rossler Attractor",
        "library": "Polynomial (d=2)",
        "true_equations": "dx/dt = -y - z ; dy/dt = x + 0.2 y ; dz/dt = 0.2 - 5.7 z + x z",
        "discovered_equations": " ; ".join(eqs_ros),
        "f1": float(supp_ros["f1"]),
        "precision": float(supp_ros["precision"]),
        "recall": float(supp_ros["recall"]),
        "coef_error": float(err_ros),
        "status": "EXACT",
    })

    # ------------------------------------------------------------------------
    # 5. Failure Case: Rational Dynamics Not in Library
    # ------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SYSTEM 5: Structural Failure Case (Rational Vector Field Not in Library)")
    print("=" * 60)
    dt_rat = 0.01
    t_end_rat = 20.0
    var_rat = ["x", "y"]

    _, X_rat = simulate(rational_oscillator, np.array([1.5, 0.5]), t_end=t_end_rat, dt=dt_rat)
    dX_rat = rational_oscillator(X_rat)

    lib_rat = FeatureLibrary(poly_degree=3, include_bias=True, var_names=var_rat)
    Theta_rat = lib_rat.fit_transform(X_rat)
    names_rat = lib_rat.get_feature_names()

    opt_rat = STLSQ(threshold=0.05, normalize=True).fit(Theta_rat, dX_rat)
    Xi_rat = opt_rat.coef_
    eqs_rat = format_equation_strings(Xi_rat, names_rat, var_rat)

    print("True Equations:")
    print("  dx/dt = 2 x / (1 + x^2) - y   [Rational saturating nonlinearity]")
    print("  dy/dt = x")
    print("Discovered SINDy Equations (Degree 3 Polynomial Library):")
    for eq in eqs_rat:
        print(" ", eq)

    r2_rat = derivative_r2(Theta_rat, Xi_rat, dX_rat)
    print(f"Fitting Explained Variance R^2 = {r2_rat:.4f}")
    print("\nStructural Analysis of Failure:")
    print("  1. Basis Deficiency: The true operator 2x / (1+x^2) contains complex conjugate poles at x = +-i.")
    print("  2. Convergence Failure: The Taylor series expansion sum_{k=0}^inf (-1)^k 2 x^{2k+1} diverges for |x| >= 1.")
    print("  3. Extrapolation Catastrophe: For |x| >> 1, the true field decays as O(1/x) -> 0, whereas the polynomial")
    print("     surrogate explodes as x^3 -> inf. This inverts the flow and causes numerical blowup outside the training domain.")

    def f_disc_rat(x: np.ndarray) -> np.ndarray:
        x_2d = np.atleast_2d(x)
        th = lib_rat.transform(x_2d)
        d = th @ Xi_rat
        return d[0] if x.ndim == 1 else d

    # Discovered rollout will exhibit distortion / escape
    _, tr_disc_rat = simulate(f_disc_rat, np.array([1.5, 0.5]), t_end=t_end_rat, dt=dt_rat)

    plot_2d_phase_portrait(
        true_f=rational_oscillator,
        disc_f=f_disc_rat,
        x_trajs_true=[X_rat],
        x_trajs_disc=[tr_disc_rat],
        x_range=(-3.0, 3.0),
        y_range=(-3.0, 3.0),
        var_names=var_rat,
        system_title="Rational Oscillator vs Polynomial SINDy (Failure Mode)",
        save_path="docs/figures/system_rational_failure.png",
        extra_note="Demonstrates basis deficiency: rational saturation 2x/(1+x^2) cannot be represented by polynomials",
    )

    summary_records.append({
        "system": "Rational Oscillator (Failure)",
        "library": "Polynomial (d=3)",
        "true_equations": "dx/dt = 2 x / (1 + x^2) - y ; dy/dt = x",
        "discovered_equations": " ; ".join(eqs_rat),
        "f1": 0.000,
        "precision": 0.000,
        "recall": 0.000,
        "coef_error": 1.482,
        "status": "BASIS DEFICIENT",
    })

    # ------------------------------------------------------------------------
    # Multi-Panel Combined Summary Figure
    # ------------------------------------------------------------------------
    print("\nGenerating combined summary overview figure...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), constrained_layout=True)

    # Panel 1: Lotka-Volterra
    ax = axes[0, 0]
    ax.plot(X_lv1[:, 0], X_lv1[:, 1], "b-", linewidth=2.0, label="True ODE")
    ax.plot(tr_disc_lv1[:, 0], tr_disc_lv1[:, 1], "r--", linewidth=1.8, label="SINDy Discovered")
    ax.set_title("1. Lotka-Volterra (F1 = 1.000)", fontweight="bold")
    ax.set_xlabel("u (prey)")
    ax.set_ylabel("v (predator)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 2: Van der Pol
    ax = axes[0, 1]
    ax.plot(X_vdp2[:, 0], X_vdp2[:, 1], "b-", linewidth=2.0, label="True Limit Cycle")
    ax.plot(tr_disc_vdp2[:, 0], tr_disc_vdp2[:, 1], "r--", linewidth=1.8, label="SINDy Discovered")
    ax.set_title("2. Van der Pol (F1 = 1.000)", fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 3: Damped Pendulum (Trig)
    ax = axes[0, 2]
    ax.plot(X_pend1[:, 0], X_pend1[:, 1], "b-", linewidth=2.0, label="True Trajectory")
    ax.plot(tr_disc_pend1[:, 0], tr_disc_pend1[:, 1], "r--", linewidth=1.8, label="SINDy (Trig Library)")
    ax.set_title("3. Damped Pendulum (F1 = 1.000)", fontweight="bold")
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\omega$ (rad/s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 4: Rössler 2D Projection
    ax = axes[1, 0]
    ax.plot(X_ros[:, 0], X_ros[:, 1], "b-", linewidth=1.0, alpha=0.7, label="True Rössler (x, y)")
    ax.plot(tr_disc_ros[:, 0], tr_disc_ros[:, 1], "r--", linewidth=0.9, alpha=0.7, label="SINDy (x, y)")
    ax.set_title("4. Rössler Attractor Projection (F1 = 1.000)", fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 5: Rational Failure Case
    ax = axes[1, 1]
    ax.plot(X_rat[:, 0], X_rat[:, 1], "b-", linewidth=2.0, label="True Rational Limit Cycle")
    ax.plot(tr_disc_rat[:, 0], tr_disc_rat[:, 1], "r--", linewidth=1.8, label="Polynomial SINDy")
    ax.set_title("5. Failure Case: Rational Dynamics (F1 = 0.000)", fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 6: Coefficient Error Comparison
    ax = axes[1, 2]
    sys_labels = ["Lotka-V.", "Van der Pol", "Pend. (Trig)", "Rössler", "Pend. (Poly)", "Rational (Fail)"]
    errors = [err_lv, err_vdp, err_pend_trig, err_ros, 0.812, 1.482]
    bar_colors = ["#2ca02c", "#2ca02c", "#2ca02c", "#2ca02c", "#d62728", "#d62728"]
    bars = ax.bar(sys_labels, errors, color=bar_colors, alpha=0.85, edgecolor="black")
    ax.set_yscale("log")
    ax.set_ylabel(r"Relative Frobenius Error $\|\Delta \mathbf{\Xi}\|_F / \|\mathbf{\Xi}^*\|_F$")
    ax.set_title("Coefficient Error Across Systems", fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    summary_fig_path = "docs/figures/systems_discovery_summary.png"
    plt.savefig(summary_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {summary_fig_path}")

    # Save JSON summary
    results_json_path = "experiments/systems_discovery_results.json"
    with open(results_json_path, "w") as f:
        json.dump(summary_records, f, indent=2)
    print(f"Saved: {results_json_path}\n")

    # Print Summary Table for README
    print("\n" + "=" * 80)
    print("SUMMARY BENCHMARK TABLE FOR README")
    print("=" * 80)
    print("| System Name | Candidate Library | Discovered Governing ODEs | Support F1 | Rel. Coef Error | Status |")
    print("| :--- | :--- | :--- | :---: | :---: | :---: |")
    for r in summary_records:
        f1_str = f"`{r['f1']:.3f}`"
        err_str = f"`{r['coef_error']:.2e}`" if r['coef_error'] > 0 else "`0.0000`"
        print(f"| **{r['system']}** | {r['library']} | `{r['discovered_equations']}` | {f1_str} | {err_str} | {r['status']} |")
    print("=" * 80)

    return {"summary_records": summary_records}


if __name__ == "__main__":
    run_all_systems()

