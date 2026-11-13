"""Lorenz-63 governing equation discovery experiment via SINDy and Pareto analysis.

Executes:
1. Simulates 5 diverse chaotic trajectories with RK4 (dt=0.002, t_end=20.0).
2. Performs SINDy discovery using both exact Oracle derivatives and 2nd-order Central Differences.
3. Evaluates a degree-3 candidate library (p=20 monomials) across lambda in logspace(-3, 0, 25).
4. Identifies the Pareto front elbow, prints side-by-side equation comparisons and parameter error tables.
5. Generates 4 publication-quality visual diagnostics:
   - docs/figures/lorenz_pareto_front.png
   - docs/figures/lorenz_xi_heatmap.png
   - docs/figures/lorenz_attractor_comparison.png
   - docs/figures/lorenz_timeseries_overlay.png
6. Validates exact support recovery and asserts max relative coefficient error < 1%.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple
import numpy as np

# Matplotlib headless backend configuration
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add source directory to Python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sindy.systems import lorenz
from sindy.simulate import simulate
from sindy.derivatives import central_diff
from sindy.library import FeatureLibrary
from sindy.stlsq import STLSQ, pretty_print_equations
from sindy.pipeline import SINDy


def run_experiment() -> Tuple[SINDy, np.ndarray, np.ndarray, List[Dict[str, float]], List[Dict[str, float]]]:
    """Execute the full Lorenz equation discovery experiment."""
    print("=" * 80)
    print("SINDy LORENZ-63 EQUATION DISCOVERY & PARETO ANALYSIS")
    print("=" * 80)

    # 1. Simulate 5 trajectories with randomized initial states
    dt = 0.002
    t_end = 20.0
    n_traj = 5
    rng = np.random.default_rng(42)

    X_list: List[np.ndarray] = []
    dXdt_oracle_list: List[np.ndarray] = []
    dXdt_cd_list: List[np.ndarray] = []

    print(f"Simulating {n_traj} chaotic trajectories (dt={dt}, t_end={t_end}, steps={int(t_end/dt) + 1})...")
    for i in range(n_traj):
        x0 = rng.uniform(low=[-15.0, -15.0, 10.0], high=[15.0, 15.0, 40.0])
        t, X = simulate(lorenz, x0, t_end=t_end, dt=dt)
        dX_oracle = lorenz(X)
        dX_cd = central_diff(X, dt)

        X_list.append(X)
        dXdt_oracle_list.append(dX_oracle)
        dXdt_cd_list.append(dX_cd)

    X_all = np.vstack(X_list)
    dX_oracle_all = np.vstack(dXdt_oracle_list)
    dX_cd_all = np.vstack(dXdt_cd_list)
    total_snapshots = X_all.shape[0]
    print(f"Total dataset: {total_snapshots} snapshots stacked across {n_traj} trajectories.\n")

    # 2. Build candidate library of degree 3
    lib = FeatureLibrary(poly_degree=3, include_bias=True, var_names=["x", "y", "z"])
    Theta = lib.fit_transform(X_all)
    feature_names = lib.get_feature_names()
    p = len(feature_names)
    print(f"Candidate feature library Theta(X) shape: {Theta.shape} (p={p} basis functions).")

    # Define true Xi matrix for degree 3
    Xi_true = np.zeros((p, 3), dtype=np.float64)
    Xi_true[feature_names.index("x"), 0] = -10.0
    Xi_true[feature_names.index("y"), 0] = 10.0
    Xi_true[feature_names.index("x"), 1] = 28.0
    Xi_true[feature_names.index("y"), 1] = -1.0
    Xi_true[feature_names.index("x z"), 1] = -1.0
    Xi_true[feature_names.index("z"), 2] = -8.0 / 3.0
    Xi_true[feature_names.index("x y"), 2] = 1.0

    # 3. Sweep lambda over np.logspace(-3, 0, 25)
    lambdas = np.logspace(-3, 0, 25)
    print("Sweeping threshold lambda over 25 log-spaced values in [10^-3, 10^0]...")

    pareto_oracle: List[Dict[str, float]] = []
    models_oracle: List[STLSQ] = []
    for lam in lambdas:
        s_opt = STLSQ(threshold=lam, max_iter=25, normalize=True)
        s_opt.fit(Theta, dX_oracle_all)
        coef = s_opt.coef_
        assert coef is not None
        nz = int(np.count_nonzero(coef != 0.0))
        mse = float(np.mean((dX_oracle_all - Theta @ coef) ** 2))
        pareto_oracle.append({"lambda": float(lam), "terms": nz, "mse": mse})
        models_oracle.append(s_opt)

    pareto_cd: List[Dict[str, float]] = []
    models_cd: List[STLSQ] = []
    for lam in lambdas:
        s_opt = STLSQ(threshold=lam, max_iter=25, normalize=True)
        s_opt.fit(Theta, dX_cd_all)
        coef = s_opt.coef_
        assert coef is not None
        nz = int(np.count_nonzero(coef != 0.0))
        mse = float(np.mean((dX_cd_all - Theta @ coef) ** 2))
        pareto_cd.append({"lambda": float(lam), "terms": nz, "mse": mse})
        models_cd.append(s_opt)

    # 4. Pick elbow of Pareto front: model with minimum MSE among those with exact true term count (7 terms)
    elbow_idx = 0
    min_elbow_mse = float("inf")
    for idx, rec in enumerate(pareto_oracle):
        if rec["terms"] == 7 and rec["mse"] < min_elbow_mse:
            min_elbow_mse = rec["mse"]
            elbow_idx = idx

    selected_lambda = pareto_oracle[elbow_idx]["lambda"]
    best_model = models_oracle[elbow_idx]
    Xi_discovered = best_model.coef_
    assert Xi_discovered is not None

    # Construct end-to-end SINDy pipeline instance with discovered model
    sindy_model = SINDy(
        poly_degree=3,
        include_bias=True,
        var_names=["x", "y", "z"],
        threshold=selected_lambda,
        max_iter=25,
        normalize=True,
    )
    sindy_model.fit(X_list, dt=dt, dXdt_list=dXdt_oracle_list)

    # 5. Print Discovered vs True Equations
    print("\n" + "=" * 80)
    print(f"DISCOVERED vs TRUE GOVERNING EQUATIONS (Selected lambda = {selected_lambda:.4e})")
    print("=" * 80)

    true_eqs = pretty_print_equations(Xi_true, feature_names, ["x", "y", "z"], precision=3)
    disc_eqs = pretty_print_equations(Xi_discovered, feature_names, ["x", "y", "z"], precision=3)

    print(f"{'State Coordinate':<18} | {'True Physics ODE':<35} | {'SINDy Discovered ODE':<35}")
    print(f"{':---':<18} | {':---':<35} | {':---':<35}")
    for k, var in enumerate(["x", "y", "z"]):
        print(f"d{var}/dt{'':<12} | {true_eqs[k]:<35} | {disc_eqs[k]:<35}")
    print("=" * 80 + "\n")

    # 6. Print Detailed Coefficient Error Table
    print("=" * 80)
    print("ACTIVE COEFFICIENT PARAMETER RECOVERY TABLE")
    print("=" * 80)
    print(f"| {'Term':<10} | {'Coord':<7} | {'True Coef':<12} | {'Discovered Coef':<16} | {'Abs Error':<14} | {'Rel Error (%)':<14} | {'Status':<8} |")
    print(f"| {':---':<10} | {':---':<7} | {':---':<12} | {':---':<16} | {':---':<14} | {':---':<14} | {':---':<8} |")

    max_rel_error = 0.0
    for j, feat in enumerate(feature_names):
        for k, var in enumerate(["x", "y", "z"]):
            c_true = Xi_true[j, k]
            c_disc = Xi_discovered[j, k]
            if c_true != 0.0 or c_disc != 0.0:
                abs_err = abs(c_disc - c_true)
                if c_true != 0.0:
                    rel_err = (abs_err / abs(c_true)) * 100.0
                    if rel_err > max_rel_error:
                        max_rel_error = rel_err
                    status = "MATCH" if abs_err < 1e-4 else "DRIFT"
                else:
                    rel_err = float("nan")
                    status = "SPURIOUS"
                print(
                    f"| {feat:<10} | d{var}/dt  | {c_true:12.4f} | {c_disc:16.4f} | {abs_err:14.4e} | {rel_err:13.6f}% | {status:<8} |"
                )
    print("=" * 80 + "\n")

    # 7. Verification and Acceptance
    support_recovered = np.array_equal((Xi_discovered != 0.0), (Xi_true != 0.0))
    passed = support_recovered and (max_rel_error < 1.0)

    print("=" * 80)
    print("ACCEPTANCE VERIFICATION CRITERIA")
    print("=" * 80)
    print(f"  Exact Support Recovery (Zero Spurious, Zero Missed): {support_recovered}")
    print(f"  Maximum Relative Parameter Error:                    {max_rel_error:.6e}% (Threshold: < 1.000000%)")
    print(f"  Overall Validation Status:                           {'PASS' if passed else 'FAIL'}")
    print("=" * 80 + "\n")

    assert passed, f"Acceptance criteria failed! support={support_recovered}, max_rel_err={max_rel_error}%"

    return sindy_model, Xi_true, Xi_discovered, pareto_oracle, pareto_cd


def generate_figures(
    sindy_model: SINDy,
    Xi_true: np.ndarray,
    Xi_discovered: np.ndarray,
    pareto_oracle: List[Dict[str, float]],
    pareto_cd: List[Dict[str, float]],
    output_dir: str = "docs/figures",
) -> None:
    """Render and save the four required publication figures."""
    os.makedirs(output_dir, exist_ok=True)
    feature_names = sindy_model.feature_names_
    p = len(feature_names)

    # ------------------------------------------------------------------------
    # Figure A: Pareto Front (MSE vs Nonzero Terms)
    # ------------------------------------------------------------------------
    fig_a, (ax_p1, ax_p2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig_a.patch.set_facecolor("#0f1117")

    lams = [r["lambda"] for r in pareto_oracle]
    terms_o = [r["terms"] for r in pareto_oracle]
    mse_o = [max(r["mse"], 1e-28) for r in pareto_oracle]

    terms_cd = [r["terms"] for r in pareto_cd]
    mse_cd = [max(r["mse"], 1e-28) for r in pareto_cd]

    # Panel 1: Pareto frontier (MSE vs Nonzero Terms)
    ax_p1.set_facecolor("#161b22")
    ax_p1.semilogy(terms_cd, mse_cd, "s--", color="#ff7b72", alpha=0.7, label="Central Differences")
    ax_p1.semilogy(terms_o, mse_o, "o-", color="#79c0ff", linewidth=2, label="Oracle Derivatives")
    ax_p1.scatter([7], [min([r["mse"] for r in pareto_oracle if r["terms"] == 7])], color="#ffa657", s=130, zorder=10, label="Selected Elbow (7 terms)")
    ax_p1.set_xlabel("Active Basis Monomials (L0 Count)", color="#e6edf3", fontsize=11)
    ax_p1.set_ylabel("Derivative Mean Squared Error (MSE)", color="#e6edf3", fontsize=11)
    ax_p1.set_title("Pareto Frontier: Parsimony vs Reconstruction Error", color="#e6edf3", fontsize=12, pad=12)
    ax_p1.grid(True, linestyle=":", color="#30363d", alpha=0.7)
    ax_p1.tick_params(colors="#e6edf3")
    leg1 = ax_p1.legend(facecolor="#0f1117", edgecolor="#30363d", fontsize=9)
    for text in leg1.get_texts():
        text.set_color("#e6edf3")

    # Panel 2: Monomial count vs lambda
    ax_p2.set_facecolor("#161b22")
    ax_p2.semilogx(lams, terms_cd, "s--", color="#ff7b72", alpha=0.7, label="Central Diff Terms")
    ax_p2.semilogx(lams, terms_o, "o-", color="#79c0ff", linewidth=2, label="Oracle Terms")
    ax_p2.axhline(7, color="#ffa657", linestyle=":", linewidth=1.5, label="True Support (7 terms)")
    ax_p2.set_xlabel(r"Sparsity Threshold $\lambda$", color="#e6edf3", fontsize=11)
    ax_p2.set_ylabel("Number of Non-zero Terms", color="#e6edf3", fontsize=11)
    ax_p2.set_title(r"Support Pruning Stability vs Threshold $\lambda$", color="#e6edf3", fontsize=12, pad=12)
    ax_p2.grid(True, linestyle=":", color="#30363d", alpha=0.7)
    ax_p2.tick_params(colors="#e6edf3")
    leg2 = ax_p2.legend(facecolor="#0f1117", edgecolor="#30363d", fontsize=9)
    for text in leg2.get_texts():
        text.set_color("#e6edf3")

    plt.tight_layout()
    path_a = os.path.join(output_dir, "lorenz_pareto_front.png")
    plt.savefig(path_a, dpi=300, facecolor=fig_a.get_facecolor(), bbox_inches="tight")
    plt.close(fig_a)
    print(f"Saved: {path_a}")

    # ------------------------------------------------------------------------
    # Figure B: Coefficient Heatmaps (True Xi vs Discovered Xi)
    # ------------------------------------------------------------------------
    fig_b, (ax_h1, ax_h2) = plt.subplots(1, 2, figsize=(10, 8), dpi=300)
    fig_b.patch.set_facecolor("#0f1117")

    vmax = max(np.max(np.abs(Xi_true)), np.max(np.abs(Xi_discovered)))
    vmin = -vmax

    im1 = ax_h1.imshow(Xi_true, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
    ax_h1.set_title(r"Ground Truth Matrix $\mathbf{\Xi}^*$", color="#e6edf3", fontsize=12, pad=12)
    ax_h1.set_xticks([0, 1, 2])
    ax_h1.set_xticklabels([r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$"], color="#e6edf3", fontsize=11)
    ax_h1.set_yticks(range(p))
    ax_h1.set_yticklabels(feature_names, color="#e6edf3", fontsize=9)
    ax_h1.tick_params(colors="#e6edf3")

    im2 = ax_h2.imshow(Xi_discovered, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
    ax_h2.set_title(r"SINDy Discovered Matrix $\hat{\mathbf{\Xi}}$", color="#e6edf3", fontsize=12, pad=12)
    ax_h2.set_xticks([0, 1, 2])
    ax_h2.set_xticklabels([r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$"], color="#e6edf3", fontsize=11)
    ax_h2.set_yticks(range(p))
    ax_h2.set_yticklabels([])
    ax_h2.tick_params(colors="#e6edf3")

    # Annotate non-zero values on cells
    for j in range(p):
        for k in range(3):
            val_t = Xi_true[j, k]
            if abs(val_t) > 1e-3:
                ax_h1.text(k, j, f"{val_t:.1f}", ha="center", va="center", color="#000000" if abs(val_t) < 15 else "#ffffff", fontsize=8, weight="bold")
            val_d = Xi_discovered[j, k]
            if abs(val_d) > 1e-3:
                ax_h2.text(k, j, f"{val_d:.1f}", ha="center", va="center", color="#000000" if abs(val_d) < 15 else "#ffffff", fontsize=8, weight="bold")

    cbar = fig_b.colorbar(im2, ax=[ax_h1, ax_h2], shrink=0.7, pad=0.04)
    cbar.set_label("Coefficient Value", color="#e6edf3", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#e6edf3", labelcolor="#e6edf3")

    path_b = os.path.join(output_dir, "lorenz_xi_heatmap.png")
    plt.savefig(path_b, dpi=300, facecolor=fig_b.get_facecolor(), bbox_inches="tight")
    plt.close(fig_b)
    print(f"Saved: {path_b}")

    # ------------------------------------------------------------------------
    # Figure C: True vs Discovered 3D Attractor
    # ------------------------------------------------------------------------
    test_x0 = np.array([-8.0, 8.0, 27.0], dtype=np.float64)
    t_eval, X_true_sim = simulate(lorenz, test_x0, t_end=20.0, dt=0.002)
    _, X_disc_sim = sindy_model.predict(test_x0, t_end=20.0, dt=0.002)

    fig_c = plt.figure(figsize=(15, 7), dpi=300)
    fig_c.patch.set_facecolor("#0f1117")

    ax_c1 = fig_c.add_subplot(121, projection="3d")
    ax_c1.set_facecolor("#0f1117")
    ax_c1.plot(X_true_sim[:, 0], X_true_sim[:, 1], X_true_sim[:, 2], color="#79c0ff", linewidth=0.8, alpha=0.9)
    ax_c1.set_title("True Lorenz-63 Chaotic Attractor\n(Analytical Forward RK4)", color="#e6edf3", fontsize=11, pad=10)
    ax_c1.set_xlabel("x", color="#e6edf3")
    ax_c1.set_ylabel("y", color="#e6edf3")
    ax_c1.set_zlabel("z", color="#e6edf3")
    ax_c1.tick_params(colors="#e6edf3", labelsize=8)
    ax_c1.view_init(elev=24, azim=-62)

    ax_c2 = fig_c.add_subplot(122, projection="3d")
    ax_c2.set_facecolor("#0f1117")
    ax_c2.plot(X_disc_sim[:, 0], X_disc_sim[:, 1], X_disc_sim[:, 2], color="#ffa657", linewidth=0.8, alpha=0.9)
    ax_c2.set_title("Discovered SINDy ODE Attractor\n(Integrated Reconstructed Model)", color="#e6edf3", fontsize=11, pad=10)
    ax_c2.set_xlabel("x", color="#e6edf3")
    ax_c2.set_ylabel("y", color="#e6edf3")
    ax_c2.set_zlabel("z", color="#e6edf3")
    ax_c2.tick_params(colors="#e6edf3", labelsize=8)
    ax_c2.view_init(elev=24, azim=-62)

    path_c = os.path.join(output_dir, "lorenz_attractor_comparison.png")
    plt.savefig(path_c, dpi=300, facecolor=fig_c.get_facecolor(), bbox_inches="tight")
    plt.close(fig_c)
    print(f"Saved: {path_c}")

    # ------------------------------------------------------------------------
    # Figure D: Time Series Overlay (First 5 Time Units)
    # ------------------------------------------------------------------------
    mask_5 = (t_eval <= 5.0)
    t_5 = t_eval[mask_5]

    fig_d, axes = plt.subplots(3, 1, figsize=(12, 7.5), dpi=300, sharex=True)
    fig_d.patch.set_facecolor("#0f1117")

    coords = ["x(t)", "y(t)", "z(t)"]
    colors = ["#79c0ff", "#7ee787", "#ffa657"]

    for k in range(3):
        ax = axes[k]
        ax.set_facecolor("#161b22")
        ax.plot(t_5, X_true_sim[mask_5, k], color="#ffffff", linewidth=2.0, label="True Analytical" if k == 0 else None)
        ax.plot(t_5, X_disc_sim[mask_5, k], "--", color=colors[k], linewidth=1.8, label=f"SINDy Model {coords[k]}")
        ax.grid(True, linestyle=":", color="#30363d", alpha=0.7)
        ax.set_ylabel(coords[k], color="#e6edf3", fontsize=11)
        ax.tick_params(colors="#e6edf3")
        leg = ax.legend(loc="upper right", facecolor="#0f1117", edgecolor="#30363d", fontsize=9)
        for text in leg.get_texts():
            text.set_color("#e6edf3")

    axes[-1].set_xlabel("Integration Time t (seconds)", color="#e6edf3", fontsize=11)
    axes[0].set_title("Trajectory Forecasting Overlay: True vs SINDy Discovered ODEs (t in [0, 5])", color="#e6edf3", fontsize=12, pad=12)

    plt.tight_layout()
    path_d = os.path.join(output_dir, "lorenz_timeseries_overlay.png")
    plt.savefig(path_d, dpi=300, facecolor=fig_d.get_facecolor(), bbox_inches="tight")
    plt.close(fig_d)
    print(f"Saved: {path_d}\n")


if __name__ == "__main__":
    sindy_model, Xi_true, Xi_discovered, pareto_oracle, pareto_cd = run_experiment()
    generate_figures(sindy_model, Xi_true, Xi_discovered, pareto_oracle, pareto_cd)

