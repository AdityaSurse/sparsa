"""Numerical differentiation benchmark and error analysis under measurement noise.

Evaluates Central Differences, Savitzky-Golay, and Total-Variation Regularized
differentiation on the chaotic Lorenz-63 system across noise levels [0, 0.001, 0.01, 0.05].

Generates comparison table and saves visualization to docs/figures/derivative_error.png.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple
import numpy as np

# Configure matplotlib headless cache directory before imports
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src is on Python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sindy.systems import lorenz
from sindy.simulate import simulate, add_noise
from sindy.derivatives import central_diff, savgol_diff, tv_diff


def run_derivative_experiment() -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Execute the multi-method differentiation benchmark."""
    dt = 0.01
    t_end = 12.0
    x0 = np.array([-8.0, 8.0, 27.0], dtype=np.float64)

    # Generate ground-truth trajectory and oracle derivatives
    t, X_clean = simulate(lorenz, x0, t_end=t_end, dt=dt)
    dX_true = lorenz(X_clean)

    noise_levels = [0.0, 0.001, 0.01, 0.05]
    metrics: List[Dict[str, float]] = []

    # Store estimates at noise=0.01 for time-domain overlay
    overlay_estimates: Dict[str, np.ndarray] = {}
    overlay_noise_level = 0.01

    for noise in noise_levels:
        X_noisy = add_noise(X_clean, noise_level=noise, seed=42)

        # 1. Central Differences (2nd order)
        dX_cd = central_diff(X_noisy, dt)

        # 2. Savitzky-Golay Filter (window=17, polyorder=3)
        dX_sg = savgol_diff(X_noisy, dt, window=17, polyorder=3)

        # 3. Total-Variation Regularized Differentiation
        alpha = max(1e-4, 0.05 * noise) if noise > 0.0 else 0.0
        dX_tv = tv_diff(X_noisy, dt, alpha=alpha, max_iter=35)

        # Compute Frobenius root-mean-squared error across all coordinates
        rmse_cd = float(np.sqrt(np.mean((dX_cd - dX_true) ** 2)))
        rmse_sg = float(np.sqrt(np.mean((dX_sg - dX_true) ** 2)))
        rmse_tv = float(np.sqrt(np.mean((dX_tv - dX_true) ** 2)))

        metrics.append({
            "noise": noise,
            "rmse_cd": rmse_cd,
            "rmse_sg": rmse_sg,
            "rmse_tv": rmse_tv,
        })

        if np.isclose(noise, overlay_noise_level):
            overlay_estimates["Central Difference"] = dX_cd
            overlay_estimates["Savitzky-Golay"] = dX_sg
            overlay_estimates["TV Regularized"] = dX_tv

    return metrics, t, dX_true, overlay_estimates


def print_results_table(metrics: List[Dict[str, float]]) -> None:
    """Print markdown-formatted results table."""
    header = (
        "| Noise Level | Noise Ratio (%) | Central Diff RMSE | Savitzky-Golay RMSE | TV Reg RMSE | SG vs CD Gain |"
    )
    separator = (
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    )
    print("\n" + "=" * 90)
    print("NUMERICAL DIFFERENTIATION BENCHMARK (Lorenz-63, dt=0.01)")
    print("=" * 90)
    print(header)
    print(separator)

    for m in metrics:
        noise = m["noise"]
        pct = noise * 100.0
        gain = m["rmse_cd"] / m["rmse_sg"] if m["rmse_sg"] > 0 else 1.0
        print(
            f"| {noise:.3f} | {pct:5.2f}% | {m['rmse_cd']:17.4f} | {m['rmse_sg']:19.4f} | "
            f"{m['rmse_tv']:11.4f} | {gain:11.2f}x |"
        )
    print("=" * 90 + "\n")


def plot_derivative_analysis(
    metrics: List[Dict[str, float]],
    t: np.ndarray,
    dX_true: np.ndarray,
    overlay: Dict[str, np.ndarray],
    output_path: str = "docs/figures/derivative_error.png",
) -> None:
    """Render and save 2-panel log-log RMSE error and time-domain overlay."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, (ax_err, ax_sig) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    fig.patch.set_facecolor("#0f1117")

    # ------------------------------------------------------------------------
    # Panel 1: Log-log RMSE vs Noise Level
    # ------------------------------------------------------------------------
    # Exclude noise=0 for pure log-log scaling, plot as horizontal asymptotes
    noisy_metrics = [m for m in metrics if m["noise"] > 0.0]
    noises = np.array([m["noise"] for m in noisy_metrics])
    cd_errors = np.array([m["rmse_cd"] for m in noisy_metrics])
    sg_errors = np.array([m["rmse_sg"] for m in noisy_metrics])
    tv_errors = np.array([m["rmse_tv"] for m in noisy_metrics])

    # Plot noise curves
    ax_err.loglog(noises, cd_errors, "s-", color="#ff7b72", linewidth=2.0, markersize=8, label="Central Differences (2nd order)")
    ax_err.loglog(noises, sg_errors, "o-", color="#79c0ff", linewidth=2.0, markersize=8, label="Savitzky-Golay (w=17, p=3)")
    ax_err.loglog(noises, tv_errors, "^-", color="#ffa657", linewidth=2.0, markersize=8, label="Total Variation Regularized")

    # Reference theoretical noise scaling line ~ sigma / dt
    ref_x = np.array([0.001, 0.05])
    ref_y = ref_x * (cd_errors[0] / noises[0])
    ax_err.loglog(ref_x, ref_y, ":", color="#8b949e", linewidth=1.5, label=r"Unfiltered Scaling $\sim \sigma / \Delta t$")

    ax_err.set_facecolor("#161b22")
    ax_err.grid(True, which="both", linestyle=":", color="#30363d", alpha=0.7)
    ax_err.set_xlabel(r"Noise Magnitude $\sigma$ ($\times \mathrm{std}(X)$)", color="#e6edf3", fontsize=11)
    ax_err.set_ylabel(r"Derivative RMSE ($\| \hat{\dot{\mathbf{X}}} - \dot{\mathbf{X}}^* \|_{\mathrm{F}}$)", color="#e6edf3", fontsize=11)
    ax_err.set_title("Differentiation Error Scaling vs Noise", color="#e6edf3", fontsize=12, pad=12)
    ax_err.tick_params(colors="#e6edf3", which="both")
    leg1 = ax_err.legend(facecolor="#0f1117", edgecolor="#30363d", fontsize=9, loc="upper left")
    for text in leg1.get_texts():
        text.set_color("#e6edf3")

    # ------------------------------------------------------------------------
    # Panel 2: Time-domain Overlay (dx/dt on a chaotic burst at 1% noise)
    # ------------------------------------------------------------------------
    ax_sig.set_facecolor("#161b22")
    # Focus on a chaotic transition window
    t_start, t_end = 4.0, 7.0
    mask = (t >= t_start) & (t <= t_end)
    t_win = t[mask]

    # Plot coordinate 0 (dx/dt)
    ax_sig.plot(t_win, overlay["Central Difference"][mask, 0], color="#ff7b72", alpha=0.55, linewidth=1.0, label="Central Diff (noisy)")
    ax_sig.plot(t_win, overlay["Savitzky-Golay"][mask, 0], color="#79c0ff", linewidth=1.8, label="Savitzky-Golay")
    ax_sig.plot(t_win, overlay["TV Regularized"][mask, 0], "--", color="#ffa657", linewidth=1.4, alpha=0.85, label="TV Regularized")
    ax_sig.plot(t_win, dX_true[mask, 0], color="#ffffff", linewidth=1.8, linestyle="-", label=r"True Oracle $\dot{x}(t)$")

    ax_sig.grid(True, linestyle=":", color="#30363d", alpha=0.7)
    ax_sig.set_xlabel("Time $t$ (seconds)", color="#e6edf3", fontsize=11)
    ax_sig.set_ylabel(r"Estimated Derivative $\dot{x}(t)$", color="#e6edf3", fontsize=11)
    ax_sig.set_title(r"Overlay: $\dot{x}(t)$ Tracking at 1% Measurement Noise", color="#e6edf3", fontsize=12, pad=12)
    ax_sig.tick_params(colors="#e6edf3")
    leg2 = ax_sig.legend(facecolor="#0f1117", edgecolor="#30363d", fontsize=9, loc="lower left")
    for text in leg2.get_texts():
        text.set_color("#e6edf3")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Benchmark figure saved to {output_path}")


if __name__ == "__main__":
    metrics, t, dX_true, overlay = run_derivative_experiment()
    print_results_table(metrics)
    plot_derivative_analysis(metrics, t, dX_true, overlay)

