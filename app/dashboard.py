"""SPARSA: Interactive SINDy Nonlinear Dynamics Discovery Dashboard.

Interactive web application built with Streamlit for exploring sparse identification of
nonlinear dynamical systems across varying noise, sampling rates, candidate libraries,
and thresholding regimes.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sindy.systems import (
    lorenz,
    lotka_volterra,
    van_der_pol,
    damped_pendulum,
    rossler,
    LORENZ_TRUE_COEFFICIENTS,
    LOTKA_VOLTERRA_TRUE_COEFFICIENTS,
    VAN_DER_POL_TRUE_COEFFICIENTS,
    DAMPED_PENDULUM_TRUE_COEFFICIENTS,
    ROSSLER_TRUE_COEFFICIENTS,
)
from sindy.simulate import simulate, add_noise
from sindy.derivatives import central_diff, savgol_diff
from sindy.weak_form import weak_form_projection
from sindy.library import FeatureLibrary
from sindy.stlsq import STLSQ, pretty_print_equations
from sindy.metrics import support_metrics, coef_error, derivative_r2, rollout_error, lyapunov_time_ratio


# ============================================================================
# Streamlit Application Configuration & Page Styling
# ============================================================================

st.set_page_config(
    page_title="SPARSA | SINDy Discovery Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #212529;
    }
    .diagnostic-box {
        padding: 14px 18px;
        border-radius: 6px;
        margin-top: 10px;
        margin-bottom: 16px;
        border-left: 5px solid;
    }
    .diag-success {
        background-color: #e8f5e9;
        border-color: #2e7d32;
        color: #1b5e20;
    }
    .diag-warning {
        background-color: #fff8e1;
        border-color: #f57f17;
        color: #e65100;
    }
    .diag-danger {
        background-color: #ffebee;
        border-color: #c62828;
        color: #b71c1c;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# System Definitions and Registry
# ============================================================================

SYSTEM_METADATA = {
    "Lorenz-63": {
        "func": lorenz,
        "vars": ["x", "y", "z"],
        "default_ic": [5.0, 5.0, 20.0],
        "default_poly": 3,
        "default_trig": False,
        "default_dt": 0.005,
        "t_end": 15.0,
        "lambda_max": 0.906,
        "true_coefs": LORENZ_TRUE_COEFFICIENTS,
        "latex_true": [
            r"\dot{x} = -10\,x + 10\,y",
            r"\dot{y} = 28\,x - y - x\,z",
            r"\dot{z} = -\frac{8}{3}\,z + x\,y",
        ],
    },
    "Lotka-Volterra": {
        "func": lotka_volterra,
        "vars": ["u", "v"],
        "default_ic": [10.0, 5.0],
        "default_poly": 2,
        "default_trig": False,
        "default_dt": 0.01,
        "t_end": 20.0,
        "lambda_max": 0.0,
        "true_coefs": LOTKA_VOLTERRA_TRUE_COEFFICIENTS,
        "latex_true": [
            r"\dot{u} = 1.000\,u - 0.100\,u\,v",
            r"\dot{v} = -1.500\,v + 0.075\,u\,v",
        ],
    },
    "Van der Pol": {
        "func": van_der_pol,
        "vars": ["x", "y"],
        "default_ic": [1.0, 1.0],
        "default_poly": 3,
        "default_trig": False,
        "default_dt": 0.01,
        "t_end": 20.0,
        "lambda_max": 0.0,
        "true_coefs": VAN_DER_POL_TRUE_COEFFICIENTS,
        "latex_true": [
            r"\dot{x} = y",
            r"\dot{y} = -x + 2\,y - 2\,x^2\,y",
        ],
    },
    "Damped Pendulum": {
        "func": damped_pendulum,
        "vars": ["theta", "omega"],
        "default_ic": [2.0, 0.0],
        "default_poly": 1,
        "default_trig": True,
        "default_dt": 0.01,
        "t_end": 15.0,
        "lambda_max": 0.0,
        "true_coefs": DAMPED_PENDULUM_TRUE_COEFFICIENTS,
        "latex_true": [
            r"\dot{\theta} = \omega",
            r"\dot{\omega} = -0.250\,\omega - 5.000\,\sin(\theta)",
        ],
    },
    "Rossler Attractor": {
        "func": rossler,
        "vars": ["x", "y", "z"],
        "default_ic": [1.0, 1.0, 1.0],
        "default_poly": 2,
        "default_trig": False,
        "default_dt": 0.01,
        "t_end": 35.0,
        "lambda_max": 0.07,
        "true_coefs": ROSSLER_TRUE_COEFFICIENTS,
        "latex_true": [
            r"\dot{x} = -y - z",
            r"\dot{y} = x + 0.200\,y",
            r"\dot{z} = 0.200 - 5.700\,z + x\,z",
        ],
    },
}


# ============================================================================
# Cached Simulation and Feature Library Computation
# ============================================================================

@st.cache_data
def get_cached_trajectories(
    system_name: str,
    n_traj: int,
    t_end: float,
    dt: float,
    noise_level: float,
    seed: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Simulate clean and noisy multi-trajectory data with caching."""
    sys_meta = SYSTEM_METADATA[system_name]
    func = sys_meta["func"]
    ic_base = np.array(sys_meta["default_ic"], dtype=np.float64)
    n_vars = len(sys_meta["vars"])

    rng = np.random.default_rng(seed)
    clean_trajs = []
    noisy_trajs = []

    for i in range(n_traj):
        if i == 0:
            x0 = ic_base.copy()
        else:
            jitter = rng.uniform(-0.5, 0.5, size=n_vars) * (np.abs(ic_base) + 1.0) * 0.4
            x0 = ic_base + jitter

        _, X_clean = simulate(func, x0, t_end=t_end, dt=dt)
        clean_trajs.append(X_clean)

        if noise_level > 0.0:
            X_noisy = add_noise(X_clean, noise_level=noise_level, seed=seed * 100 + i)
        else:
            X_noisy = X_clean.copy()
        noisy_trajs.append(X_noisy)

    return clean_trajs, noisy_trajs


@st.cache_data
def get_cached_library_and_derivatives(
    system_name: str,
    noisy_trajs: List[np.ndarray],
    dt: float,
    diff_method: str,
    poly_degree: int,
    include_bias: bool,
    trig: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Construct candidate library Theta and target derivatives dX/dt."""
    var_names = SYSTEM_METADATA[system_name]["vars"]

    # 1. Fit library definitions on first trajectory
    lib = FeatureLibrary(
        poly_degree=poly_degree,
        include_bias=include_bias,
        trig=trig,
        var_names=var_names,
    )
    lib.fit(noisy_trajs[0])
    feature_names = lib.get_feature_names()

    # 2. Extract features and derivatives according to chosen method
    if diff_method == "weak_form":
        theta_list = []
        g_list = []
        for tr in noisy_trajs:
            th = lib.transform(tr)
            th_w, g_w = weak_form_projection(tr, th, dt=dt, window_sec=0.25, stride_ratio=0.25)
            theta_list.append(th_w)
            g_list.append(g_w)
        Theta = np.vstack(theta_list)
        Target = np.vstack(g_list)
    else:
        theta_list = [lib.transform(tr) for tr in noisy_trajs]
        if diff_method == "central_diff":
            deriv_list = [central_diff(tr, dt) for tr in noisy_trajs]
        elif diff_method == "savgol":
            w = max(9, int(np.round(0.15 / dt)))
            if w % 2 == 0:
                w += 1
            w = min(w, noisy_trajs[0].shape[0] - 2)
            if w % 2 == 0:
                w -= 1
            w = max(5, w)
            deriv_list = [savgol_diff(tr, dt, window=w, polyorder=3) for tr in noisy_trajs]
        else:
            raise ValueError(f"Unknown diff method: {diff_method}")

        Theta = np.vstack(theta_list)
        Target = np.vstack(deriv_list)

    return Theta, Target, feature_names, var_names


@st.cache_data
def compute_pareto_front(
    Theta: np.ndarray,
    Target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Pareto frontier over a logarithmically spaced threshold sweep."""
    lambdas = np.logspace(-3, 0.7, 30)
    residuals = []
    active_terms = []

    for lam in lambdas:
        opt = STLSQ(threshold=lam, max_iter=20, normalize=True)
        opt.fit(Theta, Target)
        res = float(np.linalg.norm(Target - Theta @ opt.coef_))
        active = int(np.count_nonzero(opt.coef_))
        residuals.append(res)
        active_terms.append(active)

    return lambdas, np.array(residuals), np.array(active_terms)


# ============================================================================
# LaTeX Formatting Helper
# ============================================================================

def format_discovered_latex(coef: np.ndarray, feature_names: List[str], var_names: List[str]) -> List[str]:
    """Format discovered coefficients into clean LaTeX equations."""
    latex_eqs = []
    dot_vars = [rf"\dot{{{v}}}" for v in var_names]

    for j, dot_var in enumerate(dot_vars):
        terms = []
        for i, name in enumerate(feature_names):
            val = coef[i, j]
            if abs(val) < 1e-4:
                continue

            sign = "-" if val < 0 else ("+" if terms else "")
            val_abs = abs(val)

            clean_name = name.replace(" ", r"\,").replace("*", r"\cdot ")
            if clean_name == "1":
                terms.append(f"{sign} {val_abs:.3f}")
            else:
                if abs(val_abs - 1.0) < 1e-4:
                    terms.append(f"{sign} {clean_name}")
                else:
                    terms.append(f"{sign} {val_abs:.3f}\\,{clean_name}")

        eq_rhs = " ".join(terms) if terms else "0"
        latex_eqs.append(f"{dot_var} = {eq_rhs}")

    return latex_eqs


# ============================================================================
# Sidebar Controls
# ============================================================================

with st.sidebar:
    st.title("⚡ SPARSA Controls")
    st.caption("Sparse Identification of Nonlinear Dynamics")

    system_name = st.selectbox("Dynamical System", list(SYSTEM_METADATA.keys()), index=0)
    meta = SYSTEM_METADATA[system_name]

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        noise_level = st.slider("Noise Level (σ)", min_value=0.0, max_value=0.10, value=0.00, step=0.005, format="%.3f")
    with col_s2:
        dt = st.selectbox("Time Step (dt)", [0.001, 0.002, 0.005, 0.01, 0.02], index=2)

    diff_method = st.selectbox(
        "Differentiation Method",
        ["central_diff", "savgol", "weak_form"],
        index=0,
        help="central_diff: 2nd-order stencils; savgol: local polynomial smoothing; weak_form: variational integration bypassing derivatives.",
    )

    n_traj = st.slider("Number of Trajectories", min_value=1, max_value=10, value=3, step=1)
    poly_degree = st.selectbox("Library Polynomial Degree", [1, 2, 3], index=meta["default_poly"] - 1)
    trig_terms = st.checkbox("Include Trigonometric Terms sin(x), cos(x)", value=meta["default_trig"])

    st.markdown("---")
    st.subheader("Sparsity Regularization")
    threshold = st.slider("Threshold λ (Cutoff)", min_value=0.001, max_value=2.00, value=0.10, step=0.01, format="%.3f")

    with st.expander("Advanced Simulation Settings"):
        t_end = st.number_input("Duration (t_end)", value=float(meta["t_end"]), min_value=5.0, max_value=50.0, step=5.0)
        seed = st.number_input("Random Seed", value=42, min_value=0, max_value=999999, step=1)


# ============================================================================
# Main Data Processing Pipeline
# ============================================================================

clean_trajs, noisy_trajs = get_cached_trajectories(
    system_name=system_name,
    n_traj=n_traj,
    t_end=t_end,
    dt=dt,
    noise_level=noise_level,
    seed=seed,
)

Theta, Target, feature_names, var_names = get_cached_library_and_derivatives(
    system_name=system_name,
    noisy_trajs=noisy_trajs,
    dt=dt,
    diff_method=diff_method,
    poly_degree=poly_degree,
    include_bias=True,
    trig=trig_terms,
)

# STLSQ Sparse Regression
opt = STLSQ(threshold=threshold, max_iter=25, normalize=True)
opt.fit(Theta, Target)
Xi_discovered = opt.coef_
n_features = len(feature_names)
n_vars = len(var_names)

# Build True Ground Truth Xi
Xi_true = np.zeros((n_features, n_vars), dtype=np.float64)
true_dict = meta["true_coefs"]
for j, v in enumerate(var_names):
    if v in true_dict:
        for term, true_val in true_dict[v].items():
            if term in feature_names:
                Xi_true[feature_names.index(term), j] = true_val

# Metrics computation
supp_dict = support_metrics(Xi_discovered, Xi_true)
rel_coef_err = coef_error(Xi_discovered, Xi_true)
r2_val = derivative_r2(Theta, Xi_discovered, Target)
cond_theta = float(np.linalg.cond(Theta))

# Out-of-sample forward rollout evaluation
lib_eval = FeatureLibrary(poly_degree=poly_degree, include_bias=True, trig=trig_terms, var_names=var_names)
lib_eval.fit(clean_trajs[0])

def discovered_field(x: np.ndarray) -> np.ndarray:
    x_2d = np.atleast_2d(x)
    th = lib_eval.transform(x_2d)
    d = th @ Xi_discovered
    return d[0] if x.ndim == 1 else d

eval_ic = np.array(meta["default_ic"], dtype=np.float64)
try:
    t_roll, err_roll, t_div = rollout_error(
        f_true=meta["func"],
        f_found=discovered_field,
        x0=eval_ic,
        t_end=min(10.0, t_end),
        dt=dt,
        divergence_threshold=0.10,
    )
    if meta["lambda_max"] > 0:
        lyap_ratio = lyapunov_time_ratio(t_div, lambda_max=meta["lambda_max"])
        lyap_str = f"{lyap_ratio:.2f} τ_L"
    else:
        lyap_str = f"{t_div:.2f} s"
except Exception:
    t_div = 0.0
    lyap_str = "Diverged (NaN/Inf)"


# ============================================================================
# Header & Metric Cards
# ============================================================================

st.title(f"SPARSA: {system_name} Equation Discovery")
st.markdown(
    "Parsimonious continuous dynamical system identification from trajectory data using "
    "**Sequentially Thresholded Least Squares (STLSQ)**."
)

m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
with m_col1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">Support F1 Score</div>'
        f'<div class="metric-value">{supp_dict["f1"]:.3f}</div></div>',
        unsafe_allow_html=True,
    )
with m_col2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">Relative Coef Error</div>'
        f'<div class="metric-value">{rel_coef_err:.2e}</div></div>',
        unsafe_allow_html=True,
    )
with m_col3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">Predictive Horizon</div>'
        f'<div class="metric-value">{lyap_str}</div></div>',
        unsafe_allow_html=True,
    )
with m_col4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">Derivative R²</div>'
        f'<div class="metric-value">{r2_val:.4f}</div></div>',
        unsafe_allow_html=True,
    )
with m_col5:
    active_terms = int(np.count_nonzero(Xi_discovered))
    true_terms = int(np.count_nonzero(Xi_true))
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">Active Terms (Found / True)</div>'
        f'<div class="metric-value">{active_terms} / {true_terms}</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# Rule-Based "Why Did It Fail?" Diagnostic Box
# ============================================================================

fp = int(supp_dict["fp"])
fn = int(supp_dict["fn"])

if supp_dict["f1"] >= 0.999 and rel_coef_err < 0.05:
    st.markdown(
        '<div class="diagnostic-box diag-success">'
        '<strong>Diagnostic Status: Optimal Model Recovery.</strong><br/>'
        'Exact topological sparsity support and parameter values recovered to numerical precision.'
        '</div>',
        unsafe_allow_html=True,
    )
elif fp > 0 and fn == 0:
    st.markdown(
        f'<div class="diagnostic-box diag-warning">'
        f'<strong>Diagnostic Alert: Over-Fitting (False Positives Detected).</strong><br/>'
        f'The discovered model contains <strong>{fp} spurious extra term(s)</strong>. '
        f'<em>Recommended Action:</em> Increase threshold λ (move slider right) to prune noise-correlated monomials, '
        f'or switch differentiation to <code>weak_form</code> / <code>savgol</code>.'
        f'</div>',
        unsafe_allow_html=True,
    )
elif fn > 0 and fp == 0:
    st.markdown(
        f'<div class="diagnostic-box diag-warning">'
        f'<strong>Diagnostic Alert: Under-Fitting (Missing True Physics).</strong><br/>'
        f'The model is missing <strong>{fn} true physical term(s)</strong>. '
        f'<em>Recommended Action:</em> Decrease threshold λ (move slider left) to allow true weak damping terms to survive.'
        f'</div>',
        unsafe_allow_html=True,
    )
elif cond_theta > 1e7:
    st.markdown(
        f'<div class="diagnostic-box diag-danger">'
        f'<strong>Diagnostic Alert: Severe Library Collinearity (cond(Θ) = {cond_theta:.1e}).</strong><br/>'
        f'High-degree candidate monomials cause ill-conditioning in least squares. '
        f'<em>Recommended Action:</em> Lower polynomial degree to 2 or stack more diverse trajectories.'
        f'</div>',
        unsafe_allow_html=True,
    )
elif noise_level > 0.02 and diff_method == "central_diff":
    st.markdown(
        '<div class="diagnostic-box diag-danger">'
        '<strong>Diagnostic Alert: Numerical Derivative Noise Explosion.</strong><br/>'
        'Central differences amplify high-frequency noise by O(1/dt). '
        '<em>Recommended Action:</em> Switch Differentiation Method to <code>weak_form</code> or <code>savgol</code>.'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="diagnostic-box diag-warning">'
        '<strong>Diagnostic Alert: Support Mismatch.</strong><br/>'
        'Both false positives and false negatives exist. Check if candidate basis contains the true dynamics '
        '(e.g. enable trigonometric functions for pendulum systems).'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# Main Visual Panels (Tabs)
# ============================================================================

tab1, tab2, tab3 = st.tabs([
    "📐 Equations & Coefficient Heatmap",
    "🌀 Phase Portrait & Rollout Comparison",
    "📉 Pareto Frontier Analysis",
])

# ----------------------------------------------------------------------------
# Tab 1: Equations & Heatmaps
# ----------------------------------------------------------------------------
with tab1:
    eq_col1, eq_col2 = st.columns(2)
    with eq_col1:
        st.subheader("True Governing ODEs")
        for eq_str in meta["latex_true"]:
            st.latex(eq_str)

    with eq_col2:
        st.subheader("Discovered Governing ODEs")
        disc_latex = format_discovered_latex(Xi_discovered, feature_names, var_names)
        for eq_str in disc_latex:
            st.latex(eq_str)

    st.markdown("---")
    st.subheader("Coefficient Matrix Heatmap: Discovered (Ξ) vs True (Ξ*)")

    fig_hm, (ax_hm1, ax_hm2) = plt.subplots(1, 2, figsize=(12, max(4, len(feature_names) * 0.35)), constrained_layout=True)
    vmax = max(1.0, float(np.max(np.abs(Xi_true))))

    im1 = ax_hm1.imshow(Xi_discovered, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_hm1.set_title("Discovered Matrix Ξ", fontweight="bold")
    ax_hm1.set_xticks(range(n_vars))
    ax_hm1.set_xticklabels(var_names)
    ax_hm1.set_yticks(range(n_features))
    ax_hm1.set_yticklabels(feature_names)

    # Cell annotations
    for i in range(n_features):
        for j in range(n_vars):
            val = Xi_discovered[i, j]
            if abs(val) > 1e-4:
                ax_hm1.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.5 * vmax else "black", fontsize=8)

    im2 = ax_hm2.imshow(Xi_true, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_hm2.set_title("Ground-Truth Matrix Ξ*", fontweight="bold")
    ax_hm2.set_xticks(range(n_vars))
    ax_hm2.set_xticklabels(var_names)
    ax_hm2.set_yticks(range(n_features))
    ax_hm2.set_yticklabels(feature_names)

    for i in range(n_features):
        for j in range(n_vars):
            val = Xi_true[i, j]
            if abs(val) > 1e-4:
                ax_hm2.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.5 * vmax else "black", fontsize=8)

    fig_hm.colorbar(im2, ax=[ax_hm1, ax_hm2], shrink=0.8, pad=0.04, label="Coefficient Magnitude")
    st.pyplot(fig_hm)
    plt.close()

# ----------------------------------------------------------------------------
# Tab 2: Phase Portrait & Trajectory Rollout
# ----------------------------------------------------------------------------
with tab2:
    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.subheader("Trajectory Observations")
        fig_tr = plt.figure(figsize=(7, 5.5))
        if n_vars == 3:
            ax_tr = fig_tr.add_subplot(1, 1, 1, projection="3d")
            for tr_c, tr_n in zip(clean_trajs, noisy_trajs):
                if noise_level > 0:
                    ax_tr.plot(tr_n[:, 0], tr_n[:, 1], tr_n[:, 2], color="gray", alpha=0.35, linewidth=0.6, label="Noisy" if tr_c is clean_trajs[0] else "")
                ax_tr.plot(tr_c[:, 0], tr_c[:, 1], tr_c[:, 2], color="#1f77b4", linewidth=1.2, label="Clean Trajectory" if tr_c is clean_trajs[0] else "")
            ax_tr.set_xlabel(var_names[0])
            ax_tr.set_ylabel(var_names[1])
            ax_tr.set_zlabel(var_names[2])
            ax_tr.set_title("3D State Space Trajectory", fontweight="bold")
            ax_tr.legend(loc="upper right")
        else:
            ax_tr = fig_tr.add_subplot(1, 1, 1)
            for tr_c, tr_n in zip(clean_trajs, noisy_trajs):
                if noise_level > 0:
                    ax_tr.plot(tr_n[:, 0], tr_n[:, 1], color="gray", alpha=0.4, linewidth=0.8, label="Noisy Data" if tr_c is clean_trajs[0] else "")
                ax_tr.plot(tr_c[:, 0], tr_c[:, 1], color="#1f77b4", linewidth=1.6, label="True Flow" if tr_c is clean_trajs[0] else "")
            ax_tr.set_xlabel(var_names[0])
            ax_tr.set_ylabel(var_names[1])
            ax_tr.set_title("2D Phase Portrait", fontweight="bold")
            ax_tr.grid(True, linestyle="--", alpha=0.4)
            ax_tr.legend(loc="upper right")

        st.pyplot(fig_tr)
        plt.close()

    with p_col2:
        st.subheader("Forward Rollout Forecast")
        fig_ro, ax_ro = plt.subplots(figsize=(7, 5.5))
        try:
            _, X_true_sim = simulate(meta["func"], eval_ic, t_end=min(8.0, t_end), dt=dt)
            t_grid, X_disc_sim = simulate(discovered_field, eval_ic, t_end=min(8.0, t_end), dt=dt)

            ax_ro.plot(t_grid, X_true_sim[:, 0], "b-", linewidth=2.0, label=f"True {var_names[0]}(t)")
            ax_ro.plot(t_grid, X_disc_sim[:, 0], "r--", linewidth=1.8, label=f"Discovered {var_names[0]}(t)")
            if n_vars >= 2:
                ax_ro.plot(t_grid, X_true_sim[:, 1], "c-", linewidth=1.5, alpha=0.7, label=f"True {var_names[1]}(t)")
                ax_ro.plot(t_grid, X_disc_sim[:, 1], "m--", linewidth=1.5, alpha=0.7, label=f"Discovered {var_names[1]}(t)")

            ax_ro.set_xlabel("Time t (s)")
            ax_ro.set_ylabel("State Magnitude")
            ax_ro.set_title("Forward Forecast Comparison", fontweight="bold")
            ax_ro.grid(True, linestyle="--", alpha=0.4)
            ax_ro.legend(loc="upper right")
        except Exception:
            ax_ro.text(0.5, 0.5, "Discovered system encountered divergence / overflow", ha="center", va="center", color="red")

        st.pyplot(fig_ro)
        plt.close()

# ----------------------------------------------------------------------------
# Tab 3: Pareto Frontier
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Sparsity-Residual Pareto Front (Threshold Tuning)")
    st.caption("Sweeping λ across candidate cutoffs. The active threshold is marked by the golden star.")

    pareto_lams, pareto_res, pareto_terms = compute_pareto_front(Theta, Target)

    fig_pf, (ax_pf1, ax_pf2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Active terms vs lambda
    ax_pf1.semilogx(pareto_lams, pareto_terms, "b-o", linewidth=1.8, markersize=5)
    ax_pf1.axvline(threshold, color="gold", linestyle="--", linewidth=2.0, label=f"Selected λ = {threshold:.3f}")
    ax_pf1.plot(threshold, active_terms, "*", color="gold", markersize=14, markeredgecolor="black")
    ax_pf1.set_xlabel("Threshold λ (log scale)")
    ax_pf1.set_ylabel("Active Monomials Count")
    ax_pf1.set_title("Model Complexity vs Threshold", fontweight="bold")
    ax_pf1.grid(True, linestyle="--", alpha=0.4)
    ax_pf1.legend()

    # Pareto Front: Complexity vs Residual
    ax_pf2.plot(pareto_terms, pareto_res, "k-s", linewidth=1.8, markersize=5)
    current_res = float(np.linalg.norm(Target - Theta @ Xi_discovered))
    ax_pf2.plot(active_terms, current_res, "*", color="gold", markersize=14, markeredgecolor="black", label="Current Operating Point")
    ax_pf2.set_xlabel("Active Terms (Model Complexity)")
    ax_pf2.set_ylabel("Derivative Residual ||dX/dt - ΘΞ||_F")
    ax_pf2.set_title("Pareto Front (Complexity vs Error)", fontweight="bold")
    ax_pf2.grid(True, linestyle="--", alpha=0.4)
    ax_pf2.legend()

    st.pyplot(fig_pf)
    plt.close()

