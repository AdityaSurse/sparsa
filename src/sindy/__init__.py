"""SINDy / Sparsa dynamical systems simulation and equation discovery package."""

from sindy.systems import (
    DynamicalSystem,
    SYSTEMS,
    lorenz,
    lotka_volterra,
    van_der_pol,
    damped_pendulum,
    LORENZ_TRUE_COEFFICIENTS,
    LOTKA_VOLTERRA_TRUE_COEFFICIENTS,
    VAN_DER_POL_TRUE_COEFFICIENTS,
    DAMPED_PENDULUM_TRUE_COEFFICIENTS,
)
from sindy.simulate import (
    rk4_step,
    simulate,
    generate_dataset,
    add_noise,
)
from sindy.derivatives import (
    central_diff,
    savgol_diff,
    tv_diff,
)
from sindy.weak_form import (
    weak_test_function_kernels,
    weak_form_projection,
)
from sindy.library import (
    FeatureLibrary,
    print_library,
)
from sindy.stlsq import (
    STLSQ,
    pretty_print_equations,
)
from sindy.pipeline import SINDy
from sindy.metrics import (
    support_metrics,
    coef_error,
    derivative_r2,
    rollout_error,
    lyapunov_time_ratio,
    attractor_similarity,
)

__all__ = [
    "DynamicalSystem",
    "SYSTEMS",
    "lorenz",
    "lotka_volterra",
    "van_der_pol",
    "damped_pendulum",
    "LORENZ_TRUE_COEFFICIENTS",
    "LOTKA_VOLTERRA_TRUE_COEFFICIENTS",
    "VAN_DER_POL_TRUE_COEFFICIENTS",
    "DAMPED_PENDULUM_TRUE_COEFFICIENTS",
    "rk4_step",
    "simulate",
    "generate_dataset",
    "add_noise",
    "central_diff",
    "savgol_diff",
    "tv_diff",
    "weak_test_function_kernels",
    "weak_form_projection",
    "FeatureLibrary",
    "print_library",
    "STLSQ",
    "pretty_print_equations",
    "SINDy",
    "support_metrics",
    "coef_error",
    "derivative_r2",
    "rollout_error",
    "lyapunov_time_ratio",
    "attractor_similarity",
]

