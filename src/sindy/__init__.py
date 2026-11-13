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
from sindy.library import (
    FeatureLibrary,
    print_library,
)
from sindy.stlsq import (
    STLSQ,
    pretty_print_equations,
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
    "FeatureLibrary",
    "print_library",
    "STLSQ",
    "pretty_print_equations",
]

