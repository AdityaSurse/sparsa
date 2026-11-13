"""Sparsa weak-form module alias."""

from sindy.weak_form import *  # noqa: F401, F403
from sindy.weak_form import weak_test_function_kernels, weak_form_projection  # noqa: F401

__all__ = ["weak_test_function_kernels", "weak_form_projection"]

