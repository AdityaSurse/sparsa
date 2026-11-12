"""Candidate feature library generation for sparse regression (SINDy).

Builds polynomial and trigonometric candidate function basis matrices Theta(X)
using combinatorial monomial expansions.
"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np


class FeatureLibrary:
    """Nonlinear candidate basis feature library Theta(X) in R^(m x p).

    Constructs combinatorial polynomial interactions up to a specified degree,
    optional constant bias, and trigonometric basis functions.

    Parameters
    ----------
    poly_degree : int, default=2
        Maximum polynomial degree of monomials to generate.
    include_bias : bool, default=True
        Whether to include the constant unity feature '1' (degree 0).
    trig : bool, default=False
        Whether to append sin(x_i) and cos(x_i) terms for all variables.
    var_names : Optional[Sequence[str]], default=None
        Custom symbolic names for state variables. If None, defaults to
        ['x', 'y', 'z'] for n<=3, or ['x0', 'x1', ...] for general n.
    normalize : bool, default=False
        Whether to normalize columns of Theta to unit L2 norm during transform.
    """

    def __init__(
        self,
        poly_degree: int = 2,
        include_bias: bool = True,
        trig: bool = False,
        var_names: Optional[Sequence[str]] = None,
        normalize: bool = False,
    ) -> None:
        if poly_degree < 1:
            raise ValueError(f"poly_degree must be >= 1, got {poly_degree}")

        self.poly_degree = poly_degree
        self.include_bias = include_bias
        self.trig = trig
        self.var_names = list(var_names) if var_names is not None else None
        self.normalize = normalize

        # Attributes populated upon fitting
        self.n_input_features_: Optional[int] = None
        self.resolved_var_names_: List[str] = []
        self.terms_: List[Tuple[str, Union[Tuple[int, ...], int]]] = []
        self.feature_names_: List[str] = []
        self.column_norms_: Optional[np.ndarray] = None

    @property
    def n_features(self) -> int:
        """Total number of candidate basis functions p."""
        if not self.feature_names_:
            raise ValueError("FeatureLibrary has not been fitted yet. Call fit() first.")
        return len(self.feature_names_)

    def fit(self, X: np.ndarray) -> FeatureLibrary:
        """Fit the candidate library by defining basis terms for state dimension n.

        Polynomials are generated combinatorially via
        `itertools.combinations_with_replacement` over variable indices.

        Parameters
        ----------
        X : np.ndarray
            Input trajectory data matrix of shape (m, n) or vector of shape (m,).

        Returns
        -------
        FeatureLibrary
            The fitted library instance.
        """
        x_arr = np.asarray(X, dtype=np.float64)
        n = x_arr.shape[1] if x_arr.ndim == 2 else 1
        self.n_input_features_ = n

        # Resolve symbolic variable names
        if self.var_names is not None:
            if len(self.var_names) != n:
                raise ValueError(
                    f"var_names length ({len(self.var_names)}) must match state dimension ({n})"
                )
            self.resolved_var_names_ = list(self.var_names)
        else:
            if n == 1:
                self.resolved_var_names_ = ["x"]
            elif n == 2:
                self.resolved_var_names_ = ["x", "y"]
            elif n == 3:
                self.resolved_var_names_ = ["x", "y", "z"]
            else:
                self.resolved_var_names_ = [f"x{i}" for i in range(n)]

        self.terms_ = []
        self.feature_names_ = []

        # 1. Degree 0 (Bias term)
        if self.include_bias:
            self.terms_.append(("bias", ()))
            self.feature_names_.append("1")

        # 2. Polynomial monomials (Degrees 1 to poly_degree)
        for d in range(1, self.poly_degree + 1):
            for comb in itertools.combinations_with_replacement(range(n), d):
                self.terms_.append(("poly", comb))
                self.feature_names_.append(self._format_monomial(comb))

        # 3. Trigonometric basis terms
        if self.trig:
            for i in range(n):
                var = self.resolved_var_names_[i]
                self.terms_.append(("sin", i))
                self.feature_names_.append(f"sin({var})")
                self.terms_.append(("cos", i))
                self.feature_names_.append(f"cos({var})")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Evaluate candidate library functions on state observations X.

        Parameters
        ----------
        X : np.ndarray
            Input trajectory data matrix of shape (m, n) or vector of shape (m,).

        Returns
        -------
        np.ndarray
            Candidate feature matrix Theta(X) of shape (m, p).
        """
        if not self.terms_:
            raise ValueError("FeatureLibrary is not fitted. Call fit() before transform().")

        x_arr = np.asarray(X, dtype=np.float64)
        is_1d = (x_arr.ndim == 1)
        x_2d = np.atleast_2d(x_arr).T if is_1d else x_arr

        m, n = x_2d.shape
        if n != self.n_input_features_:
            raise ValueError(
                f"Dimension mismatch: expected {self.n_input_features_} features, got {n}"
            )

        p = len(self.terms_)
        Theta = np.empty((m, p), dtype=np.float64)

        for col_idx, (term_type, spec) in enumerate(self.terms_):
            if term_type == "bias":
                Theta[:, col_idx] = 1.0
            elif term_type == "poly":
                assert isinstance(spec, tuple)
                if len(spec) == 1:
                    Theta[:, col_idx] = x_2d[:, spec[0]]
                else:
                    Theta[:, col_idx] = np.prod(x_2d[:, spec], axis=1)
            elif term_type == "sin":
                assert isinstance(spec, int)
                Theta[:, col_idx] = np.sin(x_2d[:, spec])
            elif term_type == "cos":
                assert isinstance(spec, int)
                Theta[:, col_idx] = np.cos(x_2d[:, spec])

        if np.isnan(Theta).any() or np.isinf(Theta).any():
            raise FloatingPointError("Theta matrix contains NaN or Inf values")

        if self.normalize:
            norms = np.linalg.norm(Theta, axis=0)
            self.column_norms_ = norms
            scale_factors = np.where(norms == 0.0, 1.0, norms)
            Theta = Theta / scale_factors

        return Theta

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the library to X and return transformed matrix Theta(X)."""
        return self.fit(X).transform(X)

    def unnormalize_xi(self, Xi: np.ndarray) -> np.ndarray:
        """Rescale coefficient matrix Xi from normalized to original physical units.

        Since Theta_norm[:, j] = Theta[:, j] / w_j, the original coefficients satisfy
        Xi_orig[j, :] = Xi_norm[j, :] / w_j.

        Parameters
        ----------
        Xi : np.ndarray
            Coefficient matrix of shape (p, n) fitted against normalized Theta.

        Returns
        -------
        np.ndarray
            Rescaled coefficient matrix of shape (p, n).
        """
        if self.column_norms_ is None:
            return np.copy(Xi)

        Xi_arr = np.asarray(Xi, dtype=np.float64)
        scale_factors = np.where(self.column_norms_ == 0.0, 1.0, self.column_norms_)
        return Xi_arr / scale_factors[:, np.newaxis]

    def get_feature_names(self) -> List[str]:
        """Return list of human-readable feature name strings of length p."""
        if not self.feature_names_:
            raise ValueError("FeatureLibrary is not fitted. Call fit() first.")
        return list(self.feature_names_)

    def _format_monomial(self, comb: Tuple[int, ...]) -> str:
        """Format a variable index tuple into a human-readable monomial string."""
        counts = Counter(comb)
        parts = []
        for var_idx in sorted(counts.keys()):
            var_name = self.resolved_var_names_[var_idx]
            deg = counts[var_idx]
            if deg == 1:
                parts.append(var_name)
            else:
                parts.append(f"{var_name}^{deg}")
        return " ".join(parts)


def print_library(lib: FeatureLibrary) -> None:
    """Print the candidate basis feature menu as a formatted table.

    Parameters
    ----------
    lib : FeatureLibrary
        Fitted candidate feature library instance.
    """
    if not lib.feature_names_:
        print("FeatureLibrary has not been fitted yet.")
        return

    names = lib.get_feature_names()
    print("=" * 60)
    print(f"CANDIDATE FEATURE LIBRARY MENU (p = {len(names)})")
    print("=" * 60)
    print(f"| {'Index':<7} | {'Feature Name':<20} | {'Term Type':<16} |")
    print(f"| {':---':<7} | {':---':<20} | {':---':<16} |")

    for idx, (name, (term_type, spec)) in enumerate(zip(names, lib.terms_)):
        if term_type == "bias":
            type_desc = "Constant (d=0)"
        elif term_type == "poly":
            assert isinstance(spec, tuple)
            type_desc = f"Polynomial (d={len(spec)})"
        elif term_type == "sin":
            type_desc = "Trigonometric sin"
        elif term_type == "cos":
            type_desc = "Trigonometric cos"
        else:
            type_desc = term_type

        print(f"| {idx:<7} | {name:<20} | {type_desc:<16} |")
    print("=" * 60)

